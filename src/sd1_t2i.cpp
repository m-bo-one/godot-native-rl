#include "sd1_t2i.h"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>

using namespace godot;

namespace {

const char *CLIP_MARK = "_clip";
const char *UNET_MARK = "_unet";
const char *DEC_MARK = "_dec";
const char *PARAM_SUFFIX = ".ncnn.param";
const char *BIN_SUFFIX = ".ncnn.bin";

} // namespace

Sd1T2I::~Sd1T2I() {
    unload();
}

bool Sd1T2I::_load_graphs(const String &model_dir, const Dictionary &manifest, int num_threads,
        String &problem) {
    (void)manifest;
    folder = model_dir;
    Ref<DirAccess> dir = DirAccess::open(model_dir);
    if (dir.is_null()) {
        problem = String("Govorilka: \"{0}\" would not open.").format(Array::make(model_dir));
        return false;
    }
    PackedStringArray files = dir->get_files();
    files.sort();

    // Single precision for the text encoder's blobs when the manifest asks: its residual stream
    // overflows half precision inside a normalisation, and the result is a picture of something
    // else with no error anywhere.
    if (!read_pair(files, model_dir, CLIP_MARK, "text encoder", clip, num_threads,
                !text_encoder_fp32, problem)) {
        return false;
    }
    if (!read_pair(files, model_dir, DEC_MARK, "decoder", dec, num_threads, true, problem)) {
        return false;
    }

    for (const std::pair<int, int> &size : sizes) {
        if (pick(files, mark_for(size.first, size.second), PARAM_SUFFIX).is_empty()) {
            problem = String("Govorilka: the manifest offers {0}x{1} and there is no *{2}{3} "
                             "in \"{4}\".")
                              .format(Array::make(size.first, size.second,
                                      mark_for(size.first, size.second), PARAM_SUFFIX, model_dir));
            return false;
        }
    }

    const String unet_bin = pick(files, UNET_MARK, BIN_SUFFIX);
    if (unet_bin.is_empty()) {
        problem = String("Govorilka: there is no *{0}{1} in \"{2}\", so the denoiser has no "
                         "weights.")
                          .format(Array::make(UNET_MARK, BIN_SUFFIX, model_dir));
        return false;
    }
    const String unet_param = pick(files, mark_for(sizes[0].first, sizes[0].second),
            PARAM_SUFFIX);
    unet.prepare(num_threads);
    if (!unet.read(model_dir.path_join(unet_param), model_dir.path_join(unet_bin))) {
        problem = String("Govorilka: the denoiser would not load from \"{0}\" and \"{1}\".")
                          .format(Array::make(unet_param, unet_bin));
        return false;
    }
    unet_width = sizes[0].first;
    unet_height = sizes[0].second;
    return true;
}

bool Sd1T2I::read_pair(const PackedStringArray &files, const String &model_dir, const char *mark,
        const char *what, NcnnGraph &into, int num_threads, bool fp16, String &problem) {
    const String param = pick(files, mark, PARAM_SUFFIX);
    const String weights = pick(files, mark, BIN_SUFFIX);
    if (param.is_empty() || weights.is_empty()) {
        problem = String("Govorilka: the {0}'s files are not in \"{1}\": nothing there is named "
                         "*{2}{3} and *{2}{4}.")
                          .format(Array::make(what, model_dir, mark, PARAM_SUFFIX, BIN_SUFFIX));
        return false;
    }
    into.prepare(num_threads, fp16);
    if (!into.read(model_dir.path_join(param), model_dir.path_join(weights))) {
        problem = String("Govorilka: the {0} would not load from \"{1}\" and \"{2}\".")
                          .format(Array::make(what, param, weights));
        return false;
    }
    return true;
}

void Sd1T2I::_unload_graphs() {
    clip.clear();
    unet.clear();
    dec.clear();
    folder = String();
    unet_width = 0;
    unet_height = 0;
}

String Sd1T2I::mark_for(int width, int height) {
    return String("_unet_") + String::num_int64(width) + String("x") + String::num_int64(height);
}

bool Sd1T2I::_prepare_size(int width, int height, String &problem) {
    swap_ms = 0.0;
    if (unet_width == width && unet_height == height) {
        return true;
    }
    Ref<DirAccess> dir = DirAccess::open(folder);
    if (dir.is_null()) {
        problem = String("Govorilka: the model folder is no longer readable, so the {0}x{1} "
                         "structure cannot be read.")
                          .format(Array::make(width, height));
        return false;
    }
    PackedStringArray files = dir->get_files();
    files.sort();
    const String param = pick(files, mark_for(width, height), PARAM_SUFFIX);
    if (param.is_empty()) {
        problem = String("Govorilka: this folder offers {0}x{1} and has no structure file for "
                         "it. The export is incomplete.")
                          .format(Array::make(width, height));
        return false;
    }

    const double at = now_ms();
    if (!unet.reread(folder.path_join(param), threads)) {
        unet_width = 0;
        unet_height = 0;
        problem = String("Govorilka: the {0}x{1} structure would not read over these weights.")
                          .format(Array::make(width, height));
        return false;
    }
    swap_ms = now_ms() - at;
    unet_width = width;
    unet_height = height;
    return true;
}

bool Sd1T2I::_encode_text(const int *ids, int count, ncnn::Mat &hidden, String &problem) {
    ncnn::Mat window = owned_indices(ids, count);
    ncnn::Extractor ex = clip.net.create_extractor();
    if (ex.input("in0", window) != 0 || ex.extract("out0", hidden) != 0) {
        problem = String("Govorilka: the text encoder refused this prompt.");
        return false;
    }
    return true;
}

// Three inputs where the one-step family has two: the latent, the timestep the schedule is at,
// and the conditioning. The timestep travels as one float in a blob of its own, which is what
// the trace was given, and it is the whole of the difference between the two families.
bool Sd1T2I::_denoise(const ncnn::Mat &latent, const ncnn::Mat &hidden, float timestep,
        int width, int height, ncnn::Mat &out, String &problem) {
    ncnn::Mat step = owned(&timestep, 1, 1);
    if (step.empty()) {
        problem = String("Govorilka: there was no room for the timestep.");
        return false;
    }
    ncnn::Extractor ex = unet.net.create_extractor();
    if (ex.input("in0", latent) != 0 || ex.input("in1", step) != 0
            || ex.input("in2", hidden) != 0 || ex.extract("out0", out) != 0) {
        problem = String("Govorilka: the denoiser refused a {0}x{1} picture at step {2}.")
                          .format(Array::make(width, height, timestep));
        return false;
    }
    return true;
}

bool Sd1T2I::_decode_latent(const ncnn::Mat &latent, int width, int height, ncnn::Mat &rgb,
        String &problem) {
    ncnn::Extractor ex = dec.net.create_extractor();
    if (ex.input("in0", latent) != 0 || ex.extract("out0", rgb) != 0) {
        problem = String("Govorilka: the decoder refused a {0}x{1} latent.")
                          .format(Array::make(width, height));
        return false;
    }
    return true;
}

void Sd1T2I::_report_timings(Dictionary &out) const {
    out["unet_swap_ms"] = swap_ms;
    out["unet_width"] = unet_width;
    out["unet_height"] = unet_height;
}

String Sd1T2I::describe() const {
    return family.is_empty() ? String("latent diffusion (ncnn)") : family + String(" (ncnn)");
}

void Sd1T2I::_bind_methods() {
}
