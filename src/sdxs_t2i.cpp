#include "sdxs_t2i.h"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>

using namespace godot;

namespace {

// The three parts of an export, matched as fragments with the part's own underscore in front:
// an export ships them under whatever prefix it was written with, and the fragment is what lets
// a folder taken from anywhere load without somebody renaming its files first.
const char *CLIP_MARK = "_clip";
const char *UNET_MARK = "_unet";
const char *DEC_MARK = "_dec";
const char *PARAM_SUFFIX = ".ncnn.param";
const char *BIN_SUFFIX = ".ncnn.bin";

} // namespace

SdxsT2I::~SdxsT2I() {
    unload();
}

bool SdxsT2I::_load_graphs(const String &model_dir, const Dictionary &manifest, int num_threads,
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

    // The text encoder's blobs stay single precision when the manifest says so. Its residual
    // stream runs into the hundreds by the last block, and the square of that overflows half
    // precision inside a normalisation -- which comes back as a picture of something else
    // rather than as an error. It is a flag per graph for exactly this reason.
    if (!read_pair(files, model_dir, CLIP_MARK, "text encoder", clip, num_threads,
                !text_encoder_fp32, problem)) {
        return false;
    }
    if (!read_pair(files, model_dir, DEC_MARK, "decoder", dec, num_threads, true, problem)) {
        return false;
    }

    // Every size the manifest offers has to have a structure beside the weights, and it is
    // checked here rather than on the first picture at that size: a folder half exported
    // should refuse at the load a host can act on, not three prompts later.
    for (const std::pair<int, int> &size : sizes) {
        if (pick(files, mark_for(size.first, size.second), PARAM_SUFFIX).is_empty()) {
            problem = String("Govorilka: the manifest offers {0}x{1} and there is no "
                             "*{2}{3} in \"{4}\".")
                              .format(Array::make(size.first, size.second,
                                      mark_for(size.first, size.second), PARAM_SUFFIX, model_dir));
            return false;
        }
    }

    // The weights are read once here, with the structure for the first size the manifest
    // offers. Every other size re-reads the structure over these same bytes.
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

// One graph of the folder, named in the sentence it refuses with. The fragment is what a person
// then goes looking for, so the sentence carries the fragment rather than a whole file name
// nobody wrote down.
bool SdxsT2I::read_pair(const PackedStringArray &files, const String &model_dir, const char *mark,
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

void SdxsT2I::_unload_graphs() {
    clip.clear();
    unet.clear();
    dec.clear();
    folder = String();
    unet_width = 0;
    unet_height = 0;
}

String SdxsT2I::mark_for(int width, int height) {
    return String("_unet_") + String::num_int64(width) + String("x") + String::num_int64(height);
}

// The structure for this size, read over the weights already in memory. A folder whose sizes are
// two structures and one weight file costs one read of the weights and a re-parse per change;
// holding both structures at once would cost the repacked weights twice, which is most of the
// model.
bool SdxsT2I::_prepare_size(int width, int height, String &problem) {
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

bool SdxsT2I::_encode_text(const int *ids, int count, ncnn::Mat &hidden, String &problem) {
    // The ids go in as whole numbers. The lookup reads the blob's four bytes as an int rather
    // than converting them, so a float index is a bit pattern past the last row -- and out of
    // range is clamped, never reported: the picture is of nothing in particular.
    ncnn::Mat window = owned_indices(ids, count);
    ncnn::Extractor ex = clip.net.create_extractor();
    if (ex.input("in0", window) != 0 || ex.extract("out0", hidden) != 0) {
        problem = String("Govorilka: the text encoder refused this prompt.");
        return false;
    }
    return true;
}

bool SdxsT2I::_denoise(const ncnn::Mat &latent, const ncnn::Mat &hidden, float timestep,
        int width, int height, ncnn::Mat &out, String &problem) {
    // The timestep is a constant inside this export, so the schedule's value is not passed on.
    // A family whose graph takes one uses it here instead.
    (void)timestep;
    ncnn::Extractor ex = unet.net.create_extractor();
    if (ex.input("in0", latent) != 0 || ex.input("in1", hidden) != 0
            || ex.extract("out0", out) != 0) {
        problem = String("Govorilka: the denoiser refused a {0}x{1} picture.")
                          .format(Array::make(width, height));
        return false;
    }
    return true;
}

bool SdxsT2I::_decode_latent(const ncnn::Mat &latent, int width, int height, ncnn::Mat &rgb,
        String &problem) {
    // One structure for every size: the decoder is convolutions and scalings all the way down,
    // so its shapes follow the blob it is handed rather than the shape it was traced at.
    ncnn::Extractor ex = dec.net.create_extractor();
    if (ex.input("in0", latent) != 0 || ex.extract("out0", rgb) != 0) {
        problem = String("Govorilka: the decoder refused a {0}x{1} latent.")
                          .format(Array::make(width, height));
        return false;
    }
    return true;
}

void SdxsT2I::_report_timings(Dictionary &out) const {
    out["unet_swap_ms"] = swap_ms;
    out["unet_width"] = unet_width;
    out["unet_height"] = unet_height;
}

String SdxsT2I::describe() const {
    return family.is_empty() ? String("one-step diffusion (ncnn)")
                             : family + String(" (ncnn)");
}

void SdxsT2I::_bind_methods() {
}
