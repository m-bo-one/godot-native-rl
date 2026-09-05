#include "ncnn_graph.h"

#include "ncnn_report.h"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/time.hpp>

#include <cstring>

using namespace godot;

namespace {

// The reader the weights are loaded through, which the library does not supply: its own over
// memory carries no end. It answers every request with the bytes at the cursor and reports that
// it read them all, so a structure asking for more weights than the file holds -- one file cut
// short by a copy, one structure paired with the wrong weights -- is read off the end of the
// buffer, and the repack then works on whatever was after it. This one counts: a request that
// would pass the end is a short read, which is the one answer the loader does check.
class BoundedReader : public ncnn::DataReader {
public:
    BoundedReader(const unsigned char *from, size_t length) :
            base(from), size(length) {
    }

    size_t read(void *into, size_t wanted) const override {
        if (wanted > size - at) {
            return 0;
        }
        memcpy(into, base + at, wanted);
        at += wanted;
        return wanted;
    }

    // The blob is handed out as a pointer into these bytes rather than copied, which is why the
    // buffer is held for the graph's whole life and why the end has to be counted here.
    size_t reference(size_t wanted, const void **into) const override {
        if (wanted > size - at) {
            return 0;
        }
        *into = base + at;
        at += wanted;
        return wanted;
    }

private:
    const unsigned char *base;
    size_t size;
    mutable size_t at = 0;
};

} // namespace

void NcnnGraph::prepare(int num_threads, bool fp16_storage) {
    clear();
    net.opt.num_threads = num_threads;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_packed = fp16_storage;
    net.opt.use_fp16_storage = fp16_storage;
    net.opt.use_fp16_arithmetic = fp16_storage;
}

// The files are read through the engine rather than by the library, so a model inside an
// exported pack loads exactly as a folder beside the game does.
bool NcnnGraph::read(const String &param_path, const String &bin_path) {
    ncnn_report::note(String("reading the files of ") + param_path.get_file());
    param = FileAccess::get_file_as_bytes(param_path);
    weights = FileAccess::get_file_as_bytes(bin_path);
    if (param.is_empty() || weights.is_empty()) {
        return false;
    }
    ncnn_report::note(String("parsing the structure of ") + param_path.get_file());
    // The parser reads the structure as a C string and stops at the first zero byte, which a
    // file has no reason to end with. Appended here rather than trusted to the reader.
    param.append(0);
    if (net.load_param_mem((const char *)param.ptr()) != 0) {
        clear();
        return false;
    }
    // Read through a DataReader, not through load_model(const unsigned char *): that overload
    // answers how many bytes it CONSUMED and throws the loader's own status away, so a load
    // that stopped part-way -- a layer whose weights or whose pipeline could not be allocated
    // under memory pressure -- still answers a large non-zero number. Read as success it
    // leaves the net holding the layers past the failure with neither, and the first extract
    // walks into them. The buffer stays here because every weight is a pointer into it.
    ncnn_report::note(String("loading the weights of ") + param_path.get_file());
    BoundedReader reader((const unsigned char *)weights.ptr(), (size_t)weights.size());
    if (net.load_model(reader) != 0) {
        clear();
        return false;
    }
    ncnn_report::note(String("loaded ") + param_path.get_file());
    return true;
}

bool NcnnGraph::load(const String &param_path, const String &bin_path, int num_threads) {
    prepare(num_threads);
    return read(param_path, bin_path);
}

// The weight buffer is taken out of the way before the net is cleared and put back after, so
// the structure is parsed against bytes that never left memory. Every layer of the new net
// aliases them exactly as the old one did, which is why they may not be freed in between.
bool NcnnGraph::reread(const String &param_path, int num_threads, bool fp16_storage) {
    if (weights.is_empty()) {
        return false;
    }
    PackedByteArray held = weights;
    PackedByteArray next = FileAccess::get_file_as_bytes(param_path);
    if (next.is_empty()) {
        return false;
    }
    next.append(0);

    net.clear();
    net.opt.num_threads = num_threads;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_packed = fp16_storage;
    net.opt.use_fp16_storage = fp16_storage;
    net.opt.use_fp16_arithmetic = fp16_storage;
    param = next;
    weights = held;

    ncnn_report::note(String("re-parsing the structure of ") + param_path.get_file());
    if (net.load_param_mem((const char *)param.ptr()) != 0) {
        clear();
        return false;
    }
    ncnn_report::note(String("re-loading the weights under ") + param_path.get_file());
    BoundedReader reader((const unsigned char *)weights.ptr(), (size_t)weights.size());
    if (net.load_model(reader) != 0) {
        clear();
        return false;
    }
    ncnn_report::note(String("re-loaded ") + param_path.get_file());
    return true;
}

void NcnnGraph::clear() {
    net.clear();
    param = PackedByteArray();
    weights = PackedByteArray();
}

ncnn::Mat ncnn_util::owned(const float *source, int w, int h) {
    ncnn::Mat mat;
    mat.create(w, h);
    if (!mat.empty() && source != nullptr) {
        memcpy(mat.data, source, (size_t)w * (size_t)h * sizeof(float));
    }
    return mat;
}

ncnn::Mat ncnn_util::owned_index(int value) {
    ncnn::Mat mat;
    mat.create(1);
    if (!mat.empty()) {
        ((int *)mat.data)[0] = value;
    }
    return mat;
}

ncnn::Mat ncnn_util::owned_indices(const int *source, int count) {
    ncnn::Mat mat;
    mat.create(count);
    if (!mat.empty() && source != nullptr) {
        memcpy(mat.data, source, (size_t)count * sizeof(int));
    }
    return mat;
}

String ncnn_util::pick(const PackedStringArray &files, const String &mark, const String &suffix) {
    for (int i = 0; i < files.size(); i++) {
        const String lower = files[i].to_lower();
        if (lower.contains(mark) && lower.ends_with(suffix)) {
            return files[i];
        }
    }
    return String();
}

double ncnn_util::now_ms() {
    return (double)Time::get_singleton()->get_ticks_usec() / 1000.0;
}
