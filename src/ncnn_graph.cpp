#include "ncnn_graph.h"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/time.hpp>

#include <cstring>

using namespace godot;

void NcnnGraph::prepare(int num_threads) {
    clear();
    net.opt.num_threads = num_threads;
    net.opt.use_vulkan_compute = false;
}

// The files are read through the engine rather than by the library, so a model inside an
// exported pack loads exactly as a folder beside the game does.
bool NcnnGraph::read(const String &param_path, const String &bin_path) {
    param = FileAccess::get_file_as_bytes(param_path);
    weights = FileAccess::get_file_as_bytes(bin_path);
    if (param.is_empty() || weights.is_empty()) {
        return false;
    }
    // The parser reads the structure as a C string and stops at the first zero byte, which a
    // file has no reason to end with. Appended here rather than trusted to the reader.
    param.append(0);
    if (net.load_param_mem((const char *)param.ptr()) != 0) {
        return false;
    }
    // Answers how many bytes it took, and zero means it took none. The buffer stays in this
    // object because every weight in the graph is a pointer into it rather than a copy.
    return net.load_model((const unsigned char *)weights.ptr()) != 0;
}

bool NcnnGraph::load(const String &param_path, const String &bin_path, int num_threads) {
    prepare(num_threads);
    return read(param_path, bin_path);
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
