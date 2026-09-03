#include "ncnn_asr.h"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/callable_method_pointer.hpp>

#include <cstring>

using namespace godot;

namespace {

// The rate every family here is fed at. A clip recorded at another one is stretched to it,
// because a clip read at the wrong rate is words at a wrong pitch.
constexpr int SAMPLE_RATE = 16000;

// Takes the busy flag in one atomic step or reports that another path holds it, and gives it
// back unless the turn was handed on. Reading the flag and raising it separately lets two
// callers both start a worker, and assigning a thread over a joinable one is std::terminate().
struct BusyGuard {
    std::atomic<bool> *held = nullptr;

    explicit BusyGuard(std::atomic<bool> &flag) {
        bool expected = false;
        if (flag.compare_exchange_strong(expected, true)) {
            held = &flag;
        }
    }

    ~BusyGuard() {
        if (held != nullptr) {
            held->store(false);
        }
    }

    BusyGuard(const BusyGuard &) = delete;
    BusyGuard &operator=(const BusyGuard &) = delete;

    bool taken() const { return held != nullptr; }

    // The flag stays raised and this stops owning it: the delivery on the main thread is
    // what lowers it, which is what keeps the graphs taken until the answer has gone out.
    void hand_on() { held = nullptr; }
};

} // namespace

bool NcnnGraph::load(const String &param_path, const String &bin_path, int num_threads) {
    clear();
    net.opt.num_threads = num_threads;
    net.opt.use_vulkan_compute = false;

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

void NcnnGraph::clear() {
    net.clear();
    param = PackedByteArray();
    weights = PackedByteArray();
}

// The worker is joined and nothing else: the graphs belong to the subclass, whose destructor
// has already run by the time this one does, so a virtual call from here would reach a table
// that is gone. A subclass calls unload() in its own destructor for its own graphs.
NcnnASR::~NcnnASR() {
    epoch.fetch_add(1);
    join_worker();
}

// The folder is opened through the engine rather than by a library, so a model inside an
// exported pack loads exactly as a folder beside the game does. A load while a turn is in
// flight would swap the graphs under the worker, so the worker is joined first.
bool NcnnASR::load(const String &model_dir, const String &language, int num_threads) {
    unload();
    const double started = now_ms();
    threads = num_threads > 0 ? num_threads : 1;

    Ref<DirAccess> dir = DirAccess::open(model_dir);
    if (dir.is_null()) {
        return false;
    }
    if (!_load_graphs(model_dir, language, threads)) {
        _unload_graphs();
        return false;
    }

    loaded = true;
    load_ms = now_ms() - started;
    return true;
}

String NcnnASR::transcribe(const PackedFloat32Array &samples, int sample_rate) {
    BusyGuard guard(busy);
    if (!guard.taken() || !loaded || samples.is_empty()) {
        return String();
    }
    return run(samples, sample_rate);
}

bool NcnnASR::transcribe_async(const PackedFloat32Array &samples, int sample_rate) {
    BusyGuard guard(busy);
    if (!guard.taken() || !loaded || samples.is_empty()) {
        return false;
    }
    join_worker();
    // A thread that could not be started answers false with the flag given back, rather than
    // letting the system error unwind into the engine, which has no handler for one.
    try {
        worker = std::thread(&NcnnASR::work, this, samples, sample_rate, epoch.load());
    } catch (...) {
        return false;
    }
    guard.hand_on();
    return true;
}

bool NcnnASR::is_busy() const {
    return busy.load();
}

bool NcnnASR::is_loaded() const {
    return loaded;
}

void NcnnASR::unload() {
    epoch.fetch_add(1);
    join_worker();
    busy.store(false);
    loaded = false;
    _unload_graphs();
}

// What the last clip cost, so a project can measure this road rather than trust a number
// written down somewhere. The numbers are of one decode: reading them while another runs
// answers the one before it.
Dictionary NcnnASR::last_timings() const {
    Dictionary out;
    out["load_ms"] = load_ms;
    out["total_ms"] = total_ms;
    _report_timings(out);
    return out;
}

String NcnnASR::describe_family() const {
    return "ncnn";
}

// One clip through the family's graphs, at the rate they take. The decode is fenced: an
// exception out of it would unwind into the engine, which has no handler and dies, where an
// empty answer is a clip the host is told held nothing.
String NcnnASR::run(const PackedFloat32Array &samples, int sample_rate) {
    const double started = now_ms();

    const float *source = samples.ptr();
    const int count = samples.size();
    std::vector<float> audio;
    if (sample_rate == SAMPLE_RATE || sample_rate <= 0) {
        audio.assign(source, source + count);
    } else {
        const double step = (double)sample_rate / (double)SAMPLE_RATE;
        const int taken = (int)((double)count / step);
        audio.resize((size_t)(taken > 0 ? taken : 0));
        for (int i = 0; i < taken; i++) {
            const double at = (double)i * step;
            const int left = (int)at;
            const int right = left + 1 < count ? left + 1 : left;
            const float fraction = (float)(at - (double)left);
            audio[(size_t)i] = source[left] * (1.0f - fraction) + source[right] * fraction;
        }
    }

    String text;
    try {
        text = _decode(audio);
    } catch (...) {
        text = String();
    }
    total_ms = now_ms() - started;
    return text;
}

// The worker's whole life: decode, then hand the text to the main thread. Emitting from
// here instead would put a signal on a thread the engine's listeners are not written for.
void NcnnASR::work(PackedFloat32Array samples, int sample_rate, int64_t at) {
    pending_text = run(samples, sample_rate);
    callable_mp(this, &NcnnASR::deliver).call_deferred(at);
}

// The delivery, on the main thread. A turn whose model was unloaded or replaced while it
// ran is dropped: the flag it would clear belongs to whatever was started after it.
void NcnnASR::deliver(int64_t at) {
    if (at != epoch.load() || !busy.load()) {
        return;
    }
    busy.store(false);
    emit_signal("transcribed", pending_text);
}

void NcnnASR::join_worker() {
    if (worker.joinable()) {
        worker.join();
    }
}

ncnn::Mat NcnnASR::owned(const float *source, int w, int h) {
    ncnn::Mat mat;
    mat.create(w, h);
    if (!mat.empty() && source != nullptr) {
        memcpy(mat.data, source, (size_t)w * (size_t)h * sizeof(float));
    }
    return mat;
}

ncnn::Mat NcnnASR::owned_index(int value) {
    ncnn::Mat mat;
    mat.create(1);
    if (!mat.empty()) {
        ((int *)mat.data)[0] = value;
    }
    return mat;
}

String NcnnASR::pick(const PackedStringArray &files, const String &mark, const String &suffix) {
    for (int i = 0; i < files.size(); i++) {
        const String lower = files[i].to_lower();
        if (lower.contains(mark) && lower.ends_with(suffix)) {
            return files[i];
        }
    }
    return String();
}

double NcnnASR::now_ms() {
    return (double)Time::get_singleton()->get_ticks_usec() / 1000.0;
}

void NcnnASR::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load", "model_dir", "language", "num_threads"), &NcnnASR::load);
    ClassDB::bind_method(D_METHOD("transcribe", "samples", "sample_rate"), &NcnnASR::transcribe);
    ClassDB::bind_method(D_METHOD("transcribe_async", "samples", "sample_rate"),
            &NcnnASR::transcribe_async);
    ClassDB::bind_method(D_METHOD("is_busy"), &NcnnASR::is_busy);
    ClassDB::bind_method(D_METHOD("is_loaded"), &NcnnASR::is_loaded);
    ClassDB::bind_method(D_METHOD("unload"), &NcnnASR::unload);
    ClassDB::bind_method(D_METHOD("last_timings"), &NcnnASR::last_timings);
    ClassDB::bind_method(D_METHOD("describe_family"), &NcnnASR::describe_family);

    ADD_SIGNAL(MethodInfo("transcribed", PropertyInfo(Variant::STRING, "text")));
}
