#include "ncnn_asr.h"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/callable_method_pointer.hpp>

#include <cstring>

// The two fences below are a try around a decode and a catch around a thread that could not
// start; compiled without exceptions they are dead code and either fault unwinds into the
// engine. Refused here rather than discovered on a machine that could not start the thread.
#if !defined(_CPPUNWIND) && !defined(__EXCEPTIONS) && !defined(__cpp_exceptions)
#error "ncnn_asr.cpp needs C++ exceptions: build with disable_exceptions=no"
#endif

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

// The worker is joined and nothing else: the graphs belong to the subclass, whose destructor
// has already run by the time this one does, so a virtual call from here would reach a table
// that is gone. A subclass calls unload() in its own destructor for its own graphs.
NcnnASR::~NcnnASR() {
    epoch.fetch_add(1);
    join_worker();
}

// The folder is opened through the engine rather than by a library, so a model inside an
// exported pack loads exactly as a folder beside the game does. A load while a turn is in
// flight would swap the graphs under the worker, so the worker is joined first, and the graphs
// are read under the same lock a decode on any thread holds.
bool NcnnASR::load(const String &model_dir, const String &language, int num_threads) {
    unload();
    const double started = now_ms();
    threads = num_threads > 0 ? num_threads : 1;

    Ref<DirAccess> dir = DirAccess::open(model_dir);
    if (dir.is_null()) {
        return false;
    }
    std::lock_guard<std::mutex> hold(graphs_lock);
    if (!_load_graphs(model_dir, language, threads)) {
        _unload_graphs();
        return false;
    }

    loaded.store(true);
    load_ms = now_ms() - started;
    return true;
}

String NcnnASR::transcribe(const PackedFloat32Array &samples, int sample_rate) {
    BusyGuard guard(busy);
    if (!guard.taken() || !loaded.load() || samples.is_empty()) {
        return String();
    }
    return run(samples, sample_rate);
}

bool NcnnASR::transcribe_async(const PackedFloat32Array &samples, int sample_rate) {
    BusyGuard guard(busy);
    if (!guard.taken() || !loaded.load() || samples.is_empty()) {
        return false;
    }
    join_worker();
    // Marked owed before the thread exists: a delivery that raced this line would otherwise
    // lower the flag first and have the mark set over it afterwards.
    owed.store(true);
    // A thread that could not be started answers false with the flag given back, rather than
    // letting the system error unwind into the engine, which has no handler for one.
    try {
        worker = std::thread(&NcnnASR::work, this, samples, sample_rate, epoch.load());
    } catch (...) {
        owed.store(false);
        return false;
    }
    guard.hand_on();
    return true;
}

bool NcnnASR::is_busy() const {
    return busy.load();
}

bool NcnnASR::is_loaded() const {
    return loaded.load();
}

// The worker is joined and then the lock is taken, in that order: the worker's decode holds
// the lock, so taking it first would wait on a thread that is waiting to be joined. The
// blocking caller's flag is left to its own guard; only the worker's is lowered here.
void NcnnASR::unload() {
    epoch.fetch_add(1);
    join_worker();
    {
        std::lock_guard<std::mutex> hold(graphs_lock);
        loaded.store(false);
        _unload_graphs();
    }
    if (owed.exchange(false)) {
        busy.store(false);
    }
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

double NcnnASR::last_no_speech_prob() const {
    return -1.0;
}

String NcnnASR::last_detected_language() const {
    return String();
}

double NcnnASR::last_language_prob() const {
    return 0.0;
}

Array NcnnASR::last_language_candidates() const {
    return Array();
}

// One clip through the family's graphs, at the rate they take, under the lock that keeps the
// graphs in place until it returns. The decode is fenced: an exception out of it would unwind
// into the engine, which has no handler and dies, where an empty answer is a clip the host is
// told held nothing.
String NcnnASR::run(const PackedFloat32Array &samples, int sample_rate) {
    std::lock_guard<std::mutex> hold(graphs_lock);
    // Read again with the lock held: an unload() between the busy flag and this line has
    // already freed the graphs, and the answer to that is nothing rather than a decode.
    if (!loaded.load()) {
        return String();
    }
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
    owed.store(false);
    busy.store(false);
    emit_signal("transcribed", pending_text);
}

void NcnnASR::join_worker() {
    if (worker.joinable()) {
        worker.join();
    }
}

ncnn::Mat NcnnASR::owned(const float *source, int w, int h) {
    return ncnn_util::owned(source, w, h);
}

ncnn::Mat NcnnASR::owned_index(int value) {
    return ncnn_util::owned_index(value);
}

String NcnnASR::pick(const PackedStringArray &files, const String &mark, const String &suffix) {
    return ncnn_util::pick(files, mark, suffix);
}

double NcnnASR::now_ms() {
    return ncnn_util::now_ms();
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
    ClassDB::bind_method(D_METHOD("last_no_speech_prob"), &NcnnASR::last_no_speech_prob);
    ClassDB::bind_method(D_METHOD("last_detected_language"), &NcnnASR::last_detected_language);
    ClassDB::bind_method(D_METHOD("last_language_prob"), &NcnnASR::last_language_prob);
    ClassDB::bind_method(D_METHOD("last_language_candidates"),
            &NcnnASR::last_language_candidates);

    ADD_SIGNAL(MethodInfo("transcribed", PropertyInfo(Variant::STRING, "text")));
}
