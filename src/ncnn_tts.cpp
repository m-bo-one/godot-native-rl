#include "ncnn_tts.h"

#include "ncnn_report.h"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/callable_method_pointer.hpp>

#include <cmath>

// The two fences below are a try around a run and a catch around a thread that could not
// start; compiled without exceptions they are dead code and either fault unwinds into the
// engine. Refused here rather than discovered on a machine that could not start the thread.
#if !defined(_CPPUNWIND) && !defined(__EXCEPTIONS) && !defined(__cpp_exceptions)
#error "ncnn_tts.cpp needs C++ exceptions: build with disable_exceptions=no"
#endif

using namespace godot;

namespace {

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

    void hand_on() { held = nullptr; }
};

const char *NO_MODEL = "Govorilka: no model is loaded, so there is nothing to speak with.";
const char *NO_IDS = "Govorilka: the sentence came to no symbols at all, so there is nothing "
                     "to say. A line of punctuation alone reads as silence.";

} // namespace

TtsNoise::TtsNoise(uint64_t value) :
        state(value != 0 ? value : (uint64_t)Time::get_singleton()->get_ticks_usec() * 2654435761ULL + 1ULL) {
}

// splitmix64, then the top 24 bits as a fraction: taking the low bits instead reads the
// weakest part of the word, and dividing the whole 64 by 2^64 loses to rounding at 1.0.
float TtsNoise::uniform() {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (float)((z >> 40) * (1.0 / 16777216.0));
}

// Box-Muller, both halves of a pair used. The log's argument is pushed off zero, because a
// draw of exactly zero is an infinity that spreads through the whole utterance.
float TtsNoise::normal() {
    if (has_spare) {
        has_spare = false;
        return spare;
    }
    const float u1 = uniform() + 1.0f / 16777216.0f;
    const float u2 = uniform();
    const float radius = sqrtf(-2.0f * logf(u1));
    const float angle = 6.283185307179586f * u2;
    spare = radius * sinf(angle);
    has_spare = true;
    return radius * cosf(angle);
}

// The worker is joined and nothing else: the graphs belong to the subclass, whose destructor
// has already run by the time this one does, so a virtual call from here would reach a table
// that is gone. A subclass calls unload() in its own destructor for its own graphs.
NcnnTTS::~NcnnTTS() {
    epoch.fetch_add(1);
    join_worker();
}

// The folder is opened through the engine rather than by a library, so a model inside an
// exported pack loads exactly as a folder beside the game does. A load while a sentence is in
// flight would swap the graphs under the worker, so the worker is joined first, and the graphs
// are read under the same lock a run on any thread holds.
bool NcnnTTS::load(const String &model_dir, int num_threads) {
    unload();
    const double started = now_ms();
    threads = num_threads > 0 ? num_threads : 1;

    Ref<DirAccess> dir = DirAccess::open(model_dir);
    if (dir.is_null()) {
        return false;
    }
    std::lock_guard<std::mutex> hold(graphs_lock);
    if (!_load_graphs(model_dir, threads)) {
        _unload_graphs();
        return false;
    }

    loaded.store(true);
    load_ms = now_ms() - started;
    return true;
}

PackedFloat32Array NcnnTTS::synthesise(const PackedInt32Array &ids, int speaker) {
    BusyGuard guard(busy);
    if (!guard.taken()) {
        return PackedFloat32Array();
    }
    dropped.store(false);
    PackedFloat32Array samples = run(ids, speaker, problem);
    return samples;
}

bool NcnnTTS::synthesise_async(const PackedInt32Array &ids, int speaker) {
    BusyGuard guard(busy);
    if (!guard.taken()) {
        // The sentence in flight has already been thrown away, so what is left of it is only
        // in the way: joined here, on the road nobody is waiting on frames for.
        if (!dropped.load()) {
            return false;
        }
        join_worker();
        owed_at.store(-1);
        if (owed.exchange(false)) {
            busy.store(false);
        }
        BusyGuard again(busy);
        if (!again.taken()) {
            return false;
        }
        again.hand_on();
    } else {
        guard.hand_on();
    }

    dropped.store(false);
    const int64_t at = epoch.load();
    // The worker before this one has ended -- its delivery is what gave the flag back -- but a
    // thread that has ended is still joinable, and assigning over a joinable thread is
    // std::terminate(): the process dies with no message of any kind. Joined on every road in,
    // and on the ordinary one it has already finished so this costs nothing.
    join_worker();
    // Marked owed before the thread exists: a delivery that raced this line would otherwise
    // lower the flag first and have the mark set over it afterwards.
    owed_at.store(at);
    owed.store(true);
    // A thread that could not be started answers false with the flag given back, rather than
    // letting the system error unwind into the engine, which has no handler for one.
    try {
        worker = std::thread(&NcnnTTS::work, this, ids, speaker, at);
    } catch (...) {
        owed.store(false);
        owed_at.store(-1);
        busy.store(false);
        return false;
    }
    return true;
}

// The epoch moves and nothing waits. The worker keeps the graphs until it finishes and its
// delivery frees them; what it made is dropped on the way out rather than spoken.
void NcnnTTS::cancel() {
    if (!busy.load()) {
        return;
    }
    epoch.fetch_add(1);
    dropped.store(true);
}

bool NcnnTTS::is_busy() const {
    return busy.load();
}

bool NcnnTTS::is_loaded() const {
    return loaded.load();
}

// The worker is joined and then the lock is taken, in that order: the worker's run holds the
// lock, so taking it first would wait on a thread that is waiting to be joined. The blocking
// caller's flag is left to its own guard; only the worker's is lowered here.
void NcnnTTS::unload() {
    epoch.fetch_add(1);
    join_worker();
    {
        std::lock_guard<std::mutex> hold(graphs_lock);
        loaded.store(false);
        _unload_graphs();
    }
    owed_at.store(-1);
    if (owed.exchange(false)) {
        busy.store(false);
    }
    dropped.store(false);
}

String NcnnTTS::last_problem() const {
    return problem;
}

void NcnnTTS::set_seed(int64_t value) {
    seed = (uint64_t)value;
}

int64_t NcnnTTS::get_seed() const {
    return (int64_t)seed;
}

Dictionary NcnnTTS::last_timings() const {
    Dictionary out;
    out["load_ms"] = load_ms;
    out["total_ms"] = total_ms;
    _report_timings(out);
    return out;
}

String NcnnTTS::describe_family() const {
    return "ncnn";
}

int NcnnTTS::sample_rate() const {
    return 0;
}

int NcnnTTS::speaker_count() const {
    return 1;
}

// One sentence through the family's graphs, under the lock that keeps the graphs in place
// until it returns. The run is fenced: an exception out of it would unwind into the engine,
// which has no handler and dies, where no samples is a sentence the host is told about.
PackedFloat32Array NcnnTTS::run(const PackedInt32Array &ids, int speaker, String &said) {
    std::lock_guard<std::mutex> hold(graphs_lock);
    // Read again with the lock held: an unload() between the busy flag and this line has
    // already freed the graphs, and the answer to that is nothing rather than a run.
    if (!loaded.load()) {
        said = String(NO_MODEL);
        return PackedFloat32Array();
    }
    if (ids.is_empty()) {
        said = String(NO_IDS);
        return PackedFloat32Array();
    }
    const double started = now_ms();

    said = String();
    PackedFloat32Array samples;
    // What went wrong is said in words rather than swallowed: a fault here used to come back
    // as an empty answer, which reads to a host exactly like a sentence with nothing in it.
    // The class and what it was in the middle of are in the sentence, because "it faulted" on
    // its own sends the next person to a debugger.
    try {
        samples = _synthesise(ids, speaker, said);
    } catch (const std::exception &thrown) {
        samples = PackedFloat32Array();
        said = _faulted(ids.size(), ncnn_report::describe(thrown));
    } catch (...) {
        samples = PackedFloat32Array();
        said = _faulted(ids.size(), ncnn_report::describe_unknown());
    }
    total_ms = now_ms() - started;
    return samples;
}

// The worker's whole life: run, then hand the samples to the main thread. Emitting from here
// instead would put a signal on a thread the engine's listeners are not written for.
void NcnnTTS::work(PackedInt32Array ids, int speaker, int64_t at) {
    // Fenced again out here, around everything the worker does and not only the model: an
    // exception that escapes a thread body is std::terminate, and the process then ends with
    // no line anywhere. Whatever happens, the delivery is still posted, so a host waiting on
    // a signal is answered rather than left waiting for ever.
    try {
        pending_samples = run(ids, speaker, pending_problem);
    } catch (const std::exception &thrown) {
        pending_samples = PackedFloat32Array();
        pending_problem = _faulted(ids.size(), ncnn_report::describe(thrown));
    } catch (...) {
        pending_samples = PackedFloat32Array();
        pending_problem = _faulted(ids.size(), ncnn_report::describe_unknown());
    }
    callable_mp(this, &NcnnTTS::deliver).call_deferred(at);
}


// One sentence about a fault, with everything somebody would otherwise have to ask for: which
// class, what it was in the middle of, how long the sentence was, and what was thrown.
String NcnnTTS::_faulted(int symbols, const String &thrown) const {
    return String("Govorilka: {0} faulted while {1} ({2} symbols): {3}").format(Array::make(
            get_class(), ncnn_report::last_note(), symbols, thrown));
}

// The delivery, on the main thread. The flag is given back either way -- a sentence that was
// cut off has to free the graphs too, or the next one would find them taken for ever -- and
// only a turn still current says anything.
void NcnnTTS::deliver(int64_t at) {
    const bool current = at == epoch.load();
    if (owed.load() && owed_at.load() == at) {
        owed.store(false);
        owed_at.store(-1);
        busy.store(false);
    }
    if (!current) {
        return;
    }
    problem = pending_problem;
    if (pending_samples.is_empty()) {
        emit_signal("failed", pending_problem);
        return;
    }
    emit_signal("synthesised", pending_samples, sample_rate());
}

void NcnnTTS::join_worker() {
    if (worker.joinable()) {
        worker.join();
    }
}

void NcnnTTS::doing(const String &what) const {
    ncnn_report::note(get_class() + " " + what);
}


ncnn::Mat NcnnTTS::owned(const float *source, int w, int h) {
    return ncnn_util::owned(source, w, h);
}

ncnn::Mat NcnnTTS::owned_index(int value) {
    return ncnn_util::owned_index(value);
}

ncnn::Mat NcnnTTS::owned_indices(const int *source, int count) {
    return ncnn_util::owned_indices(source, count);
}

String NcnnTTS::pick(const PackedStringArray &files, const String &mark, const String &suffix) {
    return ncnn_util::pick(files, mark, suffix);
}

double NcnnTTS::now_ms() {
    return ncnn_util::now_ms();
}

void NcnnTTS::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load", "model_dir", "num_threads"), &NcnnTTS::load);
    ClassDB::bind_method(D_METHOD("synthesise", "ids", "speaker"), &NcnnTTS::synthesise);
    ClassDB::bind_method(D_METHOD("synthesise_async", "ids", "speaker"),
            &NcnnTTS::synthesise_async);
    ClassDB::bind_method(D_METHOD("cancel"), &NcnnTTS::cancel);
    ClassDB::bind_method(D_METHOD("is_busy"), &NcnnTTS::is_busy);
    ClassDB::bind_method(D_METHOD("is_loaded"), &NcnnTTS::is_loaded);
    ClassDB::bind_method(D_METHOD("unload"), &NcnnTTS::unload);
    ClassDB::bind_method(D_METHOD("last_problem"), &NcnnTTS::last_problem);
    ClassDB::bind_method(D_METHOD("set_seed", "value"), &NcnnTTS::set_seed);
    ClassDB::bind_method(D_METHOD("get_seed"), &NcnnTTS::get_seed);
    ClassDB::bind_method(D_METHOD("last_timings"), &NcnnTTS::last_timings);
    ClassDB::bind_method(D_METHOD("describe_family"), &NcnnTTS::describe_family);
    ClassDB::bind_method(D_METHOD("sample_rate"), &NcnnTTS::sample_rate);
    ClassDB::bind_method(D_METHOD("speaker_count"), &NcnnTTS::speaker_count);

    ADD_PROPERTY(PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");

    ADD_SIGNAL(MethodInfo("synthesised",
            PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "samples"),
            PropertyInfo(Variant::INT, "rate")));
    ADD_SIGNAL(MethodInfo("failed", PropertyInfo(Variant::STRING, "message")));
}
