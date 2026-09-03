#ifndef NCNN_ASR_H
#define NCNN_ASR_H

#include "ncnn_graph.h"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <mat.h>
#include <net.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace godot {

// What every speech recogniser on ncnn shares, with the model itself left to a subclass: the
// one worker thread, the flag that says the graphs are taken, the delivery on the main thread
// and the resampling to sixteen kilohertz. A family of graphs -- Whisper, or another -- fills
// in how its folder is read and how a clip becomes text, and nothing else.
//
// The graphs are used from one thread at a time. transcribe() decodes on the caller's thread,
// transcribe_async() hands the same work to one worker and delivers the text on the main
// thread; each refuses while the other holds them, or the extractors run re-entrant.
//
// The graphs are freed under a lock every decode holds: unload(), load() and the destructor
// wait for a decode already running on any thread, the way the worker is joined, so a call
// from a second thread meets a finished decode rather than graphs freed under one.
class NcnnASR : public RefCounted {
    GDCLASS(NcnnASR, RefCounted)

    // The one worker, the flag that says the graphs are taken, and the text an answer will
    // carry. The flag is raised in one atomic step and lowered by whichever path owns it --
    // the blocking decode as it returns, the worker's delivery on the main thread.
    std::thread worker;
    std::atomic<bool> busy{false};
    String pending_text;

    // Whether the worker owns the flag. unload() lowers it only then: a delivery whose epoch
    // has gone never will, while a blocking caller's flag is its own to lower and is left alone.
    std::atomic<bool> owed{false};

    // Held by every decode and by whatever frees or replaces the graphs. The busy flag says
    // the graphs are taken; this is what makes unload() wait for the thread that took them.
    std::mutex graphs_lock;

    // Which model the worker was started against. unload() and load() move it on, so a
    // result delivered after either is dropped rather than reported for a model that went.
    std::atomic<int64_t> epoch{0};

    // Written under the lock and read on a caller's thread ahead of it, so it is atomic; a
    // decode reads it again once it holds the lock, which is the read that decides.
    std::atomic<bool> loaded{false};
    double load_ms = 0.0;
    double total_ms = 0.0;

    // What the last decode refused to do, kept for a caller that has only an empty answer.
    String problem;

protected:
    static void _bind_methods();

    // How many threads the graphs were loaded with. Read by a subclass that splits its own
    // work -- a front end computed by hand -- across the same number the graphs run on.
    int threads = 1;

    // The two halves a family supplies. _load_graphs reads its own files out of the folder
    // and refuses with false; _decode takes sixteen-kilohertz mono in [-1, 1] and answers the
    // text, empty for a clip it could not read. Both run with the busy flag already held.
    virtual bool _load_graphs(const String &folder, const String &language, int num_threads) = 0;
    virtual String _decode(const std::vector<float> &samples) = 0;
    virtual void _unload_graphs() = 0;

    // What the last decode cost inside the family, added to the two numbers kept here. Read
    // after the flag is lowered; a read during a decode sees the previous clip's numbers.
    virtual void _report_timings(Dictionary &out) const {}

    // A Mat ncnn owns, filled from somebody else's memory. Every input has to go through
    // this: a Mat wrapping a foreign pointer carries a null refcount, and the first in-place
    // layer to consume it dereferences that null and takes the process down.
    // What this is in the middle of, in a few words, set by the family as it moves through
    // its graphs. It is what a fault says it was doing, and what the handler installed at
    // initialisation names when the process is already dying.
    void doing(const String &what) const;

    static ncnn::Mat owned(const float *source, int w, int h);

    // The same for a single whole number, which is what an embedding graph indexes with. The
    // lookup reads the blob's four bytes as an int rather than converting them, so a float
    // index is a bit pattern past the last row -- and out of range is clamped, never reported.
    static ncnn::Mat owned_index(int value);

    // The one file of a folder carrying a fragment and a suffix. Fragments rather than whole
    // names: an export ships them under whatever prefix it was written with, and a folder
    // taken from anywhere has to work without somebody renaming its files first.
    static String pick(const PackedStringArray &files, const String &mark, const String &suffix);

    static double now_ms();

public:
    NcnnASR() = default;
    ~NcnnASR();

    bool load(const String &model_dir, const String &language, int num_threads);
    String transcribe(const PackedFloat32Array &samples, int sample_rate);
    bool transcribe_async(const PackedFloat32Array &samples, int sample_rate);
    bool is_busy() const;
    bool is_loaded() const;
    void unload();

    // What the last clip cost, per graph. The numbers belong to the clip whose text was just
    // delivered: read after transcribed, or after the blocking call returns. Read during a
    // decode they are the clip before it, never a tear of the one running.
    // Why the last clip came back with no words, where the reason was a fault rather than a
    // quiet room. Empty for an ordinary decode. This family reports no failure by signal --
    // its answer to a clip it could not read is no words -- so this is where a fault is kept.
    String last_problem() const;

    Dictionary last_timings() const;

    // One phrase naming the family of model this reads, for a menu or a log line. A host
    // that lists recognisers shows this rather than the class name.
    virtual String describe_family() const;

    // How sure the model was that the last clip held no speech at all, from 0 to 1, or -1
    // from a family that has no such signal. Read after the answer, like the timings: the
    // number belongs to the clip whose text was just delivered, and during a decode it is the
    // previous clip's.
    virtual double last_no_speech_prob() const;

    // What the model took the last clip's language to be, as a code, with how sure it was and
    // the likeliest few as [{code, prob}]. Information and never the policy: the clip was
    // transcribed in the language the host named. A family without the signal answers "",
    // zero and an empty list.
    virtual String last_detected_language() const;
    virtual double last_language_prob() const;
    virtual Array last_language_candidates() const;

private:
    String _faulted(const String &thrown) const;
    String run(const PackedFloat32Array &samples, int sample_rate);
    void work(PackedFloat32Array samples, int sample_rate, int64_t at);
    void deliver(int64_t at);
    void join_worker();
};

} // namespace godot

#endif // NCNN_ASR_H
