#ifndef NCNN_TTS_H
#define NCNN_TTS_H

#include "ncnn_graph.h"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>

namespace godot {

// A generator whose numbers come back the same from the same seed, so a sentence made in the
// game and the same sentence made in the exporter's gate are comparable sample for sample.
// splitmix64 for the bits, Box-Muller for the shape: VITS is trained on standard normal noise,
// and feeding it anything else -- a uniform draw, say -- is a different voice, not a noisier one.
//
// Seeded from the clock when the seed is zero, which is what a game wants: two characters
// saying the same line should not say it identically.
class TtsNoise {
    uint64_t state;
    bool has_spare = false;
    float spare = 0.0f;

public:
    explicit TtsNoise(uint64_t seed);

    float uniform();
    float normal();
};

// What every speech synthesiser on ncnn shares, with the model itself left to a subclass: the
// one worker thread, the flag that says the graphs are taken, and the delivery on the main
// thread. A family of graphs fills in how its folder is read and how a run of symbol ids
// becomes samples, and nothing else.
//
// The graphs are used from one thread at a time. synthesise() runs on the caller's thread,
// synthesise_async() hands the same work to one worker and delivers on the main thread; each
// refuses while the other holds them, or the extractors run re-entrant.
//
// The graphs are freed under a lock every run holds: unload(), load() and the destructor wait
// for a run already going on any thread, the way the worker is joined, so a call from a second
// thread meets a finished run rather than graphs freed under one.
class NcnnTTS : public RefCounted {
    GDCLASS(NcnnTTS, RefCounted)

    // The one worker, the flag that says the graphs are taken, and what an answer will carry.
    // The flag is raised in one atomic step and lowered by whichever path owns it -- the
    // blocking run as it returns, the worker's delivery on the main thread.
    std::thread worker;
    std::atomic<bool> busy{false};
    PackedFloat32Array pending_samples;
    String pending_problem;

    // Whether the worker owns the flag, and which turn's worker owns it. A delivery lowers it
    // whether or not its turn still counts -- a cancelled sentence has to free the graphs too
    // -- but only when the flag is the one it raised: a turn started over the top of it owns
    // its own, and an old delivery lowering that would let two runs into the graphs at once.
    std::atomic<bool> owed{false};
    std::atomic<int64_t> owed_at{-1};

    // Held by every run and by whatever frees or replaces the graphs. The busy flag says the
    // graphs are taken; this is what makes unload() wait for the thread that took them.
    std::mutex graphs_lock;

    // Which turn the worker was started for. cancel(), unload() and load() move it on, so an
    // answer delivered after any of them is dropped rather than spoken over the next line.
    std::atomic<int64_t> epoch{0};

    // Whether the turn in flight has been cancelled. It is what lets the next sentence start
    // at once: its answer is already thrown away, so joining it costs only what is left of it.
    std::atomic<bool> dropped{false};

    // Written under the lock and read on a caller's thread ahead of it, so it is atomic; a run
    // reads it again once it holds the lock, which is the read that decides.
    std::atomic<bool> loaded{false};
    double load_ms = 0.0;
    double total_ms = 0.0;

    // What the last run refused to do, kept for a caller that took the blocking road and has
    // no signal to read it off.
    String problem;

protected:
    static void _bind_methods();

    // How many threads the graphs were loaded with. Read by a subclass that splits work of
    // its own -- a length regulator, a front end computed by hand -- across the same number.
    int threads = 1;

    // Where the noise comes from. Zero is a different voice on every call, which is what a
    // game wants; anything else is the same sentence twice, which is what a gate needs.
    uint64_t seed = 0;

    // The three halves a family supplies. _load_graphs reads its own files out of the folder
    // and refuses with false; _synthesise takes the symbol ids and answers samples in [-1, 1],
    // saying why in `problem` when it answers none. Both run with the busy flag already held.
    virtual bool _load_graphs(const String &folder, int num_threads) = 0;
    virtual PackedFloat32Array _synthesise(const PackedInt32Array &ids, int speaker,
            String &problem) = 0;
    virtual void _unload_graphs() = 0;

    // What the last run cost inside the family, added to the two numbers kept here. Read after
    // the flag is lowered; a read during a run sees the previous sentence's numbers.
    virtual void _report_timings(Dictionary &out) const {}

    static ncnn::Mat owned(const float *source, int w, int h);
    static ncnn::Mat owned_index(int value);
    static ncnn::Mat owned_indices(const int *source, int count);
    static String pick(const PackedStringArray &files, const String &mark, const String &suffix);
    static double now_ms();

public:
    NcnnTTS() = default;
    ~NcnnTTS();

    bool load(const String &model_dir, int num_threads);

    // One sentence of symbol ids to samples, on the calling thread. Empty is a sentence that
    // was refused, and last_problem() says why.
    PackedFloat32Array synthesise(const PackedInt32Array &ids, int speaker);

    // The same on a worker, answered by the synthesised or failed signal on the main thread.
    // False is a turn that never started: the graphs are held by a sentence nobody cancelled,
    // or the thread could not be made.
    bool synthesise_async(const PackedInt32Array &ids, int speaker);

    // Throws away the answer to the sentence in flight without waiting for it. Nothing is
    // emitted for that turn, and the next synthesise_async() joins what is left of it rather
    // than refusing -- which is what makes cutting a character off and starting again work.
    void cancel();

    bool is_busy() const;
    bool is_loaded() const;
    void unload();

    String last_problem() const;

    void set_seed(int64_t value);
    int64_t get_seed() const;

    // What the last sentence cost, per graph. The numbers belong to the sentence whose samples
    // were just delivered: read after synthesised, or after the blocking call returns.
    Dictionary last_timings() const;

    // One phrase naming the family of model this speaks with, for a menu or a log line.
    virtual String describe_family() const;

    // The rate the samples are made at, which is the model's and not the mixer's: whoever
    // plays them resamples. Zero before a model is loaded.
    virtual int sample_rate() const;

    // How many voices the loaded model holds. One means a model with no speaker to choose,
    // and a speaker index is then ignored rather than refused.
    virtual int speaker_count() const;

private:
    PackedFloat32Array run(const PackedInt32Array &ids, int speaker, String &said);
    void work(PackedInt32Array ids, int speaker, int64_t at);
    void deliver(int64_t at);
    void join_worker();
};

} // namespace godot

#endif // NCNN_TTS_H
