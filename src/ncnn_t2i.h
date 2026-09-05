#ifndef NCNN_T2I_H
#define NCNN_T2I_H

#include "ncnn_graph.h"

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/reg_ex.hpp>
#include <godot_cpp/templates/hash_map.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace godot {

// A generator whose numbers come back the same from the same seed, so a picture made in the
// game and the same picture made in the exporter's gate are comparable pixel for pixel.
// splitmix64 for the bits, Box-Muller for the shape: a latent is drawn from a standard normal,
// and feeding it anything else is a different picture rather than a noisier one.
//
// A seed of zero is drawn from the clock and a process-wide counter together, which is what a
// game wants: two characters asking for the same scene inside one tick should not get the same
// frame, and the clock on its own cannot tell them apart.
class T2INoise {
    uint64_t state;
    bool has_spare = false;
    float spare = 0.0f;

public:
    explicit T2INoise(uint64_t seed);

    float uniform();
    float normal();
};

// What every text-to-image family on ncnn shares, with the network itself left to a subclass:
// the one worker thread, the flag that says the graphs are taken, the delivery on the main
// thread, the CLIP tokenizer, the seeded latent, the two schedulers and the latent-to-pixels
// step. A family fills in only how its folder is read and how one denoise call reaches its
// graphs.
//
// The graphs are used from one thread at a time. generate() runs on the caller's thread,
// generate_async() hands the same work to one worker and delivers on the main thread; each
// refuses while the other holds them, or the extractors run re-entrant.
//
// The graphs are freed under a lock every run holds: unload(), load() and the destructor wait
// for a run already going on any thread, the way the worker is joined, so a call from a second
// thread meets a finished run rather than graphs freed under one.
class NcnnT2I : public RefCounted {
    GDCLASS(NcnnT2I, RefCounted)

public:
    // Which arithmetic turns one network call into the next latent. Named in the folder's
    // manifest rather than guessed from the graphs, because the same UNet trains under either.
    enum Scheduler {
        SCHEDULER_EULER,
        SCHEDULER_LCM,
    };

private:
    // The one worker, the flag that says the graphs are taken, and what an answer will carry.
    // The flag is raised in one atomic step and lowered by whichever path owns it -- the
    // blocking run as it returns, the worker's delivery on the main thread.
    std::thread worker;
    std::atomic<bool> busy{false};
    Ref<Image> pending_picture;
    String pending_problem;

    // Whether the worker owns the flag, and which turn's worker owns it. A delivery lowers it
    // whether or not its turn still counts -- a cancelled picture has to free the graphs too --
    // but only when the flag is the one it raised: a turn started over the top of it owns its
    // own, and an old delivery lowering that would let two runs into the graphs at once.
    std::atomic<bool> owed{false};
    std::atomic<int64_t> owed_at{-1};

    // The answer the worker left behind, and whether anybody has handed it out yet. A caller
    // that draws no frames drains it itself through deliver_pending(); the deferred call that
    // follows finds it already taken and says nothing twice.
    std::atomic<bool> answer_ready{false};
    std::atomic<bool> answer_taken{true};
    std::atomic<int64_t> worker_at{-1};

    // Held by every run and by whatever frees or replaces the graphs. The busy flag says the
    // graphs are taken; this is what makes unload() wait for the thread that took them.
    std::mutex graphs_lock;

    // Which turn the worker was started for. cancel(), unload() and load() move it on, so an
    // answer delivered after any of them is dropped rather than shown in place of the next one.
    std::atomic<int64_t> epoch{0};

    // Whether the turn in flight has been thrown away. It is what lets the next picture start
    // at once: its answer is already dropped, so joining it costs only what is left of it.
    std::atomic<bool> dropped{false};

    // Written under the lock and read on a caller's thread ahead of it, so it is atomic; a run
    // reads it again once it holds the lock, which is the read that decides.
    std::atomic<bool> loaded{false};

    // Whether a run has finished since the last load. The timings are empty until one has, so
    // a caller reading them cannot mistake the numbers of a model that has drawn nothing.
    bool ran = false;
    double load_ms = 0.0;
    double total_ms = 0.0;
    double tokenize_ms = 0.0;
    double text_ms = 0.0;
    double denoise_ms = 0.0;
    double decode_ms = 0.0;
    int last_width = 0;
    int last_height = 0;

    // What the last run refused to do, kept for a caller that took the blocking road and has
    // no signal to read it off.
    String problem;

    // The CLIP byte-level BPE, read out of the folder rather than compiled in: the vocabulary
    // by name, the merge ranks by the pair they join, and the map from a byte to the letter
    // the vocabulary spells it with. The splitter is the tokeniser's own word pattern.
    HashMap<String, int> vocab;
    HashMap<String, int> ranks;
    String byte_letter[256];
    Ref<RegEx> splitter;
    Ref<RegEx> spaces;
    int bos_id = -1;
    int eos_id = -1;

protected:
    static void _bind_methods();

    // One step of the schedule, filled from the manifest the exporter wrote out of the
    // checkpoint's own scheduler. Both roads use the timestep and the input scale; the rest is
    // one road's or the other's, and the unused half is left at zero.
    struct Step {
        float timestep = 0.0f;
        float input_scale = 1.0f;
        float sigma = 0.0f;
        float sigma_next = 0.0f;
        float alpha_prod = 1.0f;
        float alpha_prod_next = 1.0f;
        float c_skip = 0.0f;
        float c_out = 1.0f;
    };

    // How many threads the graphs were loaded with. Read by a subclass that reads a graph
    // again for another picture size, so the second read runs on the same number.
    int threads = 1;

    // What the manifest said, which is the whole of what the base needs to know about a
    // family: how a latent is shaped, how many steps it takes and how they are spaced, how
    // wide the token window is, and which sizes the folder actually carries graphs for.
    String family;
    Scheduler scheduler = SCHEDULER_EULER;
    float guidance = 1.0f;
    int latent_channels = 4;
    int latent_downscale = 8;
    float init_noise_sigma = 1.0f;
    int context_length = 77;
    bool text_encoder_fp32 = true;
    std::vector<Step> schedule;
    std::vector<std::pair<int, int>> sizes;

    // The three halves a family supplies. Each answers false with a sentence in `problem` --
    // naming the file or the size it could not find, because "the model did not load" sends
    // the next person to a folder listing. All run with the busy flag held and the lock taken.
    virtual bool _load_graphs(const String &folder, const Dictionary &manifest, int num_threads,
            String &problem) = 0;
    virtual void _unload_graphs() = 0;

    // The token window to the sequence the network is conditioned on.
    virtual bool _encode_text(const int *ids, int count, ncnn::Mat &hidden, String &problem) = 0;

    // Whatever a family has to do once before a picture of this size, rather than once per
    // step: reading another structure over the weights it holds is the case there is. Kept out
    // of the denoise call so what a step costs is what the network costs.
    virtual bool _prepare_size(int width, int height, String &problem) { return true; }

    // One call of the network: the scaled latent and the conditioning at this timestep, to
    // whatever the network predicts. A family whose timestep is baked into its graph ignores
    // the argument, and the size chooses which structure the call runs through.
    virtual bool _denoise(const ncnn::Mat &latent, const ncnn::Mat &hidden, float timestep,
            int width, int height, ncnn::Mat &out, String &problem) = 0;

    // The latent to three channels of [-1, 1] pixels at the asked-for size.
    virtual bool _decode_latent(const ncnn::Mat &latent, int width, int height, ncnn::Mat &rgb,
            String &problem) = 0;

    // What the last run cost inside the family, added to the numbers kept here. Read after the
    // flag is lowered; a read during a run sees the previous picture's numbers.
    virtual void _report_timings(Dictionary &out) const {}

    // What this is in the middle of, in a few words, set by the family as it moves from one
    // graph to the next. It is what a fault says it was doing, and what the handler installed
    // at initialisation names when the process is already dying.
    void doing(const String &what) const;

    static ncnn::Mat owned(const float *source, int w, int h);
    static ncnn::Mat owned_indices(const int *source, int count);
    static String pick(const PackedStringArray &files, const String &mark, const String &suffix);
    static double now_ms();

public:
    NcnnT2I() = default;
    ~NcnnT2I();

    // The folder's manifest, tokeniser and schedule are read here; the graphs are the family's.
    // A thread count of zero or less takes half the cores the library counts, because a
    // language model is usually running on the other half.
    bool load(const String &model_dir, int num_threads);

    // One prompt to one picture, on the calling thread. A null reference is a picture that was
    // refused, and last_problem() says why. A seed of zero is drawn from the clock.
    Ref<Image> generate(const String &prompt, int64_t seed, int width, int height);

    // The same on a worker, answered by the picture_ready or failed signal on the main thread.
    // False is a turn that never started: the graphs are held by a picture nobody cancelled,
    // or the thread could not be made.
    bool generate_async(const String &prompt, int64_t seed, int width, int height);

    // Throws away the answer to the picture in flight without waiting for it. Nothing is
    // emitted for that turn, and the next generate_async() joins what is left of it rather
    // than refusing.
    void cancel();

    // Hands out an answer the worker has already finished, for a caller that draws no frames:
    // a deferred call arrives on a frame, and a test runner and a headless tool draw none.
    // True when this call was the one that handed the picture out.
    bool deliver_pending();

    // The same for a caller that wants to wait: drains until the answer is out or the time is
    // up. Zero waits for as long as it takes.
    bool wait_for_picture(int timeout_ms);

    bool is_busy() const;
    bool is_loaded() const;
    void unload();

    String last_problem() const;

    // The prompt as the token window the text encoder takes, for a caller checking the
    // tokeniser against the one the export was traced with. Empty before a model is loaded.
    PackedInt32Array tokenize(const String &prompt) const;

    // The draws a latent is filled from, in the order they are consumed: channel by channel and
    // row by row within a channel. Standard normal and unscaled -- what a latent is worth is the
    // manifest's business. Bound so the generator can be held to a seed with no model loaded.
    PackedFloat32Array draw_noise(int64_t seed, int count) const;

    // One step of a schedule applied by hand, which is the arithmetic the loop runs. The step is
    // a manifest row; it may carry its own `scheduler`, so both roads are provable against a
    // reference without a folder of graphs behind them.
    PackedFloat32Array advance(const PackedFloat32Array &latent,
            const PackedFloat32Array &predicted, const Dictionary &step, bool last,
            int64_t seed) const;

    // Three channels of [-1, 1] pixels, flattened channel-first, as the picture the decoder's
    // output becomes. The run uses this same conversion, so a test of it is a test of the run.
    static Ref<Image> to_image(const PackedFloat32Array &pixels, int width, int height);

    // What the last picture cost, per graph. The numbers belong to the picture that was just
    // delivered: read after picture_ready, or after the blocking call returns.
    Dictionary last_timings() const;

    // The sizes the loaded folder carries graphs for, as [Vector2i]. A size not in the list is
    // refused with a sentence naming the ones that are.
    Array offered_sizes() const;

    // One phrase naming the family of model this draws with, for a menu or a log line.
    virtual String describe() const;

private:
    String _faulted(const String &thrown) const;
    bool _read_manifest(const String &folder, Dictionary &manifest, String &problem);
    bool _read_tokenizer(const String &folder, String &problem);
    void _clear_tokenizer();
    bool _offers(int width, int height) const;
    String _size_refused(int width, int height) const;
    void _bpe(const String &token, std::vector<String> &pieces) const;
    static Step _step_of(const Dictionary &row);
    static void _apply_step(float *sample, const float *predicted, int count, Scheduler road,
            const Step &step, bool last, T2INoise &noise);

    Ref<Image> make(const String &prompt, uint64_t seed, int width, int height, String &problem);
    Ref<Image> run(const String &prompt, uint64_t seed, int width, int height, String &said);
    void work(String prompt, uint64_t seed, int width, int height, int64_t at);
    void deliver(int64_t at);
    void join_worker();
};

} // namespace godot

VARIANT_ENUM_CAST(NcnnT2I::Scheduler);

#endif // NCNN_T2I_H
