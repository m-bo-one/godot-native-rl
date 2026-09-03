#ifndef WHISPER_RECOGNIZER_H
#define WHISPER_RECOGNIZER_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <mat.h>
#include <net.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace godot {

// One graph of a Whisper export together with the two buffers it was read out of. ncnn
// aliases the weight bytes rather than copying them, so dropping the .bin buffer would
// leave every layer of the net pointing into freed memory on the next extract.
struct WhisperGraph {
    ncnn::Net net;
    PackedByteArray param;
    PackedByteArray weights;

    bool load(const String &param_path, const String &bin_path, int num_threads);
    void clear();
};

// A complex FFT for a length whose factors are small. It splits on the smallest factor and
// recurses, which for Whisper's 400-point window is a few thousand operations per frame
// against a hundred and sixty thousand for a direct sum.
struct SmallFft {
    int size = 0;
    std::vector<float> cosines;
    std::vector<float> sines;

    void prepare(int n);
    // Transforms one real frame in place into `re`/`im`, using `work` as scratch. All three
    // arrays are `size` long; `work` may be shared between calls on the same thread.
    void run(float *re, float *im, float *work_re, float *work_im) const;
};

// A Whisper export as ncnn runs it: five graphs and a vocabulary, decoded greedily one token
// at a time. Everything is read as bytes rather than opened by a path, so a model inside an
// exported pack works exactly as a folder beside the game does.
//
// The graphs are used from one thread at a time. transcribe() decodes on the caller's thread,
// transcribe_async() hands the same work to one worker and delivers the text on the main
// thread; each refuses while the other holds them, or the extractors run re-entrant.
class WhisperRecognizer : public RefCounted {
    GDCLASS(WhisperRecognizer, RefCounted)

    // The graphs a decode actually runs. The export's sixth, the fbank graph, is not among
    // them: its ncnn Spectrogram layer costs more than the encoder, so only its weights are
    // read -- they are the mel filterbank and nothing else.
    WhisperGraph encoder;
    WhisperGraph decoder;
    WhisperGraph embed_token;
    WhisperGraph embed_position;
    WhisperGraph proj_out;
    bool loaded = false;

    // The mel filterbank out of the fbank graph's weights: 80 rows of 201, the only thing
    // that file holds. Reading it rather than recomputing it keeps this front end and the
    // exported graph agreeing on every coefficient.
    std::vector<float> mel_filters;
    SmallFft fft;
    int threads = 1;

    // How many attention caches travel between decode steps, read off the decoder's own
    // structure. It is a property of the size that was exported -- tiny has eight pairs and
    // base twelve -- so a number fixed here would decode one size and nothing else.
    int cache_pairs = 0;

    // Every token as the bytes it stands for, the byte-level alphabet already undone. Decoded
    // once at load: doing it per token would run the same mapping over the same lines all day.
    std::vector<std::string> vocab;

    // What the decoder is told before it says anything: start, the language, transcribe, and
    // no timestamps. A wrong language token here is a model that translates rather than hears.
    std::vector<int> prompt;
    int end_of_text = 0;

    // The one worker, the flag that says the graphs are taken, and the text an answer will
    // carry. The flag is raised in one atomic step and lowered by whichever path owns it --
    // the blocking decode as it returns, the worker's delivery on the main thread.
    std::thread worker;
    std::atomic<bool> busy{false};
    String pending_text;

    // Which model the worker was started against. unload() and load() move it on, so a
    // result delivered after either is dropped rather than reported for a model that went.
    std::atomic<int64_t> epoch{0};

    // What the last decode cost, in milliseconds, and how many tokens it wrote. Written by
    // whichever thread decoded and read after it has finished, which is what the busy flag
    // orders; a read while a decode runs sees the previous clip's numbers rather than a tear.
    double load_ms = 0.0;
    double mel_ms = 0.0;
    double encoder_ms = 0.0;
    double decoder_ms = 0.0;
    double total_ms = 0.0;
    int token_count = 0;

protected:
    static void _bind_methods();

public:
    WhisperRecognizer() = default;
    ~WhisperRecognizer();

    bool load(const String &model_dir, const String &language, int num_threads);
    String transcribe(const PackedFloat32Array &samples, int sample_rate);
    bool transcribe_async(const PackedFloat32Array &samples, int sample_rate);
    bool is_busy() const;
    bool is_loaded() const;
    void unload();
    Dictionary last_timings() const;

private:
    String decode(const PackedFloat32Array &samples, int sample_rate);
    void log_mel(const std::vector<float> &audio, ncnn::Mat &mel) const;
    std::vector<int> run_decoder(const ncnn::Mat &states);
    void work(PackedFloat32Array samples, int sample_rate, int64_t at);
    void deliver(int64_t at);
    void join_worker();
};

} // namespace godot

#endif // WHISPER_RECOGNIZER_H
