#ifndef WHISPER_ASR_H
#define WHISPER_ASR_H

#include "ncnn_asr.h"

#include <godot_cpp/variant/array.hpp>

#include <atomic>
#include <string>
#include <vector>

namespace godot {

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
// at a time. The threading, the resampling and the delivery are the base's; what is here is
// everything a Whisper graph needs that another family would not -- the mel front end, the
// encoder pass, the cached decoder loop, the byte-level vocabulary and the language table.
class WhisperASR : public NcnnASR {
    GDCLASS(WhisperASR, NcnnASR)

    // The graphs a decode actually runs. The export's sixth, the fbank graph, is not among
    // them: its ncnn Spectrogram layer costs more than the encoder, so only its weights are
    // read -- they are the mel filterbank and nothing else.
    NcnnGraph encoder;
    NcnnGraph decoder;
    NcnnGraph embed_token;
    NcnnGraph embed_position;
    NcnnGraph proj_out;

    // The mel filterbank out of the fbank graph's weights: 80 rows of 201, the only thing
    // that file holds. Reading it rather than recomputing it keeps this front end and the
    // exported graph agreeing on every coefficient.
    std::vector<float> mel_filters;
    SmallFft fft;

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

    // The token the model writes where it heard nobody, and how likely it found it at the
    // first step of the last clip. The number is the model's own opinion of the silence; a
    // seam reads it beside the text and drops what a graph narrated over a quiet room.
    int no_speech = 0;
    double no_speech_prob = 0.0;

    // What the model took the language to be at that same step, as the three likeliest of the
    // ninety-nine with their shares, most likely first. Information beside the text and never
    // the policy: the clip is transcribed in the language the host named. Off, the step is
    // still projected for the no-speech share and only the language part of it is skipped.
    std::atomic<bool> detecting_language{true};
    int language_index[3] = {-1, -1, -1};
    double language_share[3] = {0.0, 0.0, 0.0};

    // What the last decode cost, in milliseconds, and how many tokens it wrote. Written by
    // whichever thread decoded and read after it has finished, which is what the busy flag
    // orders; a read while a decode runs sees the previous clip's numbers rather than a tear.
    double mel_ms = 0.0;
    double encoder_ms = 0.0;
    double decoder_ms = 0.0;
    int token_count = 0;

protected:
    static void _bind_methods();

    bool _load_graphs(const String &folder, const String &language, int num_threads) override;
    String _decode(const std::vector<float> &samples) override;
    void _unload_graphs() override;
    void _report_timings(Dictionary &out) const override;

public:
    WhisperASR() = default;
    ~WhisperASR();

    String describe_family() const override;
    double last_no_speech_prob() const override;
    String last_detected_language() const override;
    double last_language_prob() const override;
    Array last_language_candidates() const override;

    void set_detect_language(bool enabled);
    bool is_detecting_language() const;

private:
    void log_mel(const std::vector<float> &audio, ncnn::Mat &mel) const;
    std::vector<int> run_decoder(const ncnn::Mat &states);
    void read_first_step(const ncnn::Mat &hidden);
};

} // namespace godot

#endif // WHISPER_ASR_H
