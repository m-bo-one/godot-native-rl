#include "gigaam_asr.h"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <cmath>
#include <cstring>
#include <thread>

using namespace godot;

namespace {

// What a clip is made of: frames of a 320-sample window every 160 samples, not centred, so
// a clip of N samples is (N - 320) / 160 + 1 frames and needs no padding at all. A centred
// front end here would be every frame half a window late.
constexpr int MEL_BANDS = 64;
constexpr int FFT_SIZE = 320;
constexpr int HOP = 160;
constexpr int SPECTRUM_BINS = FFT_SIZE / 2 + 1;
constexpr float LOG_FLOOR = 1.0e-9f;
constexpr float LOG_CEIL = 1.0e9f;

// The rotary tables the graph takes as its second and third input: the base is the
// encoder's own configuration and the width is one head. Another base is the same words at
// the wrong positions, which reads as the short ones missing.
constexpr float ROPE_BASE = 5000.0f;
constexpr int ROPE_DIM = 48;

// The longest clip the encoder is handed. Its attention is quadratic in the frames, and past
// this the package that trained it segments the audio itself; a held key never gets near it.
constexpr int SAMPLE_RATE = 16000;
constexpr int LONGEST_SECONDS = 25;

// The two lines of the token file that name something other than the letter they stand
// for, so an editor cannot strip the one and the other is found by name rather than by
// position.
const char *SPACE_TOKEN = "<space>";
const char *BLANK_TOKEN = "<blk>";

// How many frames a load probes the graph with, to read the width of its head off the graph
// rather than trust the token file to match it.
constexpr int PROBE_FRAMES = 8;

// How many positions the encoder answers for a number of frames: two convolutions of
// kernel five, stride two, padding two, one after the other.
int subsampled(int frames) {
    for (int i = 0; i < 2; i++) {
        frames = (frames + 2 * 2 - 5) / 2 + 1;
    }
    return frames;
}

// The cos and sin tables for `positions` positions as the encoder builds them: the twenty-
// four frequencies once and then again, so the rotate-half layout pairs each with itself.
// Single precision throughout, which is what the checkpoint's own buffer was computed in.
void rope_tables(int positions, ncnn::Mat &cosines, ncnn::Mat &sines) {
    cosines.create(ROPE_DIM, positions, 1);
    sines.create(ROPE_DIM, positions, 1);
    if (cosines.empty() || sines.empty()) {
        return;
    }
    float inv_freq[ROPE_DIM / 2];
    for (int i = 0; i < ROPE_DIM / 2; i++) {
        inv_freq[i] = 1.0f / powf(ROPE_BASE, (float)(2 * i) / (float)ROPE_DIM);
    }
    float *cos_out = (float *)cosines.data;
    float *sin_out = (float *)sines.data;
    for (int p = 0; p < positions; p++) {
        for (int i = 0; i < ROPE_DIM / 2; i++) {
            const float angle = (float)p * inv_freq[i];
            const float c = cosf(angle);
            const float s = sinf(angle);
            cos_out[p * ROPE_DIM + i] = c;
            cos_out[p * ROPE_DIM + i + ROPE_DIM / 2] = c;
            sin_out[p * ROPE_DIM + i] = s;
            sin_out[p * ROPE_DIM + i + ROPE_DIM / 2] = s;
        }
    }
}

} // namespace

// The graph is this class's, so it is given back here and not in the base's destructor,
// which runs after this one and would call into a table that is already gone.
GigaAMASR::~GigaAMASR() {
    unload();
}

// One graph, one table file and one token file out of the folder, each found by the
// fragment naming its part. Refused as a whole when any is missing or does not fit the
// others: the seam in front of this names the file, and a graph decoding through a table
// of the wrong width would answer letters from the wrong rows.
bool GigaAMASR::_load_graphs(const String &folder, const String &language, int num_threads) {
    (void)language;
    Ref<DirAccess> dir = DirAccess::open(folder);
    if (dir.is_null()) {
        return false;
    }
    const PackedStringArray files = dir->get_files();
    const String param_name = pick(files, "ctc", ".ncnn.param");
    const String tables_name = pick(files, "frontend", ".bin");
    const String tokens_name = pick(files, "tokens", ".txt");
    if (param_name.is_empty() || tables_name.is_empty() || tokens_name.is_empty()) {
        return false;
    }
    const String bin_name = param_name.trim_suffix(".param") + ".bin";
    if (!graph.load(folder.path_join(param_name), folder.path_join(bin_name), num_threads)) {
        return false;
    }

    // The window and then the filterbank, one band's filter after another, raw floats.
    const PackedByteArray tables = FileAccess::get_file_as_bytes(folder.path_join(tables_name));
    const int wanted = (FFT_SIZE + MEL_BANDS * SPECTRUM_BINS) * (int)sizeof(float);
    if (tables.size() != wanted) {
        ERR_PRINT(vformat("GigaAM export refused: the front-end table file is %d bytes where "
                          "a 320-point window and 64 filters of 161 bins are %d.",
                tables.size(), wanted));
        return false;
    }
    window.resize((size_t)FFT_SIZE);
    memcpy(window.data(), tables.ptr(), (size_t)FFT_SIZE * sizeof(float));
    mel_filters.resize((size_t)MEL_BANDS * SPECTRUM_BINS);
    memcpy(mel_filters.data(), tables.ptr() + (size_t)FFT_SIZE * sizeof(float),
            (size_t)MEL_BANDS * SPECTRUM_BINS * sizeof(float));
    fft.prepare(FFT_SIZE);

    // One class per line, in the order the head numbers them. The blank is found by its
    // name wherever it sits; a file without one has nothing to collapse against.
    const PackedByteArray raw = FileAccess::get_file_as_bytes(folder.path_join(tokens_name));
    if (raw.is_empty()) {
        return false;
    }
    tokens.clear();
    blank = -1;
    const char *text = (const char *)raw.ptr();
    int start = 0;
    for (int i = 0; i <= raw.size(); i++) {
        if (i != raw.size() && text[i] != '\n') {
            continue;
        }
        if (i == raw.size() && i == start) {
            break;
        }
        int length = i - start;
        if (length > 0 && text[start + length - 1] == '\r') {
            length--;
        }
        std::string token(text + start, (size_t)length);
        if (token == SPACE_TOKEN) {
            token = " ";
        } else if (token == BLANK_TOKEN) {
            blank = (int)tokens.size();
        }
        tokens.push_back(token);
        start = i + 1;
    }

    const int width = output_width();
    if (blank < 0 || width != (int)tokens.size()) {
        ERR_PRINT(vformat("GigaAM export refused: %d token lines with the blank at %d, against "
                          "a head %d wide. The token file and the graph are halves of two "
                          "exports.",
                (int)tokens.size(), blank, width));
        return false;
    }
    return true;
}

String GigaAMASR::describe_family() const {
    return "GigaAM CTC (ncnn)";
}

// How wide the head answers, found by running the graph once over a few frames of nothing.
// Read off the graph rather than assumed, so a token file of another export is refused at
// the load instead of decoding through the wrong rows.
int GigaAMASR::output_width() {
    ncnn::Mat mel;
    mel.create(PROBE_FRAMES, MEL_BANDS);
    if (mel.empty()) {
        return -1;
    }
    mel.fill(0.0f);
    ncnn::Mat logits;
    if (!run_graph(mel, logits)) {
        return -1;
    }
    return logits.w;
}

// The one pass: the log-mel and the two rotary tables for its length in, the log-probabilities
// out as one row per position. The tables are built per clip because their height is the
// clip's, which is what lets one graph take any length at all.
bool GigaAMASR::run_graph(const ncnn::Mat &mel, ncnn::Mat &logits) {
    ncnn::Mat cosines;
    ncnn::Mat sines;
    rope_tables(subsampled(mel.w), cosines, sines);
    if (cosines.empty() || sines.empty()) {
        return false;
    }
    ncnn::Extractor ex = graph.net.create_extractor();
    ex.input("in0", mel);
    ex.input("in1", cosines);
    ex.input("in2", sines);
    return ex.extract("out0", logits) == 0 && !logits.empty();
}

void GigaAMASR::_unload_graphs() {
    graph.clear();
    window.clear();
    mel_filters.clear();
    tokens.clear();
    blank = -1;
}

void GigaAMASR::_report_timings(Dictionary &out) const {
    out["mel_ms"] = mel_ms;
    out["encoder_ms"] = encoder_ms;
    out["decode_ms"] = decode_ms;
    out["frames"] = frame_count;
    out["tokens"] = token_count;
}

// The model's own log-mel, on the tables the checkpoint carries: the windowed frame through
// the 320-point FFT, the power spectrum, the 64 filters, and the natural log over a clamp.
// No normalisation of any kind follows, which is the one way this front end is simpler than
// the other family's. The frames split across the threads the graph was loaded with.
void GigaAMASR::log_mel(const float *audio, int frames, ncnn::Mat &mel) const {
    const int workers = threads > 1 ? threads : 1;
    auto band = [&](int from, int to) {
        std::vector<float> re((size_t)FFT_SIZE);
        std::vector<float> im((size_t)FFT_SIZE);
        std::vector<float> work_re((size_t)FFT_SIZE);
        std::vector<float> work_im((size_t)FFT_SIZE);
        std::vector<float> power((size_t)SPECTRUM_BINS);
        for (int frame = from; frame < to; frame++) {
            const float *source = audio + (size_t)frame * HOP;
            for (int i = 0; i < FFT_SIZE; i++) {
                re[(size_t)i] = source[i] * window[(size_t)i];
            }
            fft.run(re.data(), im.data(), work_re.data(), work_im.data());
            for (int k = 0; k < SPECTRUM_BINS; k++) {
                power[(size_t)k] = re[(size_t)k] * re[(size_t)k] + im[(size_t)k] * im[(size_t)k];
            }
            for (int b = 0; b < MEL_BANDS; b++) {
                const float *filter = mel_filters.data() + (size_t)b * SPECTRUM_BINS;
                float sum = 0.0f;
                for (int k = 0; k < SPECTRUM_BINS; k++) {
                    sum += filter[k] * power[(size_t)k];
                }
                const float clamped = sum < LOG_FLOOR ? LOG_FLOOR : (sum > LOG_CEIL ? LOG_CEIL : sum);
                mel.row(b)[frame] = logf(clamped);
            }
        }
    };

    if (workers == 1 || frames < workers) {
        band(0, frames);
        return;
    }
    std::vector<std::thread> pool;
    const int each = (frames + workers - 1) / workers;
    for (int i = 1; i < workers; i++) {
        const int from = i * each;
        const int to = (i + 1) * each < frames ? (i + 1) * each : frames;
        if (from < to) {
            pool.emplace_back(band, from, to);
        }
    }
    band(0, each);
    for (std::thread &one : pool) {
        one.join();
    }
}

// One clip through the graph. The samples arrive at sixteen kilohertz and are framed as they
// are, cut at the longest clip the encoder takes; the letters are the argmax of every
// position with repeats collapsed and the blank dropped, and the run is read as UTF-8 once.
String GigaAMASR::_decode(const std::vector<float> &samples) {
    const double started = now_ms();
    frame_count = 0;
    token_count = 0;

    const size_t longest = (size_t)LONGEST_SECONDS * SAMPLE_RATE;
    const int count = (int)(samples.size() < longest ? samples.size() : longest);
    if (count < FFT_SIZE) {
        return String();
    }
    const int frames = (count - FFT_SIZE) / HOP + 1;
    ncnn::Mat mel;
    mel.create(frames, MEL_BANDS);
    if (mel.empty()) {
        return String();
    }
    log_mel(samples.data(), frames, mel);
    const double after_mel = now_ms();

    ncnn::Mat logits;
    if (!run_graph(mel, logits) || logits.w != (int)tokens.size()) {
        return String();
    }
    const double after_encoder = now_ms();

    // Greedy CTC: the likeliest class per position, a run of one class written once, and
    // the blank written never -- it is what separates two of the same letter in a row.
    std::string bytes;
    int previous = -1;
    for (int position = 0; position < logits.h; position++) {
        const float *scores = logits.row(position);
        int best = 0;
        float top = scores[0];
        for (int i = 1; i < logits.w; i++) {
            if (scores[i] > top) {
                top = scores[i];
                best = i;
            }
        }
        if (best != blank && best != previous) {
            bytes += tokens[(size_t)best];
            token_count++;
        }
        previous = best;
    }
    const double after_decode = now_ms();

    mel_ms = after_mel - started;
    encoder_ms = after_encoder - after_mel;
    decode_ms = after_decode - after_encoder;
    frame_count = logits.h;
    // Named as UTF-8 rather than handed to the plain constructor, which reads a byte as a
    // Latin-1 letter and turns every Cyrillic one into the two that spell it.
    return String::utf8(bytes.c_str(), (int64_t)bytes.length()).strip_edges();
}

void GigaAMASR::_bind_methods() {
}
