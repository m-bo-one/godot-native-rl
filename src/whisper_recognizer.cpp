#include "whisper_recognizer.h"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/callable_method_pointer.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>

#if NCNN_VULKAN
#include <gpu.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstring>

using namespace godot;

namespace {

// What a Whisper clip is: thirty seconds at sixteen kilohertz, and the mel frames made of
// them. The window is 400 samples every 160, centred, which is 3001 frames -- the encoder
// wants one fewer, and feeding it 3001 is a silent shape error rather than a refusal.
constexpr int SAMPLE_RATE = 16000;
constexpr int CLIP_SAMPLES = 480000;
constexpr int MEL_BANDS = 80;
constexpr int MEL_FRAMES = 3000;
constexpr int FFT_SIZE = 400;
constexpr int HOP = 160;
constexpr int SPECTRUM_BINS = FFT_SIZE / 2 + 1;

// What the caches are called in every export, and the most a graph may declare. Neither the
// state width nor the number of caches is written down here: they are properties of the size
// that was exported -- tiny is 384 wide with 8 caches, base 512 with 12 -- and a constant for
// either decodes one size and answers nothing at all for every other.
const char *CACHE_MARK = "out_cache_k";
constexpr int MOST_CACHE_PAIRS = 64;

// The longest a turn may run. Whisper's own window is 448 positions and half of it is the
// prompt it never gets here; past this a greedy loop is repeating itself rather than hearing.
constexpr int MOST_TOKENS = 224;

// The languages a multilingual Whisper knows, in the order its vocabulary numbers them. The
// position in this list is the whole of the language token: one out of place transcribes a
// Russian clip as though it were Polish, which reads as words nobody said.
const char *LANGUAGES[] = {
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
    "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
    "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
};
constexpr int LANGUAGE_COUNT = 99;

// Where the special tokens sit above the vocabulary, counted from its own size rather than
// written down: end of text is the first one after it, and the four the prompt needs follow.
constexpr int TRANSCRIBE_AFTER_TRANSLATE = 1;
constexpr int NOTIMESTAMPS_AFTER_TRANSCRIBE = 4;

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

// A Mat ncnn owns, filled from somebody else's memory. Every input has to go through this:
// a Mat wrapping a foreign pointer carries a null refcount, and the first in-place layer to
// consume it dereferences that null and takes the process down.
ncnn::Mat owned(const float *source, int w, int h) {
    ncnn::Mat mat;
    mat.create(w, h);
    if (!mat.empty() && source != nullptr) {
        memcpy(mat.data, source, (size_t)w * (size_t)h * sizeof(float));
    }
    return mat;
}

// The same for a single whole number, which is what the two embedding graphs index with.
// The lookup reads the blob's four bytes as an int rather than converting them, so a float
// index is a bit pattern past the last row -- and out of range is clamped, never reported.
ncnn::Mat owned_index(int value) {
    ncnn::Mat mat;
    mat.create(1);
    if (!mat.empty()) {
        ((int *)mat.data)[0] = value;
    }
    return mat;
}

// The GPT-2 byte-level alphabet inverted: every letter a vocabulary line may carry back to
// the byte it stands for. Without it a token is the remapped spelling rather than the word,
// and every multi-byte letter comes out as two Latin ones.
void build_byte_decoder(short *table) {
    for (int i = 0; i < 512; i++) {
        table[i] = -1;
    }
    bool direct[256] = {};
    for (int b = '!'; b <= '~'; b++) direct[b] = true;
    for (int b = 0xA1; b <= 0xAC; b++) direct[b] = true;
    for (int b = 0xAE; b <= 0xFF; b++) direct[b] = true;
    for (int b = 0; b < 256; b++) {
        if (direct[b]) {
            table[b] = (short)b;
        }
    }
    int spare = 0;
    for (int b = 0; b < 256; b++) {
        if (!direct[b]) {
            table[256 + spare] = (short)b;
            spare++;
        }
    }
}

// One vocabulary line as the bytes it stands for. The lines are UTF-8 and every codepoint in
// them is below 0x800, so two forms cover the file; anything else is a line from another
// vocabulary and is dropped rather than turned into a byte that was never meant.
std::string line_to_bytes(const char *line, int length, const short *table) {
    std::string out;
    for (int i = 0; i < length;) {
        unsigned char first = (unsigned char)line[i];
        unsigned int point = first;
        if (first < 0x80) {
            i += 1;
        } else if ((first & 0xE0) == 0xC0 && i + 1 < length) {
            point = ((first & 0x1F) << 6) | ((unsigned char)line[i + 1] & 0x3F);
            i += 2;
        } else {
            i += 1;
            continue;
        }
        short byte = point < 512 ? table[point] : -1;
        if (byte >= 0) {
            out.push_back((char)(unsigned char)byte);
        }
    }
    return out;
}

// The one file of a folder carrying a fragment and a suffix. Fragments rather than whole
// names: an export ships them under whatever prefix it was written with, and a folder taken
// from anywhere has to work without somebody renaming six files first.
String pick(const PackedStringArray &files, const String &mark, const String &suffix) {
    for (int i = 0; i < files.size(); i++) {
        const String lower = files[i].to_lower();
        if (lower.contains(mark) && lower.ends_with(suffix)) {
            return files[i];
        }
    }
    return String();
}

double now_ms() {
    return (double)Time::get_singleton()->get_ticks_usec() / 1000.0;
}

// How many attention caches the decoder declares, counted off its own structure: one pair per
// MultiHeadAttention, and a decoder layer has two of those. Reading it rather than assuming it
// is what lets one loop run every size of an export instead of the one it was written against.
int count_cache_pairs(const PackedByteArray &param) {
    const char *text = (const char *)param.ptr();
    const int length = param.size();
    const int mark = (int)strlen(CACHE_MARK);
    int found = 0;
    for (int i = 0; i + mark < length; i++) {
        if (memcmp(text + i, CACHE_MARK, (size_t)mark) == 0) {
            found++;
        }
    }
    return found;
}

int smallest_factor(int n) {
    for (int p = 2; p * p <= n; p++) {
        if (n % p == 0) {
            return p;
        }
    }
    return n;
}

// One level of a Cooley-Tukey split: p sub-transforms of length n/p, then one combine over
// the master twiddle table. `work` is shared with the recursion below because a child is
// finished with it before its parent's combine begins.
void fft_step(const float *in_re, const float *in_im, int stride,
        float *out_re, float *out_im, int n, int full,
        const float *cosines, const float *sines, float *work_re, float *work_im) {
    if (n == 1) {
        out_re[0] = in_re[0];
        out_im[0] = in_im[0];
        return;
    }
    const int p = smallest_factor(n);
    const int m = n / p;
    for (int j = 0; j < p; j++) {
        fft_step(in_re + (size_t)j * stride, in_im + (size_t)j * stride, stride * p,
                out_re + j * m, out_im + j * m, m, full, cosines, sines, work_re, work_im);
    }
    const int span = full / n;
    for (int k = 0; k < n; k++) {
        float sum_re = 0.0f;
        float sum_im = 0.0f;
        const int r = k % m;
        for (int j = 0; j < p; j++) {
            const int at = j * m + r;
            const int t = (int)(((long long)j * k * span) % full);
            const float c = cosines[t];
            const float s = sines[t];
            sum_re += out_re[at] * c - out_im[at] * s;
            sum_im += out_re[at] * s + out_im[at] * c;
        }
        work_re[k] = sum_re;
        work_im[k] = sum_im;
    }
    memcpy(out_re, work_re, (size_t)n * sizeof(float));
    memcpy(out_im, work_im, (size_t)n * sizeof(float));
}

} // namespace

void SmallFft::prepare(int n) {
    size = n;
    cosines.resize((size_t)n);
    sines.resize((size_t)n);
    for (int t = 0; t < n; t++) {
        const double angle = -2.0 * 3.14159265358979323846 * (double)t / (double)n;
        cosines[(size_t)t] = (float)cos(angle);
        sines[(size_t)t] = (float)sin(angle);
    }
}

// `re` arrives holding the windowed frame and leaves holding the transform. The input is
// real, so `im` is zeroed here rather than asked for: halving the work by packing the frame
// into a half-length complex transform would buy a few milliseconds for a page of algebra.
void SmallFft::run(float *re, float *im, float *work_re, float *work_im) const {
    std::vector<float> in_re((size_t)size);
    memcpy(in_re.data(), re, (size_t)size * sizeof(float));
    std::vector<float> in_im((size_t)size, 0.0f);
    fft_step(in_re.data(), in_im.data(), 1, re, im, size, size,
            cosines.data(), sines.data(), work_re, work_im);
}

bool WhisperGraph::load(const String &param_path, const String &bin_path, int num_threads,
        bool use_gpu, bool gpu_fp16) {
    clear();
    net.opt.num_threads = num_threads;
    // Set before the structure is read, not after: ncnn decides at load time which pipeline
    // each layer gets, so flipping this afterwards leaves the graph on the side it was built for.
    net.opt.use_vulkan_compute = use_gpu;
    if (use_gpu && !gpu_fp16) {
        // Half precision is what a device is quickest at and it is not free here: measured on
        // this encoder it changes the words on a Russian clip at tiny, and at base it overflows
        // outright -- the graph then writes the same token until the loop gives up. Off, the
        // answer matches the processor's exactly and costs nothing measurable.
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
    }

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

void WhisperGraph::clear() {
    net.clear();
    param = PackedByteArray();
    weights = PackedByteArray();
}

// The devices ncnn can see, and zero on a library built without Vulkan at all. Creating the
// instance is what fills the count in, and it is never destroyed: another net in this same
// library may still hold devices of its own, and tearing the instance down under one crashes.
int WhisperRecognizer::gpu_count() {
#if NCNN_VULKAN
    static const int found = []() {
        if (ncnn::create_gpu_instance() != 0) {
            return 0;
        }
        return ncnn::get_gpu_count();
    }();
    return found;
#else
    return 0;
#endif
}

WhisperRecognizer::~WhisperRecognizer() {
    unload();
}

// The folder is read through the engine rather than opened by a library, so a model inside an
// exported pack loads exactly as a folder beside the game does. A load while a turn is in
// flight would swap the graphs under the worker, so the worker is joined first.
bool WhisperRecognizer::load(const String &model_dir, const String &language, int num_threads,
        bool use_gpu, bool gpu_fp16) {
    unload();
    const double started = now_ms();
    threads = num_threads > 0 ? num_threads : 1;
    // A device is asked for, not assumed: a library built without Vulkan or a machine with no
    // driver answers zero here, and the graphs are then built for the processor instead.
    on_gpu = use_gpu && gpu_count() > 0;

    Ref<DirAccess> dir = DirAccess::open(model_dir);
    if (dir.is_null()) {
        return false;
    }
    const PackedStringArray files = dir->get_files();

    const String vocab_name = pick(files, "vocab", ".txt");
    const String filters_name = pick(files, "fbank", ".ncnn.bin");
    if (vocab_name.is_empty() || filters_name.is_empty()) {
        return false;
    }

    // Only the encoder goes to the device. It is one pass over the whole clip and the kind of
    // work a card is built for; the decode loop is a hundred tiny steps that would each have to
    // send the attention caches across and read them back, which costs more than it saves.
    struct Part {
        WhisperGraph *graph;
        const char *mark;
        bool device;
    };
    const Part parts[] = {
        {&encoder, "encoder", true},
        {&decoder, "decoder", false},
        {&embed_token, "embed_token", false},
        {&embed_position, "embed_position", false},
        {&proj_out, "proj_out", false},
    };
    for (const Part &part : parts) {
        const String param_name = pick(files, part.mark, ".ncnn.param");
        if (param_name.is_empty()) {
            unload();
            return false;
        }
        const String bin_name = param_name.trim_suffix(".param") + ".bin";
        if (!part.graph->load(model_dir.path_join(param_name),
                    model_dir.path_join(bin_name), threads,
                    on_gpu && part.device, gpu_fp16)) {
            unload();
            return false;
        }
    }

    cache_pairs = count_cache_pairs(decoder.param);
    if (cache_pairs <= 0 || cache_pairs > MOST_CACHE_PAIRS) {
        unload();
        return false;
    }

    // The fbank graph's weights are its mel filterbank and nothing else -- one MemoryData
    // layer of 201 by 80 floats, stored raw. Anything else in that file is another export.
    const PackedByteArray filters = FileAccess::get_file_as_bytes(
            model_dir.path_join(filters_name));
    if (filters.size() != MEL_BANDS * SPECTRUM_BINS * (int)sizeof(float)) {
        unload();
        return false;
    }
    mel_filters.resize((size_t)MEL_BANDS * SPECTRUM_BINS);
    memcpy(mel_filters.data(), filters.ptr(), (size_t)filters.size());
    fft.prepare(FFT_SIZE);

    short table[512];
    build_byte_decoder(table);
    const PackedByteArray raw = FileAccess::get_file_as_bytes(model_dir.path_join(vocab_name));
    if (raw.is_empty()) {
        unload();
        return false;
    }
    vocab.clear();
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
        vocab.push_back(line_to_bytes(text + start, length, table));
        start = i + 1;
    }

    int language_index = -1;
    const String wanted = language.strip_edges().to_lower();
    for (int i = 0; i < LANGUAGE_COUNT; i++) {
        if (wanted == LANGUAGES[i]) {
            language_index = i;
            break;
        }
    }
    if (language_index < 0 || vocab.empty()) {
        unload();
        return false;
    }

    // The special tokens sit directly above the vocabulary, in the order Whisper numbers them.
    const int size = (int)vocab.size();
    end_of_text = size;
    const int start_of_transcript = size + 1;
    const int translate = start_of_transcript + 1 + LANGUAGE_COUNT;
    const int transcribe = translate + TRANSCRIBE_AFTER_TRANSLATE;
    prompt.clear();
    prompt.push_back(start_of_transcript);
    prompt.push_back(start_of_transcript + 1 + language_index);
    prompt.push_back(transcribe);
    prompt.push_back(transcribe + NOTIMESTAMPS_AFTER_TRANSCRIBE);

    loaded = true;
    load_ms = now_ms() - started;
    return true;
}

String WhisperRecognizer::transcribe(const PackedFloat32Array &samples, int sample_rate) {
    BusyGuard guard(busy);
    if (!guard.taken() || !loaded || samples.is_empty()) {
        return String();
    }
    return decode(samples, sample_rate);
}

bool WhisperRecognizer::transcribe_async(const PackedFloat32Array &samples, int sample_rate) {
    BusyGuard guard(busy);
    if (!guard.taken() || !loaded || samples.is_empty()) {
        return false;
    }
    join_worker();
    // A thread that could not be started answers false with the flag given back, rather than
    // letting the system error unwind into the engine, which has no handler for one.
    try {
        worker = std::thread(&WhisperRecognizer::work, this, samples, sample_rate, epoch.load());
    } catch (...) {
        return false;
    }
    guard.hand_on();
    return true;
}

bool WhisperRecognizer::is_busy() const {
    return busy.load();
}

bool WhisperRecognizer::is_loaded() const {
    return loaded;
}

void WhisperRecognizer::unload() {
    epoch.fetch_add(1);
    join_worker();
    busy.store(false);
    loaded = false;
    on_gpu = false;
    encoder.clear();
    decoder.clear();
    embed_token.clear();
    embed_position.clear();
    proj_out.clear();
    mel_filters.clear();
    vocab.clear();
    prompt.clear();
}

// What the last clip cost, so a project can measure this road rather than trust a number
// written down somewhere. The numbers are of one decode: reading them while another runs
// answers the one before it.
Dictionary WhisperRecognizer::last_timings() const {
    Dictionary out;
    out["load_ms"] = load_ms;
    out["mel_ms"] = mel_ms;
    out["encoder_ms"] = encoder_ms;
    out["decoder_ms"] = decoder_ms;
    out["total_ms"] = total_ms;
    out["tokens"] = token_count;
    out["on_gpu"] = on_gpu;
    return out;
}

// Whisper's own log-mel, computed here rather than by the export's fbank graph: ncnn's
// Spectrogram layer costs more for one clip than the encoder does. The frames are split
// across the threads the model was loaded with, which is the whole of the parallelism --
// the library itself is built with OpenMP off so that nothing has to travel beside it.
void WhisperRecognizer::log_mel(const std::vector<float> &audio, ncnn::Mat &mel) const {
    // Reflect padding by half a window, so frame t is centred on sample t * 160 exactly as
    // torch's centred stft has it. Without it every frame is half a window early.
    const int pad = FFT_SIZE / 2;
    std::vector<float> padded((size_t)CLIP_SAMPLES + 2 * pad);
    for (int i = 0; i < pad; i++) {
        padded[(size_t)i] = audio[(size_t)(pad - i)];
        padded[(size_t)CLIP_SAMPLES + pad + i] = audio[(size_t)CLIP_SAMPLES - 2 - i];
    }
    memcpy(padded.data() + pad, audio.data(), (size_t)CLIP_SAMPLES * sizeof(float));

    std::vector<float> window((size_t)FFT_SIZE);
    for (int i = 0; i < FFT_SIZE; i++) {
        window[(size_t)i] = 0.5f - 0.5f * (float)cos(
                2.0 * 3.14159265358979323846 * (double)i / (double)FFT_SIZE);
    }

    const int workers = threads > 1 ? threads : 1;
    auto band = [&](int from, int to) {
        std::vector<float> re((size_t)FFT_SIZE);
        std::vector<float> im((size_t)FFT_SIZE);
        std::vector<float> work_re((size_t)FFT_SIZE);
        std::vector<float> work_im((size_t)FFT_SIZE);
        std::vector<float> power((size_t)SPECTRUM_BINS);
        for (int frame = from; frame < to; frame++) {
            const float *source = padded.data() + (size_t)frame * HOP;
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
                mel.row(b)[frame] = sum;
            }
        }
    };

    if (workers == 1) {
        band(0, MEL_FRAMES);
    } else {
        std::vector<std::thread> pool;
        const int each = (MEL_FRAMES + workers - 1) / workers;
        for (int i = 1; i < workers; i++) {
            const int from = i * each;
            const int to = (i + 1) * each < MEL_FRAMES ? (i + 1) * each : MEL_FRAMES;
            if (from < to) {
                pool.emplace_back(band, from, to);
            }
        }
        band(0, each < MEL_FRAMES ? each : MEL_FRAMES);
        for (std::thread &one : pool) {
            one.join();
        }
    }

    // Whisper's normalisation, and the reason the maximum is taken over the whole matrix
    // rather than per frame: a per-frame floor would lift the silence between words.
    float top = -1.0e30f;
    for (int b = 0; b < MEL_BANDS; b++) {
        float *row = mel.row(b);
        for (int t = 0; t < MEL_FRAMES; t++) {
            const float clamped = row[t] < 1.0e-10f ? 1.0e-10f : row[t];
            row[t] = log10f(clamped);
            if (row[t] > top) {
                top = row[t];
            }
        }
    }
    const float floor = top - 8.0f;
    for (int b = 0; b < MEL_BANDS; b++) {
        float *row = mel.row(b);
        for (int t = 0; t < MEL_FRAMES; t++) {
            row[t] = ((row[t] > floor ? row[t] : floor) + 4.0f) / 4.0f;
        }
    }
}

// One clip through the graphs. The samples are padded or trimmed to Whisper's own thirty
// seconds, which is the only length the encoder has a shape for; a rate that is not sixteen
// kilohertz is stretched to it, because a clip read at the wrong rate is words at a wrong pitch.
String WhisperRecognizer::decode(const PackedFloat32Array &samples, int sample_rate) {
    const double started = now_ms();

    std::vector<float> audio((size_t)CLIP_SAMPLES, 0.0f);
    const float *source = samples.ptr();
    const int count = samples.size();
    if (sample_rate == SAMPLE_RATE || sample_rate <= 0) {
        const int taken = count < CLIP_SAMPLES ? count : CLIP_SAMPLES;
        memcpy(audio.data(), source, (size_t)taken * sizeof(float));
    } else {
        const double step = (double)sample_rate / (double)SAMPLE_RATE;
        const int taken = (int)((double)count / step);
        const int limit = taken < CLIP_SAMPLES ? taken : CLIP_SAMPLES;
        for (int i = 0; i < limit; i++) {
            const double at = (double)i * step;
            const int left = (int)at;
            const int right = left + 1 < count ? left + 1 : left;
            const float fraction = (float)(at - (double)left);
            audio[(size_t)i] = source[left] * (1.0f - fraction) + source[right] * fraction;
        }
    }

    ncnn::Mat mel;
    mel.create(MEL_FRAMES, MEL_BANDS);
    if (mel.empty()) {
        return String();
    }
    log_mel(audio, mel);
    const double after_mel = now_ms();

    ncnn::Mat states;
    {
        ncnn::Extractor ex = encoder.net.create_extractor();
        ex.input("in0", mel);
        // Whatever width this size was exported at. Measuring it against a number written
        // here is what made every size but one answer with no words at all.
        if (ex.extract("out0", states) != 0 || states.empty()) {
            return String();
        }
    }
    const double after_encoder = now_ms();

    const std::vector<int> written = run_decoder(states);
    const double after_decoder = now_ms();

    std::string bytes;
    for (int token : written) {
        if (token >= 0 && token < (int)vocab.size()) {
            bytes += vocab[(size_t)token];
        }
    }

    mel_ms = after_mel - started;
    encoder_ms = after_encoder - after_mel;
    decoder_ms = after_decoder - after_encoder;
    total_ms = after_decoder - started;
    token_count = (int)written.size();
    // The bytes are UTF-8 only once the whole run is concatenated: a Cyrillic letter routinely
    // straddles two tokens, so decoding token by token would produce mojibake. Named as UTF-8
    // rather than handed to the plain constructor, which reads a byte as a Latin-1 letter and
    // turns every Cyrillic one into the two that spell it.
    return String::utf8(bytes.c_str(), (int64_t)bytes.length()).strip_edges();
}

// The greedy loop. One token per step: the prompt is walked through first so the caches hold
// it, then whatever the graph writes is fed back until it says it is finished. The sixteen
// caches stay ncnn's own objects and are handed straight back, and the mask grows one column
// per step -- a mask of the wrong width has the attention reading past the end of its row.
std::vector<int> WhisperRecognizer::run_decoder(const ncnn::Mat &states) {
    std::vector<int> written;
    // Left unset on the first step. An Input layer is not re-run to produce a blob, and the
    // attention reads an empty cache as no history, which is exactly what step one has.
    std::vector<ncnn::Mat> cache((size_t)cache_pairs * 2);
    std::vector<float> mask_row((size_t)MOST_TOKENS + 1, 0.0f);

    int token = prompt[0];
    for (int position = 0; position < MOST_TOKENS; position++) {
        ncnn::Mat embedding;
        {
            ncnn::Mat of_token;
            ncnn::Mat of_position;
            ncnn::Mat token_in = owned_index(token);
            ncnn::Extractor ex = embed_token.net.create_extractor();
            ex.input("in0", token_in);
            if (ex.extract("out0", of_token) != 0) {
                break;
            }
            ncnn::Mat position_in = owned_index(position);
            ncnn::Extractor ex2 = embed_position.net.create_extractor();
            ex2.input("in0", position_in);
            if (ex2.extract("out0", of_position) != 0 || of_position.w != of_token.w) {
                break;
            }
            // The width the export was written at, taken from the lookup that just answered.
            embedding.create(of_token.w, 1);
            if (embedding.empty()) {
                break;
            }
            // The exporter took the position add out of the decoder graph, so the two
            // embeddings are summed here; without it every token is read at position zero.
            const float *a = of_token.row(0);
            const float *b = of_position.row(0);
            float *out = embedding.row(0);
            for (int i = 0; i < of_token.w; i++) {
                out[i] = a[i] + b[i];
            }
        }

        // One column per token already attended to, plus this one. Zeros: a greedy loop only
        // ever looks backwards, so there is nothing ahead of it that would have to be hidden.
        ncnn::Mat mask = owned(mask_row.data(), position + 1, 1);
        if (mask.empty()) {
            break;
        }

        ncnn::Mat hidden;
        std::vector<ncnn::Mat> next((size_t)cache_pairs * 2);
        {
            ncnn::Extractor ex = decoder.net.create_extractor();
            ex.input("in0", embedding);
            ex.input("in1", states);
            ex.input("in2", mask);
            char name[32];
            for (int i = 0; i < cache_pairs; i++) {
                if (cache[(size_t)i * 2].empty()) {
                    continue;
                }
                snprintf(name, sizeof(name), "cache_k%d", i);
                ex.input(name, cache[(size_t)i * 2]);
                snprintf(name, sizeof(name), "cache_v%d", i);
                ex.input(name, cache[(size_t)i * 2 + 1]);
            }
            if (ex.extract("out0", hidden) != 0) {
                break;
            }
            for (int i = 0; i < cache_pairs; i++) {
                snprintf(name, sizeof(name), "out_cache_k%d", i);
                ex.extract(name, next[(size_t)i * 2]);
                snprintf(name, sizeof(name), "out_cache_v%d", i);
                ex.extract(name, next[(size_t)i * 2 + 1]);
            }
        }
        cache.swap(next);

        // The prompt is fed for its caches and never asked what comes next: the first thing
        // worth projecting is the step after the last of it.
        if (position + 1 < (int)prompt.size()) {
            token = prompt[(size_t)position + 1];
            continue;
        }

        ncnn::Mat logits;
        {
            ncnn::Extractor ex = proj_out.net.create_extractor();
            ex.input("in0", hidden);
            if (ex.extract("out0", logits) != 0) {
                break;
            }
        }
        const float *scores = logits.row(0);
        int best = 0;
        float top = scores[0];
        for (int i = 1; i < logits.w; i++) {
            if (scores[i] > top) {
                top = scores[i];
                best = i;
            }
        }
        if (best == end_of_text) {
            break;
        }
        written.push_back(best);
        token = best;
    }
    return written;
}

// The worker's whole life: decode, then hand the text to the main thread. Emitting from
// here instead would put a signal on a thread the engine's listeners are not written for.
void WhisperRecognizer::work(PackedFloat32Array samples, int sample_rate, int64_t at) {
    pending_text = decode(samples, sample_rate);
    callable_mp(this, &WhisperRecognizer::deliver).call_deferred(at);
}

// The delivery, on the main thread. A turn whose model was unloaded or replaced while it
// ran is dropped: the flag it would clear belongs to whatever was started after it.
void WhisperRecognizer::deliver(int64_t at) {
    if (at != epoch.load() || !busy.load()) {
        return;
    }
    busy.store(false);
    emit_signal("transcribed", pending_text);
}

void WhisperRecognizer::join_worker() {
    if (worker.joinable()) {
        worker.join();
    }
}

void WhisperRecognizer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load", "model_dir", "language", "num_threads", "use_gpu",
                                 "gpu_fp16"),
            &WhisperRecognizer::load, DEFVAL(false), DEFVAL(false));
    ClassDB::bind_static_method("WhisperRecognizer", D_METHOD("gpu_count"),
            &WhisperRecognizer::gpu_count);
    ClassDB::bind_method(D_METHOD("transcribe", "samples", "sample_rate"),
            &WhisperRecognizer::transcribe);
    ClassDB::bind_method(D_METHOD("transcribe_async", "samples", "sample_rate"),
            &WhisperRecognizer::transcribe_async);
    ClassDB::bind_method(D_METHOD("is_busy"), &WhisperRecognizer::is_busy);
    ClassDB::bind_method(D_METHOD("is_loaded"), &WhisperRecognizer::is_loaded);
    ClassDB::bind_method(D_METHOD("unload"), &WhisperRecognizer::unload);
    ClassDB::bind_method(D_METHOD("last_timings"), &WhisperRecognizer::last_timings);

    ADD_SIGNAL(MethodInfo("transcribed", PropertyInfo(Variant::STRING, "text")));
}
