#include "whisper_asr.h"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>

using namespace godot;

namespace {

// What a Whisper clip is made of: mel frames of a 400-sample window every 160 samples,
// centred, which over thirty seconds is 3001 frames -- the encoder wants one fewer, and
// feeding it 3001 is a silent shape error rather than a refusal.
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

// The vocabulary that table is laid out for: the multilingual one of tiny, base, small and
// medium, 50 257 text tokens with 1 608 specials over them. A large-v3 export carries one
// language more and every token after it moves, so it is refused rather than read as words
// in the language next door; an English-only export has a shorter table and is refused too.
constexpr int MULTILINGUAL_VOCAB = 50257;
constexpr int SPECIAL_TOKENS = 1608;

// Where the special tokens sit above the vocabulary, counted from its own size rather than
// written down: end of text is the first one after it, then start, the languages, translate,
// transcribe, and after transcribe the language-model start, the previous-text start, the
// no-speech token and the no-timestamps token, in that order.
constexpr int TRANSCRIBE_AFTER_TRANSLATE = 1;
constexpr int NOSPEECH_AFTER_TRANSCRIBE = 3;
constexpr int NOTIMESTAMPS_AFTER_TRANSCRIBE = 4;

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

// How many mel frames the encoder's Input line declares, or Whisper's own 3000 where it says
// nothing: `Input in0 0 1 in0 0=1000 1=80` is an export for a ten-second window. Read off the
// graph rather than assumed, so a window chosen at export time needs no number written here.
int declared_frames(const PackedByteArray &param) {
    const char *text = (const char *)param.ptr();
    const int length = param.size();
    int at = 0;
    while (at < length) {
        int end = at;
        while (end < length && text[end] != '\n') {
            end++;
        }
        if (end - at > 5 && memcmp(text + at, "Input", 5) == 0) {
            for (int i = at; i + 2 < end; i++) {
                if (text[i] == ' ' && text[i + 1] == '0' && text[i + 2] == '=') {
                    const int frames = atoi(text + i + 3);
                    if (frames > 0 && frames % 2 == 0 && frames <= MEL_FRAMES) {
                        return frames;
                    }
                }
            }
            return MEL_FRAMES;
        }
        at = end + 1;
    }
    return MEL_FRAMES;
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

} // namespace

// The graphs are this class's, so they are given back here and not in the base's destructor,
// which runs after this one and would call into a table that is already gone.
WhisperASR::~WhisperASR() {
    unload();
}

// Six files and a vocabulary out of one folder, each found by the fragment naming its part.
// Refused as a whole when any is missing: the seam in front of this names the file, and a
// half-loaded model answering some clips would be a worse report than a load that said no.
bool WhisperASR::_load_graphs(const String &folder, const String &language, int num_threads) {
    Ref<DirAccess> dir = DirAccess::open(folder);
    if (dir.is_null()) {
        return false;
    }
    const PackedStringArray files = dir->get_files();

    const String vocab_name = pick(files, "vocab", ".txt");
    const String filters_name = pick(files, "fbank", ".ncnn.bin");
    if (vocab_name.is_empty() || filters_name.is_empty()) {
        return false;
    }

    struct Part {
        NcnnGraph *graph;
        const char *mark;
    };
    const Part parts[] = {
        {&encoder, "encoder"},
        {&decoder, "decoder"},
        {&embed_token, "embed_token"},
        {&embed_position, "embed_position"},
        {&proj_out, "proj_out"},
    };
    for (const Part &part : parts) {
        const String param_name = pick(files, part.mark, ".ncnn.param");
        if (param_name.is_empty()) {
            return false;
        }
        const String bin_name = param_name.trim_suffix(".param") + ".bin";
        if (!part.graph->load(folder.path_join(param_name), folder.path_join(bin_name),
                    num_threads)) {
            return false;
        }
    }

    cache_pairs = count_cache_pairs(decoder.param);
    if (cache_pairs <= 0 || cache_pairs > MOST_CACHE_PAIRS) {
        return false;
    }
    mel_frames = declared_frames(encoder.param);
    clip_samples = mel_frames * HOP;

    // The fbank graph's weights are its mel filterbank and nothing else -- one MemoryData
    // layer of 201 by 80 floats, stored raw. Anything else in that file is another export.
    const PackedByteArray filters = FileAccess::get_file_as_bytes(folder.path_join(filters_name));
    if (filters.size() != MEL_BANDS * SPECTRUM_BINS * (int)sizeof(float)) {
        return false;
    }
    mel_filters.resize((size_t)MEL_BANDS * SPECTRUM_BINS);
    memcpy(mel_filters.data(), filters.ptr(), (size_t)filters.size());
    fft.prepare(FFT_SIZE);

    short table[512];
    build_byte_decoder(table);
    const PackedByteArray raw = FileAccess::get_file_as_bytes(folder.path_join(vocab_name));
    if (raw.is_empty()) {
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
        return false;
    }
    // Measured off the projection rather than trusted: its output width is the whole token
    // table, specials included, and the one number that tells a large-v3 export apart.
    const int width = output_width();
    if ((int)vocab.size() != MULTILINGUAL_VOCAB || width != MULTILINGUAL_VOCAB + SPECIAL_TOKENS) {
        ERR_PRINT(vformat("Whisper export refused: %d vocabulary lines and a projection %d wide, "
                          "where the multilingual layout of tiny, base, small and medium is %d "
                          "and %d. Those four sizes are the only exports read here.",
                (int)vocab.size(), width, MULTILINGUAL_VOCAB,
                MULTILINGUAL_VOCAB + SPECIAL_TOKENS));
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
    no_speech = transcribe + NOSPEECH_AFTER_TRANSCRIBE;
    return true;
}

String WhisperASR::describe_family() const {
    return "Whisper (ncnn)";
}

// How wide the output projection answers, found by running it once on a zero vector of the
// width the token embedding answers. Both extracts failing is -1: two graphs that do not fit
// each other are halves of two exports, and a load that said no beats a decode that breaks.
int WhisperASR::output_width() {
    ncnn::Mat of_token;
    {
        ncnn::Mat token_in = owned_index(0);
        ncnn::Extractor ex = embed_token.net.create_extractor();
        ex.input("in0", token_in);
        if (ex.extract("out0", of_token) != 0 || of_token.empty()) {
            return -1;
        }
    }
    ncnn::Mat hidden;
    hidden.create(of_token.w, 1);
    if (hidden.empty()) {
        return -1;
    }
    hidden.fill(0.0f);
    ncnn::Mat logits;
    ncnn::Extractor ex = proj_out.net.create_extractor();
    ex.input("in0", hidden);
    if (ex.extract("out0", logits) != 0 || logits.empty()) {
        return -1;
    }
    return logits.w;
}

double WhisperASR::last_no_speech_prob() const {
    return no_speech_prob;
}

String WhisperASR::last_detected_language() const {
    return language_index[0] < 0 ? String() : String(LANGUAGES[language_index[0]]);
}

double WhisperASR::last_language_prob() const {
    return language_index[0] < 0 ? 0.0 : language_share[0];
}

Array WhisperASR::last_language_candidates() const {
    Array out;
    for (int i = 0; i < 3; i++) {
        if (language_index[i] < 0) {
            break;
        }
        Dictionary one;
        one["code"] = String(LANGUAGES[language_index[i]]);
        one["prob"] = language_share[i];
        out.push_back(one);
    }
    return out;
}

void WhisperASR::set_detect_language(bool enabled) {
    detecting_language.store(enabled);
}

bool WhisperASR::is_detecting_language() const {
    return detecting_language.load();
}

void WhisperASR::_unload_graphs() {
    encoder.clear();
    decoder.clear();
    embed_token.clear();
    embed_position.clear();
    proj_out.clear();
    mel_filters.clear();
    vocab.clear();
    prompt.clear();
    cache_pairs = 0;
}

void WhisperASR::_report_timings(Dictionary &out) const {
    out["mel_ms"] = mel_ms;
    out["encoder_ms"] = encoder_ms;
    out["decoder_ms"] = decoder_ms;
    out["tokens"] = token_count;
    out["no_speech_prob"] = no_speech_prob;
    out["detected_language"] = last_detected_language();
    out["language_prob"] = last_language_prob();
    out["language_candidates"] = last_language_candidates();
}

// Whisper's own log-mel, computed here rather than by the export's fbank graph: ncnn's
// Spectrogram layer costs more for one clip than the encoder does. The frames are split
// across the threads the model was loaded with, so the front end and the graphs run on the
// same number of cores.
void WhisperASR::log_mel(const std::vector<float> &audio, ncnn::Mat &mel) const {
    // Reflect padding by half a window, so frame t is centred on sample t * 160 exactly as
    // torch's centred stft has it. Without it every frame is half a window early.
    const int pad = FFT_SIZE / 2;
    std::vector<float> padded((size_t)clip_samples + 2 * pad);
    for (int i = 0; i < pad; i++) {
        padded[(size_t)i] = audio[(size_t)(pad - i)];
        padded[(size_t)clip_samples + pad + i] = audio[(size_t)clip_samples - 2 - i];
    }
    memcpy(padded.data() + pad, audio.data(), (size_t)clip_samples * sizeof(float));

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
        band(0, mel_frames);
    } else {
        std::vector<std::thread> pool;
        const int each = (mel_frames + workers - 1) / workers;
        for (int i = 1; i < workers; i++) {
            const int from = i * each;
            const int to = (i + 1) * each < mel_frames ? (i + 1) * each : mel_frames;
            if (from < to) {
                pool.emplace_back(band, from, to);
            }
        }
        band(0, each < mel_frames ? each : mel_frames);
        for (std::thread &one : pool) {
            one.join();
        }
    }

    // Whisper's normalisation, and the reason the maximum is taken over the whole matrix
    // rather than per frame: a per-frame floor would lift the silence between words.
    float top = -1.0e30f;
    for (int b = 0; b < MEL_BANDS; b++) {
        float *row = mel.row(b);
        for (int t = 0; t < mel_frames; t++) {
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
        for (int t = 0; t < mel_frames; t++) {
            row[t] = ((row[t] > floor ? row[t] : floor) + 4.0f) / 4.0f;
        }
    }
}

// One clip through the graphs. The samples arrive at sixteen kilohertz and are padded or
// trimmed to the window the encoder was exported for, which is the only length it has a shape
// for; the text is the bytes of every token run together and read as UTF-8 once.
String WhisperASR::_decode(const std::vector<float> &samples) {
    const double started = now_ms();
    no_speech_prob = 0.0;
    for (int i = 0; i < 3; i++) {
        language_index[i] = -1;
        language_share[i] = 0.0;
    }

    std::vector<float> audio((size_t)clip_samples, 0.0f);
    const size_t taken = samples.size() < (size_t)clip_samples ? samples.size() : (size_t)clip_samples;
    memcpy(audio.data(), samples.data(), taken * sizeof(float));

    ncnn::Mat mel;
    mel.create(mel_frames, MEL_BANDS);
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
    token_count = (int)written.size();
    // The bytes are UTF-8 only once the whole run is concatenated: a Cyrillic letter routinely
    // straddles two tokens, so decoding token by token would produce mojibake. Named as UTF-8
    // rather than handed to the plain constructor, which reads a byte as a Latin-1 letter and
    // turns every Cyrillic one into the two that spell it.
    return String::utf8(bytes.c_str(), (int64_t)bytes.length()).strip_edges();
}

// The greedy loop. One token per step: the prompt is walked through first so the caches hold
// it, then whatever the graph writes is fed back until it says it is finished. The caches
// stay ncnn's own objects and are handed straight back, and the mask grows one column per
// step -- a mask of the wrong width has the attention reading past the end of its row.
//
// An extract that fails ends the clip with no words at all rather than the words so far: a
// step run on a cache that was never written is a decode continuing from nothing, and half a
// sentence read as the whole of it is a worse answer than the seam's "nothing heard".
std::vector<int> WhisperASR::run_decoder(const ncnn::Mat &states) {
    const std::vector<int> nothing;
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
                return nothing;
            }
            ncnn::Mat position_in = owned_index(position);
            ncnn::Extractor ex2 = embed_position.net.create_extractor();
            ex2.input("in0", position_in);
            if (ex2.extract("out0", of_position) != 0 || of_position.w != of_token.w) {
                return nothing;
            }
            // The width the export was written at, taken from the lookup that just answered.
            embedding.create(of_token.w, 1);
            if (embedding.empty()) {
                return nothing;
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
            return nothing;
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
                return nothing;
            }
            for (int i = 0; i < cache_pairs; i++) {
                snprintf(name, sizeof(name), "out_cache_k%d", i);
                if (ex.extract(name, next[(size_t)i * 2]) != 0) {
                    return nothing;
                }
                snprintf(name, sizeof(name), "out_cache_v%d", i);
                if (ex.extract(name, next[(size_t)i * 2 + 1]) != 0) {
                    return nothing;
                }
            }
        }
        cache.swap(next);

        // The prompt is fed for its caches and, past the first token, never asked what comes
        // next. The step after the start token is the exception: it is where the model chooses
        // between the languages and the no-speech token, so it is projected for those numbers
        // and its choice is thrown away in favour of the language the host named.
        if (position == 0) {
            read_first_step(hidden);
        }
        if (position + 1 < (int)prompt.size()) {
            token = prompt[(size_t)position + 1];
            continue;
        }

        ncnn::Mat logits;
        {
            ncnn::Extractor ex = proj_out.net.create_extractor();
            ex.input("in0", hidden);
            if (ex.extract("out0", logits) != 0) {
                return nothing;
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
        // Anything above the vocabulary is grammar rather than a word -- the end, the no-speech
        // token, a timestamp -- and a greedy loop that fed one back would narrate from there.
        if (best >= (int)vocab.size()) {
            break;
        }
        written.push_back(best);
        token = best;
    }
    return written;
}

// The distribution after the start token, read for two things: the no-speech token's share of
// the whole of it, which is the model's own opinion of whether anybody spoke, and -- when asked
// -- the ninety-nine language tokens against each other, which is what the model took the
// language to be. Both taken against a maximum so the exponentials stay finite; a projection
// that fails leaves zero and nothing, which is a clip judged by its words alone.
//
// The language read is noisy on a clip of a second or two and tiny is noisier than base; it
// is information beside the text, and the transcription follows the language the host named.
void WhisperASR::read_first_step(const ncnn::Mat &hidden) {
    ncnn::Mat logits;
    ncnn::Extractor ex = proj_out.net.create_extractor();
    ex.input("in0", hidden);
    if (ex.extract("out0", logits) != 0 || no_speech >= logits.w) {
        return;
    }
    const float *scores = logits.row(0);
    float top = scores[0];
    for (int i = 1; i < logits.w; i++) {
        if (scores[i] > top) {
            top = scores[i];
        }
    }
    double sum = 0.0;
    for (int i = 0; i < logits.w; i++) {
        sum += exp((double)(scores[i] - top));
    }
    no_speech_prob = exp((double)(scores[no_speech] - top)) / sum;

    if (!detecting_language.load()) {
        return;
    }
    // The language tokens sit directly after the start token, in the order of the table.
    const int first = end_of_text + 2;
    if (first + LANGUAGE_COUNT > logits.w) {
        return;
    }
    float top_language = scores[first];
    for (int i = 1; i < LANGUAGE_COUNT; i++) {
        if (scores[first + i] > top_language) {
            top_language = scores[first + i];
        }
    }
    double language_sum = 0.0;
    for (int i = 0; i < LANGUAGE_COUNT; i++) {
        language_sum += exp((double)(scores[first + i] - top_language));
    }
    for (int rank = 0; rank < 3; rank++) {
        int best = -1;
        for (int i = 0; i < LANGUAGE_COUNT; i++) {
            bool taken = false;
            for (int r = 0; r < rank; r++) {
                taken = taken || language_index[r] == i;
            }
            if (!taken && (best < 0 || scores[first + i] > scores[first + best])) {
                best = i;
            }
        }
        language_index[rank] = best;
        language_share[rank] = exp((double)(scores[first + best] - top_language)) / language_sum;
    }
}

void WhisperASR::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_detect_language", "enabled"),
            &WhisperASR::set_detect_language);
    ClassDB::bind_method(D_METHOD("is_detecting_language"), &WhisperASR::is_detecting_language);
}
