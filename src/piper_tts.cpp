// The three layer classes below and the length regulator are ports of the reference ncnn
// implementation of Piper by nihui, which is published under the BSD 3-Clause License:
//
//   Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
//   https://opensource.org/licenses/BSD-3-Clause
//
// The full text travels with the addon in addons/govorilka/ncnn/THIRD_PARTY_LICENSES.md.

#include "piper_tts.h"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>

#include <layer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace godot;

namespace {

// The five parts of an export and the two small files beside them, matched as fragments with
// the part's own underscore in front: an export ships them under whatever prefix it was
// written with, and every published one names the parts this way.
const char *ENC_MARK = "_enc_p";
const char *DP_MARK = "_dp";
const char *FLOW_MARK = "_flow";
const char *DEC_MARK = "_dec";
const char *EMB_MARK = "_emb_g";
const char *CONFIG_MARK = "config";
const char *PARAM_SUFFIX = ".ncnn.param";
const char *BIN_SUFFIX = ".ncnn.bin";
const char *TXT_SUFFIX = ".txt";

// The names the export gives the two modules pnnx was told to keep whole. They are the
// module's own dotted path in the training package, and the graph names its layers by it.
const char *REL_K = "piper.train.vits.attentions.relative_embeddings_k_module";
const char *REL_V = "piper.train.vits.attentions.relative_embeddings_v_module";
const char *SPLINE = "piper.train.vits.modules.piecewise_rational_quadratic_transform_module";

// The attention window either side of a position, and the spline's bin count and support.
// They are the architecture's own constants rather than anything read off a file: a checkpoint
// trained with other ones is a different network and would need another layer, not another
// number here.
const int WINDOW_SIZE = 4;
const int NUM_BINS = 10;
const int FILTER_CHANNELS = 192;
const float TAIL_BOUND = 5.0f;
const float MIN_BIN_WIDTH = 1e-3f;
const float MIN_BIN_HEIGHT = 1e-3f;
const float MIN_DERIVATIVE = 1e-3f;

// The relative-position keys gathered onto the sequence, which is a scatter with a moving
// offset -- ncnn has no operator that says it, so the export keeps the module whole and this
// stands in for it. It writes a (len, len) band out of a (wsize, len) strip per head.
class RelativeEmbeddingsK : public ncnn::Layer {
public:
    RelativeEmbeddingsK() { one_blob_only = true; }

    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob,
            const ncnn::Option &opt) const {
        const int wsize = bottom_blob.w;
        const int len = bottom_blob.h;
        const int num_heads = bottom_blob.c;

        top_blob.create(len, len, num_heads, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
        top_blob.fill(0.f);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++) {
            const ncnn::Mat x0 = bottom_blob.channel(q);
            ncnn::Mat out0 = top_blob.channel(q);

            for (int i = 0; i < len; i++) {
                const float *xptr = x0.row(i) + std::max(0, WINDOW_SIZE - i);
                float *outptr = out0.row(i) + std::max(i - WINDOW_SIZE, 0);
                const int span = std::min(len, i - WINDOW_SIZE + wsize) - std::max(i - WINDOW_SIZE, 0);
                for (int j = 0; j < span; j++) {
                    *outptr++ = *xptr++;
                }
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(RelativeEmbeddingsK)

// The same gather run the other way, for the values: a (wsize, len) strip out of the
// (len, len) band, which is what the attention weights are then read against.
class RelativeEmbeddingsV : public ncnn::Layer {
public:
    RelativeEmbeddingsV() { one_blob_only = true; }

    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob,
            const ncnn::Option &opt) const {
        const int wsize = WINDOW_SIZE * 2 + 1;
        const int len = bottom_blob.h;
        const int num_heads = bottom_blob.c;

        top_blob.create(wsize, len, num_heads, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }
        top_blob.fill(0.f);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++) {
            const ncnn::Mat x0 = bottom_blob.channel(q);
            ncnn::Mat out0 = top_blob.channel(q);

            for (int i = 0; i < len; i++) {
                const float *xptr = x0.row(i) + std::max(i - WINDOW_SIZE, 0);
                float *outptr = out0.row(i) + std::max(0, WINDOW_SIZE - i);
                const int span = std::min(len, i - WINDOW_SIZE + wsize) - std::max(i - WINDOW_SIZE, 0);
                for (int j = 0; j < span; j++) {
                    *outptr++ = *xptr++;
                }
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(RelativeEmbeddingsV)

// The monotonic rational-quadratic spline the duration predictor's flow inverts. Ten bins
// whose widths, heights and slopes come from the conditioning, a bin found by a search, and a
// quadratic solved per element -- a data-dependent branch and a search, neither of which is a
// tensor operator. Outside the support the transform is the identity, which is why an element
// past the tail bound is copied and skipped.
class PiecewiseRationalQuadratic : public ncnn::Layer {
public:
    PiecewiseRationalQuadratic() { one_blob_only = false; }

    virtual int forward(const std::vector<ncnn::Mat> &bottom_blobs,
            std::vector<ncnn::Mat> &top_blobs, const ncnn::Option &opt) const {
        const ncnn::Mat &h = bottom_blobs[0];
        const ncnn::Mat &x1 = bottom_blobs[1];
        ncnn::Mat &outputs = top_blobs[0];

        const int count = x1.w;
        outputs = x1.clone(opt.blob_allocator);
        if (outputs.empty()) {
            return -100;
        }
        float *out_ptr = outputs;

        const float inv_sqrt_channels = 1.0f / sqrtf((float)FILTER_CHANNELS);
        const float edge = logf(expf(1.f - MIN_DERIVATIVE) - 1.f);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < count; i++) {
            const float x = ((const float *)x1)[i];
            if (x < -TAIL_BOUND || x > TAIL_BOUND) {
                continue;
            }
            const float *params = h.row(i);

            float widths[NUM_BINS];
            float heights[NUM_BINS];
            float derivatives[NUM_BINS + 1];
            float cumwidths[NUM_BINS + 1];
            float cumheights[NUM_BINS + 1];

            softmax_bins(params, inv_sqrt_channels, MIN_BIN_WIDTH, widths);
            softmax_bins(params + NUM_BINS, inv_sqrt_channels, MIN_BIN_HEIGHT, heights);
            accumulate(widths, cumwidths);
            accumulate(heights, cumheights);

            derivatives[0] = edge;
            derivatives[NUM_BINS] = edge;
            for (int j = 0; j < NUM_BINS - 1; j++) {
                derivatives[j + 1] = params[2 * NUM_BINS + j];
            }
            for (int j = 0; j < NUM_BINS + 1; j++) {
                derivatives[j] = MIN_DERIVATIVE + softplus(derivatives[j]);
            }

            // The inverse is the one direction the export needs, so the bin is found on the
            // heights: the forward direction would search the widths instead.
            int bin = 0;
            while (bin < NUM_BINS && cumheights[bin + 1] <= x) {
                bin++;
            }
            bin = std::max(0, std::min(bin, NUM_BINS - 1));

            const float bin_width = cumwidths[bin + 1] - cumwidths[bin];
            const float bin_height = cumheights[bin + 1] - cumheights[bin];
            const float d0 = derivatives[bin];
            const float d1 = derivatives[bin + 1];
            const float delta = bin_height / bin_width;
            const float offset = x - cumheights[bin];

            const float a = offset * (d0 + d1 - 2 * delta) + bin_height * (delta - d0);
            const float b = bin_height * d0 - offset * (d0 + d1 - 2 * delta);
            const float c = -delta * offset;
            float discriminant = b * b - 4 * a * c;
            discriminant = std::max(0.f, discriminant);
            const float root = (2 * c) / (-b - sqrtf(discriminant));
            out_ptr[i] = root * bin_width + cumwidths[bin];
        }

        return 0;
    }

private:
    static float softplus(float x) {
        return x > 0 ? x + logf(1.f + expf(-x)) : logf(1.f + expf(x));
    }

    // A softmax over the bins, floored so no bin can close entirely: a zero-width bin is a
    // division by zero in the quadratic below it.
    static void softmax_bins(const float *raw, float scale, float floor_value, float *out) {
        float largest = -INFINITY;
        for (int j = 0; j < NUM_BINS; j++) {
            largest = std::max(largest, raw[j] * scale);
        }
        float total = 0.f;
        for (int j = 0; j < NUM_BINS; j++) {
            out[j] = expf(raw[j] * scale - largest);
            total += out[j];
        }
        for (int j = 0; j < NUM_BINS; j++) {
            out[j] = floor_value + (1.f - floor_value * NUM_BINS) * (out[j] / total);
        }
    }

    // The bin edges over the support, both ends pinned: walking the sum to the last edge
    // instead would leave the far end a rounding error short of the bound.
    static void accumulate(const float *sizes, float *edges) {
        edges[0] = -TAIL_BOUND;
        float running = 0.f;
        for (int j = 0; j < NUM_BINS - 1; j++) {
            running += sizes[j];
            edges[j + 1] = -TAIL_BOUND + 2.f * TAIL_BOUND * running;
        }
        edges[NUM_BINS] = TAIL_BOUND;
    }
};

DEFINE_LAYER_CREATOR(PiecewiseRationalQuadratic)

// One `"name"=value` line of the folder's config, or nothing. The value keeps whatever the
// writer put after the equals sign, trailing `f` and quotes included, and the readers below
// strip what they do not want.
String config_value(const String &text, const String &name) {
    const PackedStringArray lines = text.split("\n");
    const String wanted = "\"" + name + "\"=";
    for (int i = 0; i < lines.size(); i++) {
        const String line = lines[i].strip_edges();
        if (line.begins_with(wanted)) {
            return line.substr(wanted.length()).strip_edges().trim_suffix("f").trim_prefix("\"")
                    .trim_suffix("\"");
        }
    }
    return String();
}

float config_float(const String &text, const String &name, float fallback) {
    const String found = config_value(text, name);
    return found.is_empty() ? fallback : (float)found.to_float();
}

int config_int(const String &text, const String &name, int fallback) {
    const String found = config_value(text, name);
    return found.is_empty() ? fallback : found.to_int();
}

} // namespace

PiperTTS::~PiperTTS() {
    unload();
}

bool PiperTTS::_load_graphs(const String &folder, int num_threads) {
    Ref<DirAccess> dir = DirAccess::open(folder);
    if (dir.is_null()) {
        return false;
    }
    PackedStringArray files = dir->get_files();
    files.sort();

    const String config_name = pick(files, CONFIG_MARK, TXT_SUFFIX);
    if (config_name.is_empty()) {
        return false;
    }
    const String config = FileAccess::get_file_as_string(folder.path_join(config_name));
    rate = config_int(config, "sample_rate", 0);
    speakers = std::max(1, config_int(config, "num_speakers", 1));
    noise_scale = config_float(config, "noise_scale", 0.667f);
    length_scale = config_float(config, "length_scale", 1.0f);
    noise_w = config_float(config, "noise_w", 0.8f);
    if (rate <= 0) {
        return false;
    }

    // The two modules pnnx was told to keep whole are registered before the structure is
    // parsed: register_custom_layer after load_param finds the layer already refused.
    enc_p.prepare(num_threads);
    enc_p.net.register_custom_layer(REL_K, RelativeEmbeddingsK_layer_creator);
    enc_p.net.register_custom_layer(REL_V, RelativeEmbeddingsV_layer_creator);
    dp.prepare(num_threads);
    dp.net.register_custom_layer(SPLINE, PiecewiseRationalQuadratic_layer_creator);

    struct Part {
        NcnnGraph *graph;
        const char *mark;
        bool prepared;
    };
    const Part parts[] = {
        {&enc_p, ENC_MARK, true},
        {&dp, DP_MARK, true},
        {&flow, FLOW_MARK, false},
        {&dec, DEC_MARK, false},
    };
    for (const Part &part : parts) {
        const String param_name = pick(files, part.mark, PARAM_SUFFIX);
        const String bin_name = pick(files, part.mark, BIN_SUFFIX);
        if (param_name.is_empty() || bin_name.is_empty()) {
            return false;
        }
        if (!part.prepared) {
            part.graph->prepare(num_threads);
        }
        if (!part.graph->read(folder.path_join(param_name), folder.path_join(bin_name))) {
            return false;
        }
    }

    // What the decoder takes says whether this export has voices: a single-voice model takes
    // the sampled sequence alone, a multi-voice one takes the speaker embedding beside it.
    has_speakers = dec.net.input_indexes().size() == 2;
    if (!has_speakers) {
        speakers = 1;
        return true;
    }

    const String emb_param = pick(files, EMB_MARK, PARAM_SUFFIX);
    const String emb_bin = pick(files, EMB_MARK, BIN_SUFFIX);
    if (emb_param.is_empty() || emb_bin.is_empty()) {
        return false;
    }
    return emb_g.load(folder.path_join(emb_param), folder.path_join(emb_bin), num_threads);
}

void PiperTTS::_unload_graphs() {
    emb_g.clear();
    enc_p.clear();
    dp.clear();
    flow.clear();
    dec.clear();
    has_speakers = false;
    speakers = 1;
    rate = 0;
}

PackedFloat32Array PiperTTS::_synthesise(const PackedInt32Array &ids, int speaker,
        String &problem) {
    if (speaker < 0 || speaker >= speakers) {
        problem = String("Govorilka: this model holds {0} voice(s), so there is no voice {1} "
                         "to speak in.").format(Array::make(speakers, speaker));
        return PackedFloat32Array();
    }

    TtsNoise noise(seed);
    symbol_count = ids.size();

    // enc_p: the symbols to their encodings and to the mean and spread the sampler draws from.
    ncnn::Mat x;
    ncnn::Mat m_p;
    ncnn::Mat logs_p;
    double at = now_ms();
    {
        ncnn::Mat sequence = owned_indices((const int *)ids.ptr(), ids.size());
        ncnn::Extractor ex = enc_p.net.create_extractor();
        ex.input("in0", sequence);
        if (ex.extract("out0", x) != 0 || ex.extract("out1", m_p) != 0
                || ex.extract("out2", logs_p) != 0) {
            problem = String("Govorilka: the encoder graph refused this sentence.");
            return PackedFloat32Array();
        }
    }
    enc_ms = now_ms() - at;

    // emb_g: the chosen voice as one embedding, handed to the three graphs below.
    ncnn::Mat g;
    if (has_speakers) {
        ncnn::Mat index = owned_index(speaker);
        ncnn::Extractor ex = emb_g.net.create_extractor();
        ex.input("in0", index);
        if (ex.extract("out0", g) != 0) {
            problem = String("Govorilka: the voice embedding graph refused voice {0}.")
                              .format(Array::make(speaker));
            return PackedFloat32Array();
        }
        g = g.reshape(1, g.w);
    }

    // dp: how long each symbol is held, as a log duration. The noise it samples with is a
    // graph input rather than a layer, which is what makes the seed reach it.
    ncnn::Mat logw;
    at = now_ms();
    {
        ncnn::Mat draw(x.w, 2);
        if (draw.empty()) {
            problem = String("Govorilka: there was no room for the duration noise.");
            return PackedFloat32Array();
        }
        for (int i = 0; i < draw.w * draw.h; i++) {
            draw[i] = noise.normal() * noise_w;
        }
        ncnn::Extractor ex = dp.net.create_extractor();
        ex.input("in0", x);
        ex.input("in1", draw);
        if (has_speakers) {
            ex.input("in2", g);
        }
        if (ex.extract("out0", logw) != 0) {
            problem = String("Govorilka: the duration graph refused this sentence.");
            return PackedFloat32Array();
        }
    }
    dp_ms = now_ms() - at;

    ncnn::Mat z_p;
    regulate_length(logw, m_p, logs_p, noise, z_p);
    if (z_p.empty()) {
        problem = String("Govorilka: the sentence came to no frames at all. A length_scale "
                         "near zero holds every symbol for no time.");
        return PackedFloat32Array();
    }

    // flow: the sampled sequence back through the normalising flow, into what the decoder
    // was trained to read.
    ncnn::Mat z;
    at = now_ms();
    {
        ncnn::Extractor ex = flow.net.create_extractor();
        ex.input("in0", z_p);
        if (has_speakers) {
            ex.input("in1", g);
        }
        if (ex.extract("out0", z) != 0) {
            problem = String("Govorilka: the flow graph refused this sentence.");
            return PackedFloat32Array();
        }
    }
    flow_ms = now_ms() - at;

    // dec: the waveform.
    ncnn::Mat out;
    at = now_ms();
    {
        ncnn::Extractor ex = dec.net.create_extractor();
        ex.input("in0", z);
        if (has_speakers) {
            ex.input("in1", g);
        }
        if (ex.extract("out0", out) != 0) {
            problem = String("Govorilka: the decoder graph refused this sentence.");
            return PackedFloat32Array();
        }
    }
    dec_ms = now_ms() - at;

    // Clipped and never scaled. Normalising each sentence to full scale is what the reference
    // runtime does, and in a conversation it pumps: a short quiet line comes out as loud as a
    // shouted one. Whoever plays these decides the loudness of the character, once.
    frame_count = out.w;
    PackedFloat32Array samples;
    samples.resize(out.w);
    float *write = samples.ptrw();
    for (int i = 0; i < out.w; i++) {
        write[i] = std::min(std::max(out[i], -1.f), 1.f);
    }
    return samples;
}

void PiperTTS::regulate_length(const ncnn::Mat &logw, const ncnn::Mat &m_p,
        const ncnn::Mat &logs_p, TtsNoise &noise, ncnn::Mat &z_p) const {
    const int symbols = logw.w;
    const int depth = m_p.h;

    std::vector<int> held((size_t)symbols);
    int frames = 0;
    for (int i = 0; i < symbols; i++) {
        held[(size_t)i] = (int)ceilf(expf(logw[i]) * length_scale);
        held[(size_t)i] = std::max(0, held[(size_t)i]);
        frames += held[(size_t)i];
    }
    if (frames <= 0) {
        z_p = ncnn::Mat();
        return;
    }

    z_p.create(frames, depth);
    if (z_p.empty()) {
        return;
    }

    for (int i = 0; i < depth; i++) {
        const float *mean = m_p.row(i);
        const float *spread = logs_p.row(i);
        float *write = z_p.row(i);

        for (int j = 0; j < symbols; j++) {
            const float centre = mean[j];
            const float scale = expf(spread[j]) * noise_scale;
            const int duration = held[(size_t)j];

            for (int k = 0; k < duration; k++) {
                write[k] = centre + noise.normal() * scale;
            }
            write += duration;
        }
    }
}

void PiperTTS::_report_timings(Dictionary &out) const {
    out["enc_ms"] = enc_ms;
    out["dp_ms"] = dp_ms;
    out["flow_ms"] = flow_ms;
    out["dec_ms"] = dec_ms;
    out["symbols"] = symbol_count;
    out["frames"] = frame_count;
}

String PiperTTS::describe_family() const {
    return "Piper VITS (ncnn)";
}

int PiperTTS::sample_rate() const {
    return rate;
}

int PiperTTS::speaker_count() const {
    return speakers;
}

void PiperTTS::set_noise_scale(double value) {
    noise_scale = (float)value;
}

double PiperTTS::get_noise_scale() const {
    return noise_scale;
}

void PiperTTS::set_length_scale(double value) {
    length_scale = (float)value;
}

double PiperTTS::get_length_scale() const {
    return length_scale;
}

void PiperTTS::set_noise_w(double value) {
    noise_w = (float)value;
}

double PiperTTS::get_noise_w() const {
    return noise_w;
}

void PiperTTS::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_noise_scale", "value"), &PiperTTS::set_noise_scale);
    ClassDB::bind_method(D_METHOD("get_noise_scale"), &PiperTTS::get_noise_scale);
    ClassDB::bind_method(D_METHOD("set_length_scale", "value"), &PiperTTS::set_length_scale);
    ClassDB::bind_method(D_METHOD("get_length_scale"), &PiperTTS::get_length_scale);
    ClassDB::bind_method(D_METHOD("set_noise_w", "value"), &PiperTTS::set_noise_w);
    ClassDB::bind_method(D_METHOD("get_noise_w"), &PiperTTS::get_noise_w);

    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "noise_scale"), "set_noise_scale",
            "get_noise_scale");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length_scale"), "set_length_scale",
            "get_length_scale");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "noise_w"), "set_noise_w", "get_noise_w");
}
