#ifndef PIPER_TTS_H
#define PIPER_TTS_H

#include "ncnn_tts.h"

namespace godot {

// A Piper VITS export as ncnn runs it: five graphs, a run of symbol ids in, one sentence of
// samples out. The threading and the delivery are the base's; what is here is the five nets,
// the three layers ncnn has no operator for, and the length regulator that stands where the
// data-dependent part of the network was cut out.
//
// The regulator is the reason this is a class and not a graph. VITS repeats each symbol's
// encoding by a duration the network itself predicts, and a graph whose shapes depend on its
// own output is not a graph any of these runtimes will lower. So the duration predictor ends
// at the durations, this repeats them, and the flow and the decoder take the result.
//
// The noise the sampler needs is drawn here rather than inside the graphs: the export promotes
// it to a graph input for exactly that reason, and drawing it here is what lets a seed make
// the same sentence twice.
class PiperTTS : public NcnnTTS {
    GDCLASS(PiperTTS, NcnnTTS)

    NcnnGraph emb_g;
    NcnnGraph enc_p;
    NcnnGraph dp;
    NcnnGraph flow;
    NcnnGraph dec;

    // Whether the export carries a speaker embedding. A single-voice model has no emb_g and
    // its other four graphs take one input fewer, which is read off the decoder's own inputs
    // rather than assumed from the folder's contents.
    bool has_speakers = false;
    int speakers = 1;
    int rate = 0;

    // What the checkpoint was published with, read out of the folder's config: how much noise
    // the sampler adds, how much the duration predictor takes, and how the durations are
    // stretched. A host moves length_scale to make a character speak faster or slower.
    float noise_scale = 0.667f;
    float length_scale = 1.0f;
    float noise_w = 0.8f;

    // What the last sentence cost, in milliseconds, with the symbols in and the samples out.
    // Written by whichever thread ran and read after it has finished, which the busy flag
    // orders; a read during a run sees the sentence before it.
    double enc_ms = 0.0;
    double dp_ms = 0.0;
    double flow_ms = 0.0;
    double dec_ms = 0.0;
    int symbol_count = 0;
    int frame_count = 0;

protected:
    static void _bind_methods();

    bool _load_graphs(const String &folder, int num_threads) override;
    PackedFloat32Array _synthesise(const PackedInt32Array &ids, int speaker,
            String &problem) override;
    void _unload_graphs() override;
    void _report_timings(Dictionary &out) const override;

public:
    PiperTTS() = default;
    ~PiperTTS();

    String describe_family() const override;
    int sample_rate() const override;
    int speaker_count() const override;

    void set_noise_scale(double value);
    double get_noise_scale() const;
    void set_length_scale(double value);
    double get_length_scale() const;
    void set_noise_w(double value);
    double get_noise_w() const;

private:
    // The durations the predictor answered, turned into the sampled sequence the flow takes:
    // each symbol's mean and spread repeated as many frames as it was given, with the noise
    // drawn per frame. Serial on purpose -- the draws are a sequence, and a thread pool would
    // shuffle them and lose the one property a seed buys.
    void regulate_length(const ncnn::Mat &logw, const ncnn::Mat &m_p, const ncnn::Mat &logs_p,
            TtsNoise &noise, ncnn::Mat &z_p) const;
};

} // namespace godot

#endif // PIPER_TTS_H
