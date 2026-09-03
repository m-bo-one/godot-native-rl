#ifndef GIGAAM_ASR_H
#define GIGAAM_ASR_H

#include "ncnn_asr.h"
#include "small_fft.h"

#include <string>
#include <vector>

namespace godot {

// A GigaAM v3 CTC export as ncnn runs it: one graph and a table of thirty-four classes,
// decoded in a single pass. The threading, the resampling and the delivery are the base's;
// what is here is the front end of this family's own shape, the two rotary tables the graph
// takes as inputs, and the argmax-and-collapse that turns the frames into letters.
//
// The model reads one language and nothing else, so the `language` a load is handed is
// accepted and never read: the seam asks one of every recogniser, and this one has no use
// for it. There is no no-speech share and no language read either -- a CTC head writes
// blanks over a silent room rather than an opinion of it -- so both answer the base's none.
class GigaAMASR : public NcnnASR {
    GDCLASS(GigaAMASR, NcnnASR)

    NcnnGraph graph;

    // The window and the filterbank the checkpoint carries, read out of the file beside the
    // graph rather than recomputed, so this front end and the exporter's reference agree on
    // every coefficient: 320 window values, then 64 filters of 161 bins each.
    std::vector<float> window;
    std::vector<float> mel_filters;
    SmallFft fft;

    // Every class as the UTF-8 it stands for, in the order the head numbers them, and which
    // of them is the blank. Read off the token file: a table written here would decode this
    // export and answer another vocabulary with the wrong letters.
    std::vector<std::string> tokens;
    int blank = -1;

    // What the last decode cost, in milliseconds, with the positions the encoder answered
    // and the letters written. Written by whichever thread decoded and read after it has
    // finished, which the busy flag orders; a read during a decode sees the previous clip.
    double mel_ms = 0.0;
    double encoder_ms = 0.0;
    double decode_ms = 0.0;
    int frame_count = 0;
    int token_count = 0;

protected:
    static void _bind_methods();

    bool _load_graphs(const String &folder, const String &language, int num_threads) override;
    String _decode(const std::vector<float> &samples) override;
    void _unload_graphs() override;
    void _report_timings(Dictionary &out) const override;

public:
    GigaAMASR() = default;
    ~GigaAMASR();

    String describe_family() const override;

private:
    int output_width();
    bool run_graph(const ncnn::Mat &mel, ncnn::Mat &logits);
    void log_mel(const float *audio, int frames, ncnn::Mat &mel) const;
};

} // namespace godot

#endif // GIGAAM_ASR_H
