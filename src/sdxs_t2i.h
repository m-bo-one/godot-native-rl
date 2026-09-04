#ifndef SDXS_T2I_H
#define SDXS_T2I_H

#include "ncnn_t2i.h"

namespace godot {

// A one-step diffusion export as ncnn runs it: a text encoder, a denoiser and a tiny decoder,
// one prompt in, one picture out. The threading, the tokeniser, the schedule and the pixels are
// the base's; what is here is the three nets and which of the denoiser's structures a size
// reaches.
//
// The timestep is baked into the denoiser rather than taken as an input, which is what a
// one-step model allows: the sinusoidal embedding and the two time projections fold away at
// trace time, and with them every broadcast of a timestep vector onto a feature map -- none of
// which this runtime lowers well. The cost is that the export is that one step and no other.
//
// One weight file stands behind every size the folder offers, because the denoiser's weights do
// not depend on the picture's size -- only the shapes written into the structure do. So a size
// is a second .param over the same .bin, and changing size re-reads the structure rather than
// the weights.
class SdxsT2I : public NcnnT2I {
    GDCLASS(SdxsT2I, NcnnT2I)

    NcnnGraph clip;
    NcnnGraph unet;
    NcnnGraph dec;

    // The folder, kept so a size change can find the structure to read; and which size the
    // denoiser currently holds, so a run at that size reads nothing.
    String folder;
    int unet_width = 0;
    int unet_height = 0;

    // What the last picture cost inside the family. Written by whichever thread ran and read
    // after it has finished, which the busy flag orders.
    double swap_ms = 0.0;

protected:
    static void _bind_methods();

    bool _load_graphs(const String &folder, const Dictionary &manifest, int num_threads,
            String &problem) override;
    void _unload_graphs() override;
    bool _encode_text(const int *ids, int count, ncnn::Mat &hidden, String &problem) override;
    bool _denoise(const ncnn::Mat &latent, const ncnn::Mat &hidden, float timestep, int width,
            int height, ncnn::Mat &out, String &problem) override;
    bool _decode_latent(const ncnn::Mat &latent, int width, int height, ncnn::Mat &rgb,
            String &problem) override;
    bool _prepare_size(int width, int height, String &problem) override;
    void _report_timings(Dictionary &out) const override;

public:
    SdxsT2I() = default;
    ~SdxsT2I();

    String describe() const override;

private:
    // The fragment the structure for one size is named by. A size is part of a file name here
    // rather than a row in the manifest, so a folder with a third size in it needs no new key.
    static String mark_for(int width, int height);
    static bool read_pair(const PackedStringArray &files, const String &model_dir,
            const char *mark, const char *what, NcnnGraph &into, int num_threads, bool fp16,
            String &problem);
};

} // namespace godot

#endif // SDXS_T2I_H
