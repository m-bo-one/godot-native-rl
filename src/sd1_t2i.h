#ifndef SD1_T2I_H
#define SD1_T2I_H

#include "ncnn_t2i.h"

namespace godot {

// A multi-step diffusion export as ncnn runs it, of the shape a latent-diffusion checkpoint of
// the first generation has: the same three nets as the one-step family, with the timestep taken
// as an input rather than baked into the structure. That one difference is what lets a schedule
// of several steps run through one graph, and it is why this is a second family rather than a
// flag on the first.
//
// Which arithmetic joins the steps is the manifest's to say -- Euler over sigmas, or the
// consistency model's own update -- and so is whether a second pass without the prompt is
// weighed against the first. Neither is written here: the base holds both roads, this holds
// the graphs.
class Sd1T2I : public NcnnT2I {
    GDCLASS(Sd1T2I, NcnnT2I)

    NcnnGraph clip;
    NcnnGraph unet;
    NcnnGraph dec;

    String folder;
    int unet_width = 0;
    int unet_height = 0;
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
    Sd1T2I() = default;
    ~Sd1T2I();

    String describe() const override;

private:
    static String mark_for(int width, int height);
    static bool read_pair(const PackedStringArray &files, const String &model_dir,
            const char *mark, const char *what, NcnnGraph &into, int num_threads, bool fp16,
            String &problem);
};

} // namespace godot

#endif // SD1_T2I_H
