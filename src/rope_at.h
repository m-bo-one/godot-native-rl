// Copyright 2026 Futz12 <pchar.cn>
// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause
//
// The layer below is ncnn's own `RotaryEmbed` -- `src/layer/rotaryembed.cpp` and
// `src/layer/vulkan/shader/rotaryembed.comp` -- carried here almost verbatim and changed in one
// place, so it keeps the notice its authors put on it. The same notice covers the shipped binary
// in `addons/govorilka/ncnn/THIRD_PARTY_LICENSES.md`.

#ifndef ROPE_AT_H
#define ROPE_AT_H

#include <godot_cpp/variant/string.hpp>

#include <layer.h>
#include <mat.h>
#include <pipeline.h>

// ncnn's platform header pulls in windows.h, whose CONNECT_DEFERRED collides with Godot's
// Object::ConnectFlags member of the same name. Dropped for the same reason ncnn_graph.h drops
// it, and nothing here calls a WNet function.
#ifdef CONNECT_DEFERRED
#undef CONNECT_DEFERRED
#endif

namespace godot {

// A rotary embedding that reads its cos and sin tables at the tables' own stride on both
// devices. It replaces ncnn's built-in `RotaryEmbed` by type name, which the library's own
// registration seam allows, so a graph carrying that layer gets this one without being exported
// any differently.
//
// The built-in is right on the processor -- `cos_cache.row(i)`, the table's stride -- and wrong
// on the card: `rotaryembed.comp` indexes `row * embed_dim / 2`. A table wider than half the
// embedding therefore reaches the shader at the wrong rate, and positions read each other's
// angles with nothing reported. Here the row width travels to the shader as a constant.
class RopeAt : public ncnn::Layer {
public:
    RopeAt();

    int load_param(const ncnn::ParamDict &pd) override;

    int forward(const std::vector<ncnn::Mat> &bottom_blobs, std::vector<ncnn::Mat> &top_blobs,
                const ncnn::Option &opt) const override;

#if NCNN_VULKAN
    int create_pipeline(const ncnn::Option &opt) override;
    int destroy_pipeline(const ncnn::Option &opt) override;

    int forward(const std::vector<ncnn::VkMat> &bottom_blobs, std::vector<ncnn::VkMat> &top_blobs,
                ncnn::VkCompute &cmd, const ncnn::Option &opt) const override;
#endif

    // Param 0, the built-in's only one: pairs are adjacent rather than half a row apart.
    int interleaved = 0;

private:
#if NCNN_VULKAN
    ncnn::Pipeline *pipeline = nullptr;
#endif
};

namespace rope_at {

// Handed to `Net::register_custom_layer("RotaryEmbed", ...)`, which the library documents as
// overwriting a built-in when the name is one of its own.
ncnn::Layer *create(void *userdata);
void destroy(ncnn::Layer *layer, void *userdata);

// Puts the layer above in front of the built-in for one net. It has to precede the parse: the
// registry is read as the structure names its layers, so a call after `load_param` changes
// nothing and says so nowhere.
//
// `param_text` is the structure about to be parsed, when the caller has it. The registration is
// then made only for a graph that actually names the layer, because the library announces every
// overwrite on stderr and a line per model load is noise about nothing. A null text registers
// unconditionally, for the road that holds a path rather than the bytes.
void install(ncnn::Net &net, const char *param_text = nullptr);

// The same, for a road that holds a path rather than the bytes: the structure is read once to
// see whether it names the layer at all.
void install_for_file(ncnn::Net &net, const String &param_path);

// How many nets in this process have had the layer put in front of the built-in. It is the one
// way to hold "a graph that does not name it registers nothing" without reading stderr.
int registrations();

} // namespace rope_at

} // namespace godot

#endif // ROPE_AT_H
