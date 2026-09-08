// Copyright 2026 Futz12 <pchar.cn>
// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause
//
// ncnn's `RotaryEmbed`, carried here almost verbatim: the processor path is `rotaryembed.cpp`
// unchanged and the shader is `rotaryembed.comp` with one index and one constant altered. The
// header says why; this notice is the condition on which those sources may be copied at all.

#include "rope_at.h"

#include <godot_cpp/classes/file_access.hpp>

#include <atomic>

// The registration reaches into Net, which layer.h only forward-declares.
#include <net.h>

#include <cstring>

#if NCNN_VULKAN
#include <gpu.h>
#endif

using namespace godot;

namespace {

#if NCNN_VULKAN

// ncnn's own rotaryembed.comp with one line changed and one constant added. The library splits
// a source on its `#version 450` line and injects its macro preamble -- sfp, afp, psc, buffer_ld1
// -- so this is compiled exactly as the built-in shaders are, and reads at half precision where
// the option asks for it.
//
// The change is `cache_offset`: the built-in indexes the tables as `gy * halfdim + gx`, which is
// only the row stride when the table is exactly half the embedding wide. Here the width arrives
// as `cache_w` and the shader reads the row the processor reads.
const char *ROPE_AT_COMP = R"GLSL(#version 450

layout(constant_id = 0) const int interleaved = 0;

#define shape_constant_id_offset 1
layout(constant_id = shape_constant_id_offset + 0) const int embed_dim = 0;
layout(constant_id = shape_constant_id_offset + 1) const int seqlen = 0;
layout(constant_id = shape_constant_id_offset + 2) const int num_heads = 0;
layout(constant_id = shape_constant_id_offset + 3) const int cstep = 0;
layout(constant_id = shape_constant_id_offset + 4) const int cache_w = 0;

layout(binding = 0) readonly buffer bottom_blob { sfp bottom_blob_data[]; };
layout(binding = 1) readonly buffer cos_cache { sfp cos_cache_data[]; };
layout(binding = 2) readonly buffer sin_cache { sfp sin_cache_data[]; };
layout(binding = 3) writeonly buffer top_blob { sfp top_blob_data[]; };

layout(push_constant) uniform parameter
{
    int embed_dim;
    int seqlen;
    int num_heads;
    int cstep;
    int cache_w;
} p;

void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);

    const int halfdim = psc(embed_dim) / 2;

    if (gx >= halfdim || gy >= psc(seqlen) || gz >= psc(num_heads))
        return;

    const int cache_offset = gy * psc(cache_w) + gx;

    afp cosv = buffer_ld1(cos_cache_data, cache_offset);
    afp sinv = buffer_ld1(sin_cache_data, cache_offset);

    if (interleaved == 1)
    {
        const int gi = gz * psc(cstep) + gy * psc(embed_dim) + gx * 2;

        afp x0 = buffer_ld1(bottom_blob_data, gi);
        afp x1 = buffer_ld1(bottom_blob_data, gi + 1);

        buffer_st1(top_blob_data, gi, x0 * cosv - x1 * sinv);
        buffer_st1(top_blob_data, gi + 1, x0 * sinv + x1 * cosv);
    }
    else
    {
        const int gi = gz * psc(cstep) + gy * psc(embed_dim) + gx;

        afp x0 = buffer_ld1(bottom_blob_data, gi);
        afp x1 = buffer_ld1(bottom_blob_data, gi + halfdim);

        buffer_st1(top_blob_data, gi, x0 * cosv - x1 * sinv);
        buffer_st1(top_blob_data, gi + halfdim, x0 * sinv + x1 * cosv);
    }
}
)GLSL";

#endif // NCNN_VULKAN

// Raised once per net that actually had the layer put in front of the built-in. Atomic because
// two families load on two workers, and read by a check that a graph naming no rotary layer
// registers nothing at all.
std::atomic<int> overwrites{0};

} // namespace

RopeAt::RopeAt() {
    one_blob_only = false;
    support_inplace = false;
#if NCNN_VULKAN
    support_vulkan = true;
    // One shader rather than the built-in's pair, so the blobs arrive unpacked and the packed
    // road is the library's own conversion. Correctness first: a second shader is a second
    // place for the same indexing to go wrong.
    support_vulkan_packing = false;
#endif
}

int RopeAt::load_param(const ncnn::ParamDict &pd) {
    interleaved = pd.get(0, 0);
    return 0;
}

// The processor's arithmetic, which the built-in already had right: the tables are walked by
// their own rows, so a table of any width past half the embedding reads the same angles.
int RopeAt::forward(const std::vector<ncnn::Mat> &bottom_blobs, std::vector<ncnn::Mat> &top_blobs,
                    const ncnn::Option &opt) const {
    if (bottom_blobs.size() < 3) {
        return -1;
    }
    const ncnn::Mat &bottom_blob = bottom_blobs[0];
    const ncnn::Mat &cos_cache = bottom_blobs[1];
    const ncnn::Mat &sin_cache = bottom_blobs[2];

    const int embed_dim = bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int num_heads = bottom_blob.c;
    const int half = embed_dim / 2;

    // A table too small for what the rotation reads is read past its end, and neither the
    // built-in nor a shader says so: the answer is plausible numbers off the next allocation.
    // Refused here instead, on both paths, because an export whose tables do not fit its heads
    // is a mistake worth hearing about at the first pass rather than at the transcript.
    if (cos_cache.w < half || sin_cache.w < half || cos_cache.h < seqlen || sin_cache.h < seqlen) {
        return -1;
    }

    ncnn::Mat &top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++) {
        const ncnn::Mat head = bottom_blob.channel(q);
        ncnn::Mat out_head = top_blob.channel(q);

        for (int i = 0; i < seqlen; i++) {
            const float *cos_ptr = cos_cache.row(i);
            const float *sin_ptr = sin_cache.row(i);

            if (interleaved) {
                const float *ptr = head.row(i);
                float *outptr = out_head.row(i);
                for (int j = 0; j < half; j++) {
                    const float x0 = ptr[0];
                    const float x1 = ptr[1];
                    const float c = cos_ptr[j];
                    const float s = sin_ptr[j];
                    outptr[0] = x0 * c - x1 * s;
                    outptr[1] = x0 * s + x1 * c;
                    ptr += 2;
                    outptr += 2;
                }
            } else {
                const float *ptr0 = head.row(i);
                const float *ptr1 = ptr0 + half;
                float *outptr0 = out_head.row(i);
                float *outptr1 = outptr0 + half;
                for (int j = 0; j < half; j++) {
                    const float x0 = ptr0[j];
                    const float x1 = ptr1[j];
                    const float c = cos_ptr[j];
                    const float s = sin_ptr[j];
                    outptr0[j] = x0 * c - x1 * s;
                    outptr1[j] = x0 * s + x1 * c;
                }
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN

int RopeAt::create_pipeline(const ncnn::Option &opt) {
    // A layer that replaces a built-in is made for every net, the processor's included, and the
    // load calls this on all of them. ncnn's own Vulkan layer is only ever built for a Vulkan
    // net and so needs no such guard; without it here, `vkdev` is null and the pipeline below
    // faults while the weights are being read.
    if (vkdev == nullptr || !opt.use_vulkan_compute) {
        return 0;
    }

    const ncnn::Mat &shape = bottom_shapes.empty() ? ncnn::Mat() : bottom_shapes[0];

    // Every shape constant is left at zero on purpose, so `psc` reads all five from the push
    // constants the dispatch fills in from the blob it actually got. A hint is the shape the
    // parser recorded, and this layer refuses packing -- one conversion between the two and a
    // specialised `cstep` would address a layout that is no longer there. The tables are not
    // described by a hint at all.
    std::vector<ncnn::vk_specialization_type> specializations(1 + 5);
    specializations[0].i = interleaved;
    for (int i = 0; i < 5; i++) {
        specializations[1 + i].i = 0;
    }

    ncnn::Mat local_size_xyz;
    if (shape.dims == 3) {
        const int half = shape.w / 2;
        local_size_xyz.w = std::min(64, std::max(1, half));
        local_size_xyz.h = std::min(4, std::max(1, shape.h));
        local_size_xyz.c = std::min(4, std::max(1, shape.c));
    } else {
        local_size_xyz.w = 8;
        local_size_xyz.h = 8;
        local_size_xyz.c = 1;
    }

    // Compiled here rather than at build time: the library's own online compiler injects the
    // macro preamble a shader of this dialect needs, and it is the same one the built-ins go
    // through. The explicit length is deliberate -- the convenience overload that takes a bare
    // string passes `strlen - 1` and drops the last character.
    std::vector<uint32_t> spirv;
    const int length = (int)strlen(ROPE_AT_COMP);
    if (ncnn::compile_spirv_module(ROPE_AT_COMP, length, opt, spirv) != 0) {
        return -1;
    }

    pipeline = new ncnn::Pipeline(vkdev);
    pipeline->set_optimal_local_size_xyz(local_size_xyz);
    if (pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0) {
        delete pipeline;
        pipeline = nullptr;
        return -1;
    }

    return 0;
}

int RopeAt::destroy_pipeline(const ncnn::Option & /*opt*/) {
    delete pipeline;
    pipeline = nullptr;
    return 0;
}

int RopeAt::forward(const std::vector<ncnn::VkMat> &bottom_blobs,
                    std::vector<ncnn::VkMat> &top_blobs, ncnn::VkCompute &cmd,
                    const ncnn::Option &opt) const {
    if (bottom_blobs.size() < 3 || pipeline == nullptr) {
        return -1;
    }
    const ncnn::VkMat &bottom_blob = bottom_blobs[0];
    const ncnn::VkMat &cos_cache = bottom_blobs[1];
    const ncnn::VkMat &sin_cache = bottom_blobs[2];

    // The tables are indexed one element at a time, so they are read unpacked whatever the blob
    // beside them arrived as.
    ncnn::VkMat cos_flat = cos_cache;
    if (cos_cache.elempack != 1) {
        vkdev->convert_packing(cos_cache, cos_flat, 1, cmd, opt);
        if (cos_flat.empty()) {
            return -100;
        }
    }
    ncnn::VkMat sin_flat = sin_cache;
    if (sin_cache.elempack != 1) {
        vkdev->convert_packing(sin_cache, sin_flat, 1, cmd, opt);
        if (sin_flat.empty()) {
            return -100;
        }
    }

    // The same refusal as the processor's, for the same reason: a short table is read off the
    // end of its buffer by a shader that reports nothing.
    if (cos_flat.w < bottom_blob.w / 2 || sin_flat.w < bottom_blob.w / 2
            || cos_flat.h < bottom_blob.h || sin_flat.h < bottom_blob.h) {
        return -1;
    }

    ncnn::VkMat &top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_vkallocator);
    if (top_blob.empty()) {
        return -100;
    }

    std::vector<ncnn::VkMat> bindings(4);
    bindings[0] = bottom_blob;
    bindings[1] = cos_flat;
    bindings[2] = sin_flat;
    bindings[3] = top_blob;

    std::vector<ncnn::vk_constant_type> constants(5);
    constants[0].i = bottom_blob.w;
    constants[1].i = bottom_blob.h;
    constants[2].i = bottom_blob.c;
    constants[3].i = (int)bottom_blob.cstep;
    // The whole point: the table's own row width, not an assumption about it.
    constants[4].i = cos_flat.w;

    ncnn::VkMat dispatcher;
    dispatcher.w = bottom_blob.w / 2;
    dispatcher.h = bottom_blob.h;
    dispatcher.c = bottom_blob.c;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    return 0;
}

#endif // NCNN_VULKAN

namespace godot {
namespace rope_at {

ncnn::Layer *create(void * /*userdata*/) {
    return new RopeAt();
}

void destroy(ncnn::Layer *layer, void * /*userdata*/) {
    delete layer;
}

void install(ncnn::Net &net, const char *param_text) {
    if (param_text != nullptr && strstr(param_text, "RotaryEmbed") == nullptr) {
        return;
    }
    overwrites.fetch_add(1, std::memory_order_relaxed);
    net.register_custom_layer("RotaryEmbed", create, destroy);
}

void install_for_file(ncnn::Net &net, const String &param_path) {
    const PackedByteArray text = FileAccess::get_file_as_bytes(param_path);
    if (text.is_empty()) {
        // Unreadable here says nothing about the load that follows, which has its own reader and
        // its own sentence; registering is the safe way to be wrong.
        install(net);
        return;
    }
    const String named = String::utf8((const char *)text.ptr(), text.size());
    if (!named.contains("RotaryEmbed")) {
        return;
    }
    install(net);
}

int registrations() {
    return overwrites.load(std::memory_order_relaxed);
}

} // namespace rope_at
} // namespace godot
