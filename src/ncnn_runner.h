#ifndef NCNN_RUNNER_H
#define NCNN_RUNNER_H

#include "ncnn_device.h"

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>

#include <memory>
#include <atomic>
#include <thread>
#include <vector>

namespace ncnn {
class Net;
class Mat;
}

namespace godot {

// Godot-facing wrapper around a statically linked ncnn::Net: loads a converted
// *.ncnn.{param,bin} model (from file paths or in-memory buffers) and runs CPU
// inference — flat float vectors, image inputs, multi-input/multi-output (e.g.
// recurrent hidden state), plus an argmax convenience for discrete policies.
// Input/output blob names and an optional explicit input shape are configurable
// to match whatever the converter emitted.
class NcnnRunner : public Node {
    GDCLASS(NcnnRunner, Node)

protected:
    static void _bind_methods();

public:
    NcnnRunner();
    ~NcnnRunner() override;

    bool load_model(const String &p_param_path, const String &p_bin_path);
    bool load_model_from_buffers(const PackedByteArray &p_param, const PackedByteArray &p_bin);
    PackedFloat32Array run_inference(const PackedFloat32Array &p_input);
    PackedFloat32Array run_inference_image(const Ref<Image> &p_image, bool p_normalize_to_zero_one = true, bool p_grayscale = false);
    Dictionary run_inference_multi(const Array &p_inputs, const PackedStringArray &p_output_names);
    Array run_inference_batch(const Array &p_inputs, int p_num_threads = -1);
    int run_discrete_action(const PackedFloat32Array &p_input);
    bool is_model_loaded() const;

    // Non-blocking forward pass on a worker thread (#19): runs run_inference off the main
    // thread and emits `inference_completed(output)` on the main thread when done. Returns
    // true if the request was accepted (model loaded, input valid, no request in flight).
    // One request at a time — re-request after the signal (check is_inference_running()).
    bool run_inference_async(const PackedFloat32Array &p_input);
    bool is_inference_running() const;
    // Deliver a finished async result on the calling (main) thread if one is pending: emits
    // inference_completed and clears the in-flight flag. An in-tree runner drains automatically
    // every frame via _process; out-of-tree/headless callers (e.g. a SceneTree --script test) call
    // this each frame instead, since the worker's call_deferred completion is not reliably flushed
    // there on every platform (macOS mono — #222). Idempotent: a no-op when nothing is pending.
    void poll();
    void _process(double p_delta) override;

    // Which device the next load puts the graph on, "gpu" or "cpu". Set before load_model();
    // a machine with no device loads on the processor whatever this says, and device_used() is
    // what actually happened rather than what was asked for.
    void set_device(const String &p_word);
    String get_device() const;
    String device_used() const;

    // The driver's own name for the card the graph is on, or "" from a graph on the processor.
    // Two cards in one machine are two different answers, so a row that names one asks here.
    String device_name() const;

    // Why the card was asked for and the processor got the graph, as one sentence, or "" where
    // nothing went wrong: the request was met, or the processor was what was asked for.
    String device_problem() const;

    // What the card has free and in total, for a host's own memory line. Empty from a graph on
    // the processor and from a machine with no device.
    Dictionary device_memory() const;

    // Where the compiled shaders are kept between runs, as a path the C library can open. It is
    // the host's own location and it is process-wide: every class in this library shares one
    // cache, so whichever loads first names the file for all of them.
    void set_shader_cache(const String &p_path);
    String shader_cache() const;

    void set_input_blob_name(const String &p_name);
    String get_input_blob_name() const;
    void set_output_blob_name(const String &p_name);
    String get_output_blob_name() const;
    void set_input_shape(const PackedInt32Array &p_shape);
    PackedInt32Array get_input_shape() const;
    void clear_input_shape();

private:
    bool create_input_mat_from_array(const PackedFloat32Array &p_input, ncnn::Mat &r_input) const;
    bool build_mat_from_shape(const PackedFloat32Array &p_data, const PackedInt32Array &p_shape, ncnn::Mat &r_mat) const;
    bool run_inference_internal(const ncnn::Mat &p_input, ncnn::Mat &r_output) const;
    static PackedFloat32Array output_mat_to_packed_float_array(const ncnn::Mat &p_output);
    // Main-thread delivery of a finished async result: if result_ready_ is set, join the worker,
    // clear the in-flight flag, and emit inference_completed exactly once (the atomic exchange makes
    // concurrent drain sites — _process, poll(), and the worker's best-effort call_deferred — race-
    // free and single-delivery). Driven by _process (in-tree) and poll() (out-of-tree). Not bound to
    // GDScript by name, so a script can't fake a completion.
    void deliver_if_ready();
    // The device and the shared shader cache written onto a fresh net, before its structure is
    // parsed: the loader reads them there and turns Vulkan off itself where there is no device,
    // so this must run before load_param or the graph lands on the processor whatever was asked.
    void apply_device();
    // This runner's entry in the library's count of graphs on the card, raised once a load has
    // landed there and lowered before the net goes. Without it a runner freed with a graph still
    // on the card would hold the card open against the shutdown that frees the shared cache.
    void count_it_in();
    void count_it_out();
    // True if it's safe to replace net_ now: refuses while an async inference is in flight
    // (logs via p_where) and joins a finished worker. Call before swapping net_ in load_*.
    bool ready_to_swap_net(const char *p_where);

    // Private copy of the .bin bytes for buffer-loaded models: ncnn's ModelBin prefers
    // DataReaderFromMemory::reference() (zero-copy), so the loaded Net's weight Mats ALIAS the
    // memory they were read from. Aliasing the caller's PackedByteArray is a use-after-free once
    // that temporary dies; aliasing THIS runner-owned copy is safe. Declared BEFORE net_ on
    // purpose: members destroy in reverse declaration order, so net_ (and its aliasing Mats) is
    // torn down first in ~NcnnRunner. Cleared on path-based loads (those copy internally).
    // (Composition, not a DataReader subclass: iOS links ncnn without RTTI, so a subclass can't
    // resolve `typeinfo for ncnn::DataReader`.)
    std::vector<unsigned char> bin_copy_;
    std::unique_ptr<ncnn::Net> net_;
    // Atomic like the device beside it: a load writes it on whichever thread called, and
    // device_memory() and is_model_loaded() read it while a host draws a row on the main one.
    std::atomic<bool> model_loaded_{false};
    // Which device was asked for and which the graph got, the same pair every class in this
    // library carries. Read at the load and written back from the net's own options after it.
    ncnn_device::Choice device_;
    // Whether this runner's graph is counted as one of the library's nets on the card.
    bool counted_ = false;
    String input_blob_name_ = "input";
    String output_blob_name_ = "output";
    PackedInt32Array input_shape_;
    std::thread async_worker_;
    std::atomic<bool> inference_running_{false};
    // Published by the worker (release) and consumed once on the main thread (deliver_if_ready's
    // exchange-acquire). pending_output_ is written by the worker before the flag is set and read by
    // the main thread only after it observes the flag, so the atomic establishes the happens-before.
    std::atomic<bool> result_ready_{false};
    PackedFloat32Array pending_output_;
};

} // namespace godot

#endif // NCNN_RUNNER_H
