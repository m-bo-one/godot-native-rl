#ifndef NCNN_GRAPH_H
#define NCNN_GRAPH_H

#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <datareader.h>
#include <mat.h>
#include <net.h>

// ncnn's platform header pulls in windows.h, which brings a CONNECT_DEFERRED of its own from
// the network API -- and Object::ConnectFlags has a member of that name, so Godot's generated
// header stops parsing halfway through the class in any file that reached ncnn first. Dropped
// here rather than worked around by include order, which only holds until somebody reorders.
// Nothing in this library calls a WNet function, and the enum this frees is Godot's.
#ifdef CONNECT_DEFERRED
#undef CONNECT_DEFERRED
#endif

namespace godot {

// One graph of an export together with the two buffers it was read out of. ncnn aliases the
// weight bytes rather than copying them, so dropping the .bin buffer would leave every layer
// of the net pointing into freed memory on the next extract.
//
// prepare() and read() are load() split in two: a family whose graph carries a layer ncnn does
// not know registers it between them, because register_custom_layer only counts before the
// structure is parsed.
//
// A load that runs out of memory still ends the process and nothing here can stop it: the
// runtime answers a failed allocation with an empty blob rather than an error, the weight repack
// writes through it, and a handler around the call was measured being entered and never reached.
// What is answered here instead of faulted is the load going wrong for a reason of its own: a
// structure asking for more weights than the file holds, which is the reader below.
struct NcnnGraph {
    // What one graph is opened as, beside the thread count. Both halves are per graph and never
    // per model, and they are named rather than handed over as two adjacent bools: a call site
    // reading `true, false` says nothing about which of them is the precision.
    //
    // Half-precision blob storage is per graph because a residual stream that grows past the
    // format's ceiling -- a text encoder's does -- has to be kept single while the graphs beside
    // it stay half. The device is per graph because the graphs of one model do not all win on the
    // card, and one carrying a layer the backend has no shader for pays a round trip per layer.
    struct Options {
        enum Precision { HALF, SINGLE };
        enum Device { PROCESSOR, CARD };

        Precision precision = HALF;
        Device device = PROCESSOR;

        Options() = default;
        Options(Precision p_precision, Device p_device) :
                precision(p_precision), device(p_device) {}

        // A family's own word for the device turned into this one, in the single place that
        // knows both spellings. A family holds a request as a flag and a graph as an enum.
        static Device wanted(bool wants_the_card);
    };

    ncnn::Net net;
    PackedByteArray param;
    PackedByteArray weights;

    // A graph that asks for the card and finds none loads onto the processor rather than
    // refusing -- runs_on_gpu() is what it got, not what it asked for.
    void prepare(int num_threads, const Options &how = Options());
    bool read(const String &param_path, const String &bin_path);
    bool load(const String &param_path, const String &bin_path, int num_threads);

    // The structure read again over the weights this graph already holds. One weight file
    // behind several structures is what a network exported for two picture sizes is, and
    // reading the file a second time would cost its size in memory for nothing.
    bool reread(const String &param_path, int num_threads, const Options &how);

    // Which device this graph actually loaded onto. It is read off the net's own options, which
    // the loader turns off itself where there is no usable device, so it is the answer and not
    // the request. Meaningful only once read() or reread() has answered true.
    bool runs_on_gpu() const;

    void clear();

    // The count of graphs on the card is kept for the whole library, and this is what lowers
    // this graph's entry in it. Without it a graph dropped without a clear() would hold the
    // card open against the shutdown that frees the shared cache.
    ~NcnnGraph();

private:
    // Whether this graph is counted as one of the library's nets on the card. It follows the
    // net's own options rather than the request: a graph asked for a card there was none of is
    // on the processor and holds nothing.
    bool counted = false;

    void count_it_in();
    void count_it_out();
};

// The small things every family on ncnn does the same way, kept in one place so the traps in
// them are fixed once. Reached through the recogniser's and the synthesiser's own statics.
namespace ncnn_util {

// A Mat ncnn owns, filled from somebody else's memory. Every input has to go through this:
// a Mat wrapping a foreign pointer carries a null refcount, and the first in-place layer to
// consume it dereferences that null and takes the process down.
ncnn::Mat owned(const float *source, int w, int h);

// The same for a single whole number, which is what an embedding graph indexes with. The
// lookup reads the blob's four bytes as an int rather than converting them, so a float index
// is a bit pattern past the last row -- and out of range is clamped, never reported.
ncnn::Mat owned_index(int value);

// The same again for a run of whole numbers, which is what a symbol sequence is.
ncnn::Mat owned_indices(const int *source, int count);

// The one file of a folder carrying a fragment and a suffix. Fragments rather than whole
// names: an export ships them under whatever prefix it was written with, and a folder taken
// from anywhere has to work without somebody renaming its files first.
String pick(const PackedStringArray &files, const String &mark, const String &suffix);

double now_ms();

} // namespace ncnn_util

} // namespace godot

#endif // NCNN_GRAPH_H
