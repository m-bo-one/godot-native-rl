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
struct NcnnGraph {
    ncnn::Net net;
    PackedByteArray param;
    PackedByteArray weights;

    // Half-precision blob storage is a flag per graph rather than one over a whole model: a
    // residual stream that grows past the format's ceiling -- a text encoder's does -- has to
    // be kept single while the graphs beside it stay half, and one switch cannot say that.
    void prepare(int num_threads, bool fp16_storage = true);
    bool read(const String &param_path, const String &bin_path);
    bool load(const String &param_path, const String &bin_path, int num_threads);

    // The structure read again over the weights this graph already holds. One weight file
    // behind several structures is what a network exported for two picture sizes is, and
    // reading the file a second time would cost its size in memory for nothing.
    bool reread(const String &param_path, int num_threads, bool fp16_storage = true);

    void clear();
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
