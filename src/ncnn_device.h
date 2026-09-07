#ifndef NCNN_DEVICE_H
#define NCNN_DEVICE_H

#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

#include <atomic>

namespace ncnn {
class PipelineCache;
}

namespace godot {

// The one place in this library that knows whether there is a card and what runs on it. Every
// graph asks here rather than touching the library's globals: the instance, the device and the
// compiled shaders are one per process, and a second copy of any of them is a second driver
// context on the same card.
//
// Nothing here creates anything until something asks for the card. A machine with no driver
// answers "no" to is_available() and pays a failed LoadLibrary for it, once.
namespace ncnn_device {

// The two words a device goes by. They are the words the addon writes on a footprint row, so
// they are spelled the same on both sides of the boundary.
extern const char *CPU_WORD;
extern const char *GPU_WORD;

// Whether this library carries the Vulkan backend at all. False is a build with the flag off,
// where every graph runs on the processor whatever it was asked for.
bool is_built_with_vulkan();

// Whether a graph asked for the card would get one: the loader found a driver, the library made
// an instance, and the device it picked is usable. Asked once and remembered -- probing costs a
// Vulkan instance, and a machine without one must not pay for it on every load.
bool is_available();

// Why there is no card, as one sentence a person can act on, or "" where there is one. It says
// which of the three it was: the build, the driver, or the device.
String unavailable_reason();

// The driver's own name for the device a graph lands on, or "" where there is none. It is the
// raw name and never a word with the name behind it: the two halves of a row are joined on the
// other side of the boundary, where the addon's own rule for that already lives.
String name();

// What the card has free and in total, in bytes, or an empty dictionary where no driver reports a
// budget. The pair comes from VK_EXT_memory_budget and from nowhere else -- without the extension
// the library's own answer is a fixed fraction of the heap, which is not a reading of anything.
Dictionary memory();

// How many nets are loaded on the card right now. A graph raises this as it lands there and lowers
// it as it is cleared, and shut_down() refuses while any is up: the cache holds the pipelines those
// nets run through, so freeing it under a loaded net leaves its layers pointing into freed memory.
void net_opened();
void net_closed();

// The pipeline cache every net shares, or null where there is no device. Without it each net
// builds its own and compiles every shader again, which is seconds per graph rather than
// milliseconds. Never freed: it outlives the nets that point at it, by design.
ncnn::PipelineCache *shared_cache();

// Where the compiled shaders are kept between runs, as a path the C library can open -- the
// caller globalises it. Set before the first load; an empty path keeps the cache in memory for
// the run alone. Setting it reads the file at once, and a file that will not read is a miss.
void set_cache_path(const String &path);
String cache_path();

// The cache written back to that path, true when the file was written. A load compiles what the
// file did not carry, so this is called after one; it costs nothing when the path is empty.
bool save_cache();

// The latch and everything behind it put back as it was before anything looked for a card. It is
// called from the extension's own initialiser: the latch belongs to this library rather than to
// the process, and a host that takes the extension down and brings it up again would otherwise be
// answered "the card was given back" for the rest of the run.
void wake_up();

// The compiled shaders and the device given back, in that order, while the library is still
// loaded. It is called from the extension's own terminator: left to the library's static
// destructors the device is torn down with the cache's pipelines still alive on it, and the
// process stops answering on the way out instead of exiting.
//
// Once it has run nothing looks for a card again -- a question asked during teardown would build
// a fresh instance on the way out -- so every answer here is the answer of a machine with none.
void shut_down();

// The device a family asked for and the one its graphs got, which is the whole of what a class on
// these graphs has to carry. Every family holds one of these and binds the same four methods over
// it, so a host reads the same answer from the recogniser, the picture model and the runner.
//
// The two are kept apart on purpose: a machine with no driver runs the graphs on the processor and
// a row that reported the request would say the card while nothing was on it.
// Both halves are atomic. The request is written on whichever thread set the property and read on
// the pool thread the load runs on; the answer is written there and read by a host on the main
// thread as it draws a row, and a torn bool would be a row saying the wrong device.
struct Choice {
    // What the host asked for. Anything that is not the card's word is the processor: a row is
    // one of two, and a word nobody recognises must not become a third answer.
    void ask_for(const String &word);

    // The word back, for the property a host reads. It is what was asked, not what happened.
    String asked() const;

    // What the graphs are handed. True only when the card was asked for; the loader still turns
    // it off per net where there is no device.
    bool wants_the_card() const;

    // Written by the family once its graphs are loaded, from what each graph reports it got.
    void landed_on(bool on_gpu);

    // Which device the graphs are on, as the bare word and as the question behind it. Before a
    // load it is the processor, which is true -- nothing is loaded anywhere.
    String landed_word() const;
    bool is_on_the_card() const;

private:
    std::atomic<bool> wants_gpu{false};
    std::atomic<bool> on_gpu{false};
};

} // namespace ncnn_device

} // namespace godot

#endif // NCNN_DEVICE_H
