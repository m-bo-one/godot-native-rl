#include "ncnn_device.h"

#include <godot_cpp/variant/variant.hpp>

#include <platform.h>

#if NCNN_VULKAN
#include <gpu.h>
#include <pipelinecache.h>
#endif

#include <mutex>
#include <string>

using namespace godot;

namespace {

// One lock over everything below. The library's own instance and device creation are locked
// already; what is not is this file's three answers, and two loads on two threads reach them at
// once. It is held for a handful of statements and never around a load.
std::mutex device_lock;

// Whether the card has been looked for yet, and what was found. Looking costs a Vulkan instance,
// so it happens once: a machine with no driver would otherwise pay a failed LoadLibrary per graph.
//
// The three answers are held as plain strings and turned into the engine's own at the boundary.
// A godot::String at file scope is constructed before the library's entry point runs, and its
// constructor reaches through a function table that does not exist yet -- which Windows reports
// as a failed initialisation routine and the engine as an extension it cannot open.
bool has_looked = false;
bool has_device = false;
std::string no_device_said;
std::string device_named;

#if NCNN_VULKAN
// The cache every net points at, and the file it is kept in between runs. It is never freed: a
// net holds a bare pointer to it, and freeing it while any graph is loaded is a dangling read.
ncnn::PipelineCache *cache = nullptr;
#endif
std::string cache_file;

const char *NO_VULKAN_BUILD = "Govorilka: this build of the runner carries no Vulkan backend, so "
                              "every graph runs on the processor.";
const char *NO_DRIVER = "Govorilka: no Vulkan driver was found on this machine, so every graph "
                        "runs on the processor. Install the graphics driver for this card, or "
                        "leave the device setting on the processor.";
const char *NO_USABLE_DEVICE = "Govorilka: a Vulkan driver is installed and it offers no device "
                               "this library can use, so every graph runs on the processor.";

// The card looked for once, under the lock. It creates the library's instance and its one device,
// both of which the library then caches; every net afterwards lands on that same device.
void look_for_a_device() {
    if (has_looked) {
        return;
    }
    has_looked = true;
#if NCNN_VULKAN
    if (ncnn::get_gpu_count() <= 0) {
        no_device_said = NO_DRIVER;
        return;
    }
    ncnn::VulkanDevice *device = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    if (device == nullptr || !device->is_valid()) {
        no_device_said = NO_USABLE_DEVICE;
        return;
    }
    has_device = true;
    device_named = device->info.device_name();
#else
    no_device_said = NO_VULKAN_BUILD;
#endif
}

} // namespace

const char *ncnn_device::CPU_WORD = "cpu";
const char *ncnn_device::GPU_WORD = "gpu";

bool ncnn_device::is_built_with_vulkan() {
#if NCNN_VULKAN
    return true;
#else
    return false;
#endif
}

bool ncnn_device::is_available() {
    std::lock_guard<std::mutex> held(device_lock);
    look_for_a_device();
    return has_device;
}

String ncnn_device::unavailable_reason() {
    std::lock_guard<std::mutex> held(device_lock);
    look_for_a_device();
    return has_device ? String() : String(no_device_said.c_str());
}

String ncnn_device::name() {
    std::lock_guard<std::mutex> held(device_lock);
    look_for_a_device();
    return String(device_named.c_str());
}

String ncnn_device::word_for(bool on_gpu) {
    if (!on_gpu) {
        return String(CPU_WORD);
    }
    const String named = name();
    if (named.is_empty()) {
        return String(GPU_WORD);
    }
    return String(GPU_WORD) + String(" ") + named;
}

Dictionary ncnn_device::memory() {
    Dictionary answer;
#if NCNN_VULKAN
    {
        std::lock_guard<std::mutex> held(device_lock);
        look_for_a_device();
        if (!has_device) {
            return answer;
        }
    }
    const int index = ncnn::get_default_gpu_index();
    ncnn::VulkanDevice *device = ncnn::get_gpu_device(index);
    if (device == nullptr) {
        return answer;
    }
    // The budget the driver reports is what an allocation is actually held to, and it is in
    // megabytes. The total is the largest heap the card keeps to itself: the library picks its
    // buffers out of one of those, and a host heap counted here would read as video memory.
    const int64_t megabyte = 1024 * 1024;
    answer["free_bytes"] = (int64_t)device->get_heap_budget() * megabyte;
    const VkPhysicalDeviceMemoryProperties &properties =
            ncnn::get_gpu_info(index).physicalDeviceMemoryProperties();
    int64_t total = 0;
    for (uint32_t heap = 0; heap < properties.memoryHeapCount; heap++) {
        if ((properties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) {
            continue;
        }
        const int64_t size = (int64_t)properties.memoryHeaps[heap].size;
        if (size > total) {
            total = size;
        }
    }
    answer["total_bytes"] = total;
#endif
    return answer;
}

#if NCNN_VULKAN
ncnn::PipelineCache *ncnn_device::shared_cache() {
    std::lock_guard<std::mutex> held(device_lock);
    look_for_a_device();
    if (!has_device) {
        return nullptr;
    }
    if (cache == nullptr) {
        cache = new ncnn::PipelineCache(ncnn::get_gpu_device(ncnn::get_default_gpu_index()));
        if (!cache_file.empty()) {
            // A file that will not read is a miss and never a refusal: the cache is an
            // optimisation, and every shader it does not carry is compiled as it always was.
            cache->load_cache(cache_file.c_str());
        }
    }
    return cache;
}
#else
ncnn::PipelineCache *ncnn_device::shared_cache() {
    return nullptr;
}
#endif

void ncnn_device::set_cache_path(const String &path) {
#if NCNN_VULKAN
    std::lock_guard<std::mutex> held(device_lock);
    const std::string wanted = path.utf8().get_data();
    if (cache_file == wanted) {
        return;
    }
    cache_file = wanted;
    // A cache already built reads the new file at once rather than at the next load: the caller
    // names the file before the first graph, and one named after it would otherwise never be read.
    if (cache != nullptr && !cache_file.empty()) {
        cache->load_cache(cache_file.c_str());
    }
#else
    (void)path;
#endif
}

String ncnn_device::cache_path() {
    std::lock_guard<std::mutex> held(device_lock);
    return String(cache_file.c_str());
}

void ncnn_device::Choice::ask_for(const String &word) {
    wants_gpu = word.strip_edges().to_lower().begins_with(GPU_WORD);
}

String ncnn_device::Choice::asked() const {
    return String(wants_gpu ? GPU_WORD : CPU_WORD);
}

void ncnn_device::Choice::landed_on(bool p_on_gpu) {
    on_gpu = p_on_gpu;
}

String ncnn_device::Choice::landed() const {
    return word_for(on_gpu);
}


bool ncnn_device::save_cache() {
#if NCNN_VULKAN
    std::lock_guard<std::mutex> held(device_lock);
    if (cache == nullptr || cache_file.empty()) {
        return false;
    }
    return cache->save_cache(cache_file.c_str()) == 0;
#else
    return false;
#endif
}
