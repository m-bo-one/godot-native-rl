#include "ncnn_device.h"

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <platform.h>

#if NCNN_VULKAN
#include <gpu.h>
#include <pipelinecache.h>
#endif

#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <system_error>

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

// Whether the card has been given back. After that nothing looks for one again: a question asked
// during teardown -- a row drawn as the tree comes apart -- would build a fresh instance on the
// way out of the process and leave it standing with nothing left to destroy it.
bool has_shut = false;

// How many nets are loaded on a card. shut_down() refuses while any is up: the cache below holds
// the pipelines they run through, and freeing it under them is a dangling read on the next extract.
int nets_on_the_card = 0;

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
const char *STILL_HOLDING = "Govorilka: the card was asked for back with {0} graph(s) still "
                            "loaded on it, and it was kept. Give every model back before the "
                            "library is taken down, or the layers of those graphs are left "
                            "pointing into freed pipelines.";

// The card looked for once, under the lock. It creates the library's instance and its one device,
// both of which the library then caches; every net afterwards lands on that same device.
void look_for_a_device() {
    if (has_looked || has_shut) {
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

// The pair the driver itself keeps, and nothing derived from it. Both numbers are read out of one
// VK_EXT_memory_budget query over the same heap, so what a caller subtracts is what the driver
// says this process may have against what it has taken -- rather than a budget put beside a raw
// heap size, which is two different things and reads as gigabytes used by nobody.
//
// Without the extension there is no reading to report: the library's own answer is then a flat
// 70 or 50 per cent of the heap, which never moves and says nothing about what is on the card.
Dictionary ncnn_device::memory() {
    Dictionary answer;
#if NCNN_VULKAN
    // Held across the query and not only across the look: the instance the query reads through is
    // what shut_down() destroys, and a reading taken while that runs is a read of freed memory.
    std::lock_guard<std::mutex> held(device_lock);
    look_for_a_device();
    if (!has_device) {
        return answer;
    }
    const int index = ncnn::get_default_gpu_index();
    const ncnn::GpuInfo &info = ncnn::get_gpu_info(index);
    const bool can_be_read = info.support_VK_EXT_memory_budget()
            && ncnn::vkGetPhysicalDeviceMemoryProperties2KHR != nullptr;
    if (!can_be_read) {
        return answer;
    }

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgets;
    budgets.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    budgets.pNext = nullptr;
    VkPhysicalDeviceMemoryProperties2KHR properties;
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR;
    properties.pNext = &budgets;
    ncnn::vkGetPhysicalDeviceMemoryProperties2KHR(info.physicalDevice(), &properties);

    // The largest heap the card keeps to itself, which is the one a graph's weights go into. A
    // host heap counted here would read as video memory a game never had.
    const VkPhysicalDeviceMemoryProperties &heaps = properties.memoryProperties;
    int64_t budget = 0;
    int64_t used = 0;
    for (uint32_t heap = 0; heap < heaps.memoryHeapCount; heap++) {
        if ((heaps.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) {
            continue;
        }
        if ((int64_t)budgets.heapBudget[heap] <= budget) {
            continue;
        }
        budget = (int64_t)budgets.heapBudget[heap];
        used = (int64_t)budgets.heapUsage[heap];
    }
    if (budget <= 0) {
        return answer;
    }
    answer["total_bytes"] = budget;
    answer["free_bytes"] = budget > used ? budget - used : (int64_t)0;
#endif
    return answer;
}

void ncnn_device::net_opened() {
    std::lock_guard<std::mutex> held(device_lock);
    nets_on_the_card++;
}

void ncnn_device::net_closed() {
    std::lock_guard<std::mutex> held(device_lock);
    if (nets_on_the_card > 0) {
        nets_on_the_card--;
    }
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
    wants_gpu.store(word.strip_edges().to_lower().begins_with(GPU_WORD));
}

String ncnn_device::Choice::asked() const {
    return String(wants_gpu.load() ? GPU_WORD : CPU_WORD);
}

bool ncnn_device::Choice::wants_the_card() const {
    return wants_gpu.load();
}

void ncnn_device::Choice::landed_on(bool p_on_gpu) {
    on_gpu.store(p_on_gpu);
}

String ncnn_device::Choice::landed_word() const {
    return String(on_gpu.load() ? GPU_WORD : CPU_WORD);
}

bool ncnn_device::Choice::is_on_the_card() const {
    return on_gpu.load();
}

// Written beside the file and moved over it in one step. save_cache() truncates what it is handed
// and then writes, so a process that stopped part-way through would leave the next run a file that
// reads as a miss for every shader in it; the sibling carries this process's own number, because
// two games saving at once under one name would each remove the other's half-written file.
bool ncnn_device::save_cache() {
#if NCNN_VULKAN
    std::lock_guard<std::mutex> held(device_lock);
    if (cache == nullptr || cache_file.empty()) {
        return false;
    }
    const std::string beside = cache_file + ".writing."
            + std::to_string(OS::get_singleton()->get_process_id());
    if (cache->save_cache(beside.c_str()) != 0) {
        std::remove(beside.c_str());
        return false;
    }
    // One move that replaces what is there, rather than a remove and a rename: between those two
    // a reader finds no file at all, and on Windows a rename onto an existing name simply fails.
    std::error_code failed;
    std::filesystem::rename(std::filesystem::u8path(beside),
            std::filesystem::u8path(cache_file), failed);
    if (failed) {
        std::remove(beside.c_str());
        return false;
    }
    return true;
#else
    return false;
#endif
}

void ncnn_device::shut_down() {
#if NCNN_VULKAN
    std::lock_guard<std::mutex> held(device_lock);
    if (has_shut) {
        return;
    }
    if (nets_on_the_card > 0) {
        UtilityFunctions::push_error(
                String(STILL_HOLDING).format(Array::make(nets_on_the_card)));
        return;
    }
    // The cache first: it holds shader modules and pipelines made on the device below, and a
    // device destroyed under them is what leaves the process hanging on the way out.
    if (cache != nullptr) {
        delete cache;
        cache = nullptr;
    }
    // On has_looked and not on has_device: the instance exists the moment get_gpu_count() ran,
    // whether or not a usable device came out of it, and one left standing is one nothing destroys.
    if (has_looked) {
        ncnn::destroy_gpu_instance();
    }
    // Latched rather than reset. Anything asking after this -- a row drawn as the tree comes
    // apart -- would otherwise look again and build a fresh instance on the way out.
    has_shut = true;
    has_looked = false;
    has_device = false;
    device_named.clear();
#endif
}
