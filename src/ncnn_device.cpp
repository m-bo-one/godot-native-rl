#include "ncnn_device.h"

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <platform.h>

#if NCNN_VULKAN
#include <gpu.h>
#include <pipelinecache.h>
#endif

#include <filesystem>
#include <mutex>
#include <string>
#include <system_error>

using namespace godot;

namespace {

// One lock over everything below. The library's own instance and device creation are locked
// already; what is not is this file's three answers, and two loads on two threads reach them at
// once. The one call it is held across for longer than a handful of statements is the first look
// for a card, which is seconds on a cold driver -- so memory() asks for it and gives up.
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

// The one lock over reading and writing that file, held for the megabytes and never for anything
// else. The device's lock is never held while this one is taken, so the two cannot deadlock.
std::mutex file_lock;

// How many bytes of shaders that file already carries, as this process last read or wrote it. A
// load that compiled nothing new leaves the count where it is, and nothing is written.
size_t cache_bytes = 0;

const char *NO_VULKAN_BUILD = "Govorilka: this build of the runner carries no Vulkan backend, so "
                              "every graph runs on the processor.";
const char *NO_DRIVER = "Govorilka: no Vulkan driver was found on this machine, so every graph "
                        "runs on the processor. Install the graphics driver for this card, or "
                        "leave the device setting on the processor.";
const char *NO_USABLE_DEVICE = "Govorilka: a Vulkan driver is installed and it offers no device "
                               "this library can use, so every graph runs on the processor.";
const char *STILL_HOLDING = "Govorilka: the card was given back with {0} graph(s) still loaded on "
                            "it, whose layers now point into freed pipelines. Give every model "
                            "back before the library is taken down; this is the last moment "
                            "anything can free the card, so it is freed anyway.";
const char *GIVEN_BACK = "Govorilka: the card was given back as this library was taken down, so "
                         "every graph from here on runs on the processor. Nothing asks for it "
                         "again inside a process that has shut it.";

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

#if NCNN_VULKAN
// The file read into a cache, with the device's own lock given back. A file that will not read is
// a miss and never a refusal: every shader it does not carry is compiled as it always was, and the
// bytes it did carry are what says whether a later save has anything new to write.
void read_the_cache(ncnn::PipelineCache *into, const std::string &named) {
    std::lock_guard<std::mutex> reading(file_lock);
    if (into->load_cache(named.c_str()) != 0) {
        cache_bytes = 0;
        return;
    }
    std::error_code unread;
    const uintmax_t weighs = std::filesystem::file_size(std::filesystem::u8path(named), unread);
    cache_bytes = unread ? 0 : (size_t)weighs;
}
#endif

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
    // Asked for rather than waited on. The lock is held across the query -- the instance it reads
    // through is what shut_down() destroys -- but a host draws this on the main thread while a
    // load may be inside the first look for a card, and a frame may not wait seconds for a driver.
    std::unique_lock<std::mutex> held(device_lock, std::try_to_lock);
    if (!held.owns_lock()) {
        return answer;
    }
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
    // The file lock first and the device's second, which is the order shut_down() takes them in
    // and the only order that cannot deadlock. Holding this one for the whole call is also what
    // keeps the cache alive across the read: shut_down() cannot free it without this lock.
    std::lock_guard<std::mutex> reading(file_lock);
    ncnn::PipelineCache *made = nullptr;
    std::string wanted;
    bool is_fresh = false;
    {
        std::lock_guard<std::mutex> held(device_lock);
        look_for_a_device();
        if (!has_device) {
            return nullptr;
        }
        if (cache == nullptr) {
            cache = new ncnn::PipelineCache(ncnn::get_gpu_device(ncnn::get_default_gpu_index()));
            is_fresh = !cache_file.empty();
            wanted = cache_file;
        }
        made = cache;
    }
    // The file is read with the device's own lock given back: it is megabytes off a disk, and a
    // host drawing a memory row on the main thread asks that lock for the card's free bytes.
    if (is_fresh) {
        read_the_cache(made, wanted);
    }
    return made;
}
#else
ncnn::PipelineCache *ncnn_device::shared_cache() {
    return nullptr;
}
#endif

void ncnn_device::set_cache_path(const String &path) {
#if NCNN_VULKAN
    // The file lock first, as everywhere: it is what shut_down() has to take before it can free
    // the cache, so a pointer read under the device's lock below stays alive while this holds it.
    std::lock_guard<std::mutex> reading(file_lock);
    ncnn::PipelineCache *made = nullptr;
    std::string wanted;
    {
        std::lock_guard<std::mutex> held(device_lock);
        wanted = path.utf8().get_data();
        if (cache_file == wanted) {
            return;
        }
        cache_file = wanted;
        made = cache;
    }
    // A cache already built reads the new file at once rather than at the next load: the caller
    // names the file before the first graph, and one named after it would otherwise never be read.
    if (made != nullptr && !wanted.empty()) {
        read_the_cache(made, wanted);
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

// Written only where this process holds more shaders than the file carried when it last looked.
// The library writes it atomically itself -- a sibling of its own, then a replacing move -- so
// nothing here touches the file; what is saved is the megabytes of writing that a load which
// compiled nothing new would otherwise repeat.
bool ncnn_device::save_cache() {
#if NCNN_VULKAN
    // The file lock first, as everywhere: shut_down() takes it before the device's and cannot
    // free the cache without it, which is what makes the pointer read below safe to use here.
    std::lock_guard<std::mutex> writing(file_lock);
    ncnn::PipelineCache *kept = nullptr;
    std::string wanted;
    {
        std::lock_guard<std::mutex> held(device_lock);
        kept = cache;
        wanted = cache_file;
    }
    if (kept == nullptr || wanted.empty()) {
        return false;
    }
    // The device's own lock is given back before any of this: serialising is megabytes and the
    // write is a disk, and a host drawing a memory row on the main thread waits on that lock.
    std::lock_guard<std::mutex> writing(file_lock);
    std::vector<unsigned char> held_now;
    if (kept->save_cache(held_now) != 0) {
        return false;
    }
    if (held_now.size() <= cache_bytes) {
        return false;
    }
    if (kept->save_cache(wanted.c_str()) != 0) {
        return false;
    }
    cache_bytes = held_now.size();
    return true;
#else
    return false;
#endif
}

void ncnn_device::wake_up() {
#if NCNN_VULKAN
    std::lock_guard<std::mutex> held(device_lock);
    // The latch and the three answers behind it, and nothing else. The count of graphs on the
    // card is a count of objects somebody still holds: zeroing it here would lose them.
    has_shut = false;
    has_looked = false;
    has_device = false;
    no_device_said.clear();
    device_named.clear();
#endif
}

void ncnn_device::shut_down() {
#if NCNN_VULKAN
    // Taken before the device's, and in that order everywhere: a save in flight holds this one
    // and the cache it is serialising may not be freed under it.
    std::lock_guard<std::mutex> writing(file_lock);
    std::lock_guard<std::mutex> held(device_lock);
    if (has_shut) {
        return;
    }
    // Said and not obeyed. This is the last hook there is -- the library's own static destructors
    // run next and do this same teardown in the order that hangs the process -- so a graph still
    // loaded is a warning to whoever wrote the host, never a reason to leave the card standing.
    if (nets_on_the_card > 0) {
        UtilityFunctions::push_error(
                String(STILL_HOLDING).format(Array::make(nets_on_the_card)));
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
    nets_on_the_card = 0;
    cache_bytes = 0;
    device_named.clear();
    // The reason goes with the device. "There is no card" with nothing after it is a sentence a
    // host would print blank, and after this there really is none.
    no_device_said = GIVEN_BACK;
#endif
}
