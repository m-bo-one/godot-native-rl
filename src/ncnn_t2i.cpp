#include "ncnn_t2i.h"

#include "ncnn_report.h"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/reg_ex_match.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/callable_method_pointer.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <cpu.h>

#include <algorithm>
#include <cmath>
#include <cstring>

// The two fences below are a try around a run and a catch around a thread that could not
// start; compiled without exceptions they are dead code and either fault unwinds into the
// engine. Refused here rather than discovered on a machine that could not start the thread.
#if !defined(_CPPUNWIND) && !defined(__EXCEPTIONS) && !defined(__cpp_exceptions)
#error "ncnn_t2i.cpp needs C++ exceptions: build with disable_exceptions=no"
#endif

using namespace godot;

namespace {

// Takes the busy flag in one atomic step or reports that another path holds it, and gives it
// back unless the turn was handed on. Reading the flag and raising it separately lets two
// callers both start a worker, and assigning a thread over a joinable one is std::terminate().
struct BusyGuard {
    std::atomic<bool> *held = nullptr;

    explicit BusyGuard(std::atomic<bool> &flag) {
        bool expected = false;
        if (flag.compare_exchange_strong(expected, true)) {
            held = &flag;
        }
    }

    ~BusyGuard() {
        if (held != nullptr) {
            held->store(false);
        }
    }

    BusyGuard(const BusyGuard &) = delete;
    BusyGuard &operator=(const BusyGuard &) = delete;

    bool taken() const { return held != nullptr; }

    void hand_on() { held = nullptr; }
};

const char *MANIFEST_NAME = "manifest.json";
const char *VOCAB_NAME = "vocab.json";
const char *MERGES_NAME = "merges.txt";
const char *BOS_TOKEN = "<|startoftext|>";
const char *EOS_TOKEN = "<|endoftext|>";
const char *WORD_END = "</w>";

const char *NO_MODEL = "Govorilka: no model is loaded, so there is nothing to draw with.";

// The tokeniser's own word pattern, matched against lower-cased text with its whitespace
// already collapsed. Written here rather than read out of the folder because it belongs to the
// tokeniser class and not to a checkpoint: every model of this family splits words this way.
const char *WORD_PATTERN = "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|"
                           "[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+";

// The byte-to-letter map every byte-level vocabulary of this family is written in: the printable
// range keeps its own letter, and the 68 bytes with no printable letter are moved above 255 so
// that no byte spells as whitespace. Built once rather than tabulated, because the arithmetic
// is shorter than the table and cannot be mistyped.
void build_byte_letters(String out[256]) {
    bool printable[256] = {false};
    for (int b = '!'; b <= '~'; b++) {
        printable[b] = true;
    }
    for (int b = 0xA1; b <= 0xAC; b++) {
        printable[b] = true;
    }
    for (int b = 0xAE; b <= 0xFF; b++) {
        printable[b] = true;
    }
    int moved = 0;
    for (int b = 0; b < 256; b++) {
        const char32_t letter = printable[b] ? (char32_t)b : (char32_t)(256 + moved);
        if (!printable[b]) {
            moved++;
        }
        out[b] = String::chr(letter);
    }
}

float read_float(const Dictionary &from, const char *key, float fallback) {
    return from.has(key) ? (float)(double)from[key] : fallback;
}

int read_int(const Dictionary &from, const char *key, int fallback) {
    return from.has(key) ? (int)(int64_t)from[key] : fallback;
}

} // namespace

T2INoise::T2INoise(uint64_t value) :
        state(value != 0 ? value
                         : (uint64_t)Time::get_singleton()->get_ticks_usec() * 2654435761ULL + 1ULL) {
}

// splitmix64, then the top 24 bits as a fraction: taking the low bits instead reads the
// weakest part of the word, and dividing the whole 64 by 2^64 loses to rounding at 1.0.
float T2INoise::uniform() {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (float)((z >> 40) * (1.0 / 16777216.0));
}

// Box-Muller, both halves of a pair used. The log's argument is pushed off zero, because a
// draw of exactly zero is an infinity that spreads through the whole picture.
float T2INoise::normal() {
    if (has_spare) {
        has_spare = false;
        return spare;
    }
    const float u1 = uniform() + 1.0f / 16777216.0f;
    const float u2 = uniform();
    const float radius = sqrtf(-2.0f * logf(u1));
    const float angle = 6.283185307179586f * u2;
    spare = radius * sinf(angle);
    has_spare = true;
    return radius * cosf(angle);
}

// The worker is joined and nothing else: the graphs belong to the subclass, whose destructor
// has already run by the time this one does, so a virtual call from here would reach a table
// that is gone. A subclass calls unload() in its own destructor for its own graphs.
NcnnT2I::~NcnnT2I() {
    epoch.fetch_add(1);
    join_worker();
}

// The folder is opened through the engine rather than by a library, so a model inside an
// exported pack loads exactly as a folder beside the game does. A load while a picture is in
// flight would swap the graphs under the worker, so the worker is joined first, and the graphs
// are read under the same lock a run on any thread holds.
bool NcnnT2I::load(const String &model_dir, int num_threads) {
    unload();
    const double started = now_ms();
    // Half the cores when nobody chose: a picture is not the only thing on this processor, and
    // a language model answering beside it is the usual case rather than the odd one.
    threads = num_threads > 0 ? num_threads : std::max(1, ncnn::get_cpu_count() / 2);

    Ref<DirAccess> dir = DirAccess::open(model_dir);
    if (dir.is_null()) {
        problem = String("Govorilka: \"{0}\" is not a folder this project can open, so there is "
                         "no model to draw with.")
                          .format(Array::make(model_dir));
        return false;
    }
    std::lock_guard<std::mutex> hold(graphs_lock);

    problem = String();
    Dictionary manifest;
    if (!_read_manifest(model_dir, manifest, problem)
            || !_read_tokenizer(model_dir, problem)
            || !_load_graphs(model_dir, manifest, threads, problem)) {
        _unload_graphs();
        _clear_tokenizer();
        return false;
    }

    loaded.store(true);
    ran = false;
    load_ms = now_ms() - started;
    return true;
}

// What the folder says about itself. Everything the base needs is here rather than measured off
// the graphs: the schedule is the checkpoint's own scheduler, written down by the exporter, so
// no beta table and no timestep arithmetic has to be kept in step with a package's release.
bool NcnnT2I::_read_manifest(const String &folder, Dictionary &manifest, String &problem) {
    const String path = folder.path_join(MANIFEST_NAME);
    const String text = FileAccess::get_file_as_string(path);
    if (text.is_empty()) {
        problem = String("Govorilka: there is no readable {0} in \"{1}\". Every picture model "
                         "carries one, and it is what says how its graphs are run.")
                          .format(Array::make(MANIFEST_NAME, folder));
        return false;
    }
    const Variant parsed = JSON::parse_string(text);
    if (parsed.get_type() != Variant::DICTIONARY) {
        problem = String("Govorilka: {0} is not a JSON object.").format(Array::make(path));
        return false;
    }
    manifest = parsed;

    // Every row below has a sentence naming the one field to fix, because a manifest is written
    // by hand as often as by an exporter and "the model did not load" says nothing about which.
    const char *required[] = {"family", "scheduler", "sizes", "schedule"};
    for (const char *field : required) {
        if (!manifest.has(field)) {
            problem = String("Govorilka: {0} has no \"{1}\" in it.")
                              .format(Array::make(MANIFEST_NAME, field));
            return false;
        }
    }

    family = manifest["family"];
    if (family.is_empty()) {
        problem = String("Govorilka: {0} names no family, so nothing says what these graphs are.")
                          .format(Array::make(MANIFEST_NAME));
        return false;
    }
    const String named = manifest["scheduler"];
    if (named == "euler") {
        scheduler = SCHEDULER_EULER;
    } else if (named == "lcm") {
        scheduler = SCHEDULER_LCM;
    } else {
        problem = String("Govorilka: \"{0}\" is not a scheduler this addon has. It runs \"euler\" "
                         "and \"lcm\", and which one a model takes is not a guess.")
                          .format(Array::make(named));
        return false;
    }
    guidance = read_float(manifest, "guidance", 1.0f);
    latent_channels = read_int(manifest, "latent_channels", 4);
    latent_downscale = read_int(manifest, "latent_downscale", 8);
    init_noise_sigma = read_float(manifest, "init_noise_sigma", 1.0f);
    context_length = read_int(manifest, "context_length", 77);
    text_encoder_fp32 = manifest.has("text_encoder_fp32") ? (bool)manifest["text_encoder_fp32"]
                                                          : true;

    sizes.clear();
    const Array offered = manifest.has("sizes") ? (Array)manifest["sizes"] : Array();
    for (int i = 0; i < offered.size(); i++) {
        const Array pair = offered[i];
        if (pair.size() == 2) {
            sizes.push_back({(int)(int64_t)pair[0], (int)(int64_t)pair[1]});
        }
    }

    schedule.clear();
    const Array steps = manifest["schedule"];
    for (int i = 0; i < steps.size(); i++) {
        schedule.push_back(_step_of(steps[i]));
    }

    if (schedule.empty()) {
        problem = String("Govorilka: the manifest's schedule is empty, so there is no step to "
                         "run and no picture to make.");
        return false;
    }
    if (sizes.empty()) {
        problem = String("Govorilka: the manifest offers no sizes, so there is nothing this "
                         "model can be asked for.");
        return false;
    }
    if (latent_downscale <= 0 || latent_channels <= 0) {
        problem = String("Govorilka: the manifest describes a latent of {0} channels at one "
                         "part in {1}, which is not a shape.")
                          .format(Array::make(latent_channels, latent_downscale));
        return false;
    }
    for (const std::pair<int, int> &size : sizes) {
        if (size.first % latent_downscale != 0 || size.second % latent_downscale != 0) {
            problem = String("Govorilka: the manifest offers {0}x{1}, which does not divide by "
                             "the latent's {2}.")
                              .format(Array::make(size.first, size.second, latent_downscale));
            return false;
        }
    }
    return true;
}

NcnnT2I::Step NcnnT2I::_step_of(const Dictionary &row) {
    Step step;
    step.timestep = read_float(row, "timestep", 0.0f);
    step.input_scale = read_float(row, "input_scale", 1.0f);
    step.sigma = read_float(row, "sigma", 0.0f);
    step.sigma_next = read_float(row, "sigma_next", 0.0f);
    step.alpha_prod = read_float(row, "alpha_prod", 1.0f);
    step.alpha_prod_next = read_float(row, "alpha_prod_next", 1.0f);
    step.c_skip = read_float(row, "c_skip", 0.0f);
    step.c_out = read_float(row, "c_out", 1.0f);
    return step;
}

// The one place either road's arithmetic is written. The loop calls it once per channel and the
// bound `advance` calls it over a whole array, so a reference the second is held to is a
// reference the first is held to.
void NcnnT2I::_apply_step(float *sample, const float *predicted, int count, Scheduler road,
        const Step &step, bool last, T2INoise &noise) {
    for (int i = 0; i < count; i++) {
        if (road == SCHEDULER_EULER) {
            // Euler on an epsilon prediction: the move towards the next sigma is the prediction
            // itself, and the last step, whose next sigma is zero, lands on the denoised latent.
            sample[i] += predicted[i] * (step.sigma_next - step.sigma);
            continue;
        }
        // The consistency model's own update: undo the noise to reach x0, weigh it against the
        // sample by the boundary condition, then put back as much noise as the next step
        // expects. The last step keeps the answer rather than re-noising it.
        const float root_alpha = std::max(1e-8f, sqrtf(step.alpha_prod));
        const float root_rest = sqrtf(std::max(0.0f, 1.0f - step.alpha_prod));
        const float x0 = (sample[i] - root_rest * predicted[i]) / root_alpha;
        const float denoised = step.c_skip * sample[i] + step.c_out * x0;
        if (last) {
            sample[i] = denoised;
            continue;
        }
        sample[i] = sqrtf(step.alpha_prod_next) * denoised
                + sqrtf(std::max(0.0f, 1.0f - step.alpha_prod_next)) * noise.normal();
    }
}

// The vocabulary and the merge ranks out of the folder. Both are the tokeniser the export was
// traced with: a prompt tokenised any other way conditions the network on a sequence it was
// never shown, and the picture is of something else with no error anywhere.
bool NcnnT2I::_read_tokenizer(const String &folder, String &problem) {
    _clear_tokenizer();
    build_byte_letters(byte_letter);

    const String vocab_text = FileAccess::get_file_as_string(folder.path_join(VOCAB_NAME));
    if (vocab_text.is_empty()) {
        problem = String("Govorilka: there is no readable {0} in \"{1}\". It is the vocabulary "
                         "the export was traced with and the graphs cannot be read without it.")
                          .format(Array::make(VOCAB_NAME, folder));
        return false;
    }
    const Variant parsed = JSON::parse_string(vocab_text);
    if (parsed.get_type() != Variant::DICTIONARY) {
        problem = String("Govorilka: {0} is not a JSON object of names to numbers.")
                          .format(Array::make(folder.path_join(VOCAB_NAME)));
        return false;
    }
    const Dictionary table = parsed;
    const Array names = table.keys();
    for (int i = 0; i < names.size(); i++) {
        const String name = names[i];
        vocab.insert(name, (int)(int64_t)table[name]);
    }
    if (!vocab.has(BOS_TOKEN) || !vocab.has(EOS_TOKEN)) {
        _clear_tokenizer();
        problem = String("Govorilka: {0} has no {1} and {2} in it, so nothing marks where a "
                         "prompt starts and ends.")
                          .format(Array::make(VOCAB_NAME, BOS_TOKEN, EOS_TOKEN));
        return false;
    }
    bos_id = vocab[BOS_TOKEN];
    eos_id = vocab[EOS_TOKEN];

    const String merges_text = FileAccess::get_file_as_string(folder.path_join(MERGES_NAME));
    if (merges_text.is_empty()) {
        _clear_tokenizer();
        problem = String("Govorilka: there is no readable {0} in \"{1}\". Without the merge "
                         "table every word tokenises one letter at a time.")
                          .format(Array::make(MERGES_NAME, folder));
        return false;
    }
    const PackedStringArray lines = merges_text.split("\n");
    int rank = 0;
    for (int i = 0; i < lines.size(); i++) {
        const String line = lines[i].strip_edges();
        // The first line is the file's own version stamp, and a blank one is the trailing
        // newline. A rank is the order the pair appears in, so neither may be counted.
        if (line.is_empty() || line.begins_with("#")) {
            continue;
        }
        const PackedStringArray pair = line.split(" ");
        if (pair.size() != 2) {
            continue;
        }
        ranks.insert(pair[0] + String(" ") + pair[1], rank);
        rank++;
    }
    if (ranks.is_empty()) {
        _clear_tokenizer();
        problem = String("Govorilka: {0} holds no pairs at all.").format(Array::make(MERGES_NAME));
        return false;
    }

    splitter = RegEx::create_from_string(WORD_PATTERN);
    spaces = RegEx::create_from_string("\\s+");
    if (splitter.is_null() || spaces.is_null() || !splitter->is_valid() || !spaces->is_valid()) {
        _clear_tokenizer();
        problem = String("Govorilka: this build of the engine would not compile the tokeniser's "
                         "word pattern.");
        return false;
    }
    return true;
}

void NcnnT2I::_clear_tokenizer() {
    vocab.clear();
    ranks.clear();
    splitter.unref();
    spaces.unref();
    bos_id = -1;
    eos_id = -1;
}

// One word through the merge table, exactly as the tokeniser class does it: the word's letters
// with an end mark on the last, then the lowest-ranked adjacent pair joined over and over until
// no pair is in the table. A different order of joins is a different sequence of ids.
void NcnnT2I::_bpe(const String &token, std::vector<String> &pieces) const {
    pieces.clear();
    const int length = token.length();
    if (length == 0) {
        return;
    }
    for (int i = 0; i < length - 1; i++) {
        pieces.push_back(String::chr(token[i]));
    }
    pieces.push_back(String::chr(token[length - 1]) + String(WORD_END));
    if (pieces.size() == 1) {
        return;
    }

    while (true) {
        int best = -1;
        int at = -1;
        for (size_t i = 0; i + 1 < pieces.size(); i++) {
            const HashMap<String, int>::ConstIterator found =
                    ranks.find(pieces[i] + String(" ") + pieces[i + 1]);
            if (found != ranks.end() && (best < 0 || found->value < best)) {
                best = found->value;
                at = (int)i;
            }
        }
        if (at < 0) {
            break;
        }
        const String first = pieces[(size_t)at];
        const String second = pieces[(size_t)at + 1];
        std::vector<String> joined;
        size_t i = 0;
        while (i < pieces.size()) {
            if (i + 1 < pieces.size() && pieces[i] == first && pieces[i + 1] == second) {
                joined.push_back(first + second);
                i += 2;
                continue;
            }
            joined.push_back(pieces[i]);
            i++;
        }
        pieces.swap(joined);
        if (pieces.size() == 1) {
            break;
        }
    }
}

PackedInt32Array NcnnT2I::tokenize(const String &prompt) const {
    PackedInt32Array ids;
    if (splitter.is_null() || bos_id < 0) {
        return ids;
    }
    ids.push_back(bos_id);

    // Lower case with the whitespace collapsed, which is what the tokeniser is fed. A prompt
    // that keeps its line breaks splits into words the vocabulary has never seen.
    const String text = spaces->sub(prompt, " ", true).strip_edges().to_lower();
    const TypedArray<RegExMatch> words = splitter->search_all(text);
    std::vector<String> pieces;
    for (int i = 0; i < words.size() && ids.size() < context_length - 1; i++) {
        const Ref<RegExMatch> match = words[i];
        const CharString utf8 = match->get_string(0).utf8();
        String mapped;
        for (int b = 0; b < utf8.length(); b++) {
            mapped += byte_letter[(unsigned char)utf8[b]];
        }
        _bpe(mapped, pieces);
        for (size_t p = 0; p < pieces.size() && ids.size() < context_length - 1; p++) {
            const HashMap<String, int>::ConstIterator found = vocab.find(pieces[p]);
            if (found != vocab.end()) {
                ids.push_back(found->value);
            }
        }
    }

    ids.push_back(eos_id);
    // The window is padded with the end mark rather than with a pad of its own: that is what
    // the checkpoint was trained on and what the export was traced with.
    while (ids.size() < context_length) {
        ids.push_back(eos_id);
    }
    return ids;
}

Ref<Image> NcnnT2I::generate(const String &prompt, int64_t seed, int width, int height) {
    BusyGuard guard(busy);
    if (!guard.taken()) {
        return Ref<Image>();
    }
    dropped.store(false);
    return run(prompt, (uint64_t)seed, width, height, problem);
}

bool NcnnT2I::generate_async(const String &prompt, int64_t seed, int width, int height) {
    BusyGuard guard(busy);
    if (!guard.taken()) {
        // The picture in flight has already been thrown away, so what is left of it is only in
        // the way: joined here, on the road nobody is waiting on frames for.
        if (!dropped.load()) {
            return false;
        }
        join_worker();
        owed_at.store(-1);
        if (owed.exchange(false)) {
            busy.store(false);
        }
        BusyGuard again(busy);
        if (!again.taken()) {
            return false;
        }
        again.hand_on();
    } else {
        guard.hand_on();
    }

    dropped.store(false);
    const int64_t at = epoch.load();
    // The worker before this one has ended -- its delivery is what gave the flag back -- but a
    // thread that has ended is still joinable, and assigning over a joinable thread is
    // std::terminate(): the process dies with no message of any kind.
    join_worker();
    answer_ready.store(false);
    answer_taken.store(false);
    worker_at.store(at);
    // Marked owed before the thread exists: a delivery that raced this line would otherwise
    // lower the flag first and have the mark set over it afterwards.
    owed_at.store(at);
    owed.store(true);
    try {
        worker = std::thread(&NcnnT2I::work, this, prompt, (uint64_t)seed, width, height, at);
    } catch (...) {
        answer_taken.store(true);
        owed.store(false);
        owed_at.store(-1);
        busy.store(false);
        return false;
    }
    return true;
}

// The epoch moves and nothing waits. The worker keeps the graphs until it finishes and its
// delivery frees them; what it made is dropped on the way out rather than shown.
void NcnnT2I::cancel() {
    if (!busy.load()) {
        return;
    }
    epoch.fetch_add(1);
    dropped.store(true);
}

bool NcnnT2I::deliver_pending() {
    if (!answer_ready.load() || answer_taken.load()) {
        return false;
    }
    deliver(worker_at.load());
    return true;
}

bool NcnnT2I::wait_for_picture(int timeout_ms) {
    const double deadline = now_ms() + (double)std::max(0, timeout_ms);
    while (true) {
        if (deliver_pending()) {
            return true;
        }
        if (!busy.load()) {
            return false;
        }
        if (timeout_ms > 0 && now_ms() >= deadline) {
            return false;
        }
        OS::get_singleton()->delay_usec(500);
    }
}

bool NcnnT2I::is_busy() const {
    return busy.load();
}

bool NcnnT2I::is_loaded() const {
    return loaded.load();
}

// The worker is joined and then the lock is taken, in that order: the worker's run holds the
// lock, so taking it first would wait on a thread that is waiting to be joined. The blocking
// caller's flag is left to its own guard; only the worker's is lowered here.
void NcnnT2I::unload() {
    epoch.fetch_add(1);
    join_worker();
    {
        std::lock_guard<std::mutex> hold(graphs_lock);
        loaded.store(false);
        ran = false;
        _unload_graphs();
        _clear_tokenizer();
    }
    answer_ready.store(false);
    answer_taken.store(true);
    pending_picture.unref();
    owed_at.store(-1);
    if (owed.exchange(false)) {
        busy.store(false);
    }
    dropped.store(false);
}

String NcnnT2I::last_problem() const {
    return problem;
}

// Empty until a run has finished, so a caller reading them cannot mistake the numbers of a model
// that has drawn nothing for a picture that took no time.
Dictionary NcnnT2I::last_timings() const {
    Dictionary out;
    if (!ran) {
        return out;
    }
    out["load_ms"] = load_ms;
    out["total_ms"] = total_ms;
    out["tokenize_ms"] = tokenize_ms;
    out["text_ms"] = text_ms;
    out["denoise_ms"] = denoise_ms;
    out["decode_ms"] = decode_ms;
    out["width"] = last_width;
    out["height"] = last_height;
    out["steps"] = (int)schedule.size();
    _report_timings(out);
    return out;
}

Array NcnnT2I::offered_sizes() const {
    Array out;
    for (const std::pair<int, int> &size : sizes) {
        out.push_back(Vector2i(size.first, size.second));
    }
    return out;
}

String NcnnT2I::describe() const {
    return "ncnn";
}

bool NcnnT2I::_offers(int width, int height) const {
    for (const std::pair<int, int> &size : sizes) {
        if (size.first == width && size.second == height) {
            return true;
        }
    }
    return false;
}

String NcnnT2I::_size_refused(int width, int height) const {
    String offered;
    for (const std::pair<int, int> &size : sizes) {
        if (!offered.is_empty()) {
            offered += ", ";
        }
        offered += String::num_int64(size.first) + String("x") + String::num_int64(size.second);
    }
    return String("Govorilka: this model carries no graph for {0}x{1}. It offers {2}. A size is "
                  "a structure the export was written for, not a setting.")
            .format(Array::make(width, height, offered));
}

// One prompt through the family's graphs, under the lock that keeps the graphs in place until
// it returns. The run is fenced: an exception out of it would unwind into the engine, which has
// no handler and dies, where no picture is something the host is told about.
Ref<Image> NcnnT2I::run(const String &prompt, uint64_t seed, int width, int height, String &said) {
    std::lock_guard<std::mutex> hold(graphs_lock);
    // Read again with the lock held: an unload() between the busy flag and this line has
    // already freed the graphs, and the answer to that is nothing rather than a run.
    if (!loaded.load()) {
        said = String(NO_MODEL);
        return Ref<Image>();
    }
    if (!_offers(width, height)) {
        said = _size_refused(width, height);
        return Ref<Image>();
    }
    const double started = now_ms();

    said = String();
    Ref<Image> picture;
    try {
        picture = make(prompt, seed, width, height, said);
    } catch (const std::exception &thrown) {
        picture = Ref<Image>();
        said = _faulted(ncnn_report::describe(thrown));
    } catch (...) {
        picture = Ref<Image>();
        said = _faulted(ncnn_report::describe_unknown());
    }
    total_ms = now_ms() - started;
    return picture;
}

// The whole of a picture: the prompt to a token window, the window to a conditioning sequence,
// a seeded latent, the schedule's steps, and the latent to pixels. Everything here is shared by
// every family; what a family answers is which graph each of the three calls reaches.
Ref<Image> NcnnT2I::make(const String &prompt, uint64_t seed, int width, int height,
        String &problem_out) {
    last_width = width;
    last_height = height;

    doing("reading the prompt");
    double at = now_ms();
    const PackedInt32Array ids = tokenize(prompt);
    tokenize_ms = now_ms() - at;
    if (ids.size() != context_length) {
        problem_out = String("Govorilka: the prompt came to {0} tokens where this model takes a "
                             "window of {1}. The tokeniser did not load.")
                              .format(Array::make(ids.size(), context_length));
        return Ref<Image>();
    }

    doing("encoding the prompt");
    at = now_ms();
    ncnn::Mat hidden;
    if (!_encode_text(ids.ptr(), ids.size(), hidden, problem_out)) {
        return Ref<Image>();
    }
    // Cloned off the extractor that made it: the sequence is handed to the denoiser once per
    // step, and a blob still owned by the encoder's extractor is one an in-place layer of the
    // denoiser may write through on the first step and read as conditioning on the second.
    hidden = hidden.clone();
    ncnn::Mat blank;
    // A second sequence only when the manifest asks for one: guidance of 1 is the model's own
    // answer with nothing to weigh it against, and the second pass then costs a whole UNet.
    if (guidance > 1.0f) {
        const PackedInt32Array empty = tokenize(String());
        if (!_encode_text(empty.ptr(), empty.size(), blank, problem_out)) {
            return Ref<Image>();
        }
        blank = blank.clone();
    }
    text_ms = now_ms() - at;

    const int lw = width / latent_downscale;
    const int lh = height / latent_downscale;
    ncnn::Mat latent(lw, lh, latent_channels);
    if (latent.empty()) {
        problem_out = String("Govorilka: there was no room for a {0}x{1} latent.")
                              .format(Array::make(lw, lh));
        return Ref<Image>();
    }
    // Channel by channel and row by row within a channel, which is the order the reference
    // implementation flattens the same tensor in. A generator consumed in another order is a
    // different picture from the same seed.
    T2INoise noise(seed);
    for (int c = 0; c < latent_channels; c++) {
        float *write = latent.channel(c);
        for (int i = 0; i < lw * lh; i++) {
            write[i] = noise.normal() * init_noise_sigma;
        }
    }

    doing("reading the structure for this size");
    if (!_prepare_size(width, height, problem_out)) {
        return Ref<Image>();
    }

    denoise_ms = 0.0;
    for (size_t s = 0; s < schedule.size(); s++) {
        const Step &step = schedule[s];
        doing("running the denoiser");

        ncnn::Mat scaled(lw, lh, latent_channels);
        if (scaled.empty()) {
            problem_out = String("Govorilka: there was no room for the scaled latent.");
            return Ref<Image>();
        }
        for (int c = 0; c < latent_channels; c++) {
            const float *read = latent.channel(c);
            float *write = scaled.channel(c);
            for (int i = 0; i < lw * lh; i++) {
                write[i] = read[i] * step.input_scale;
            }
        }

        at = now_ms();
        ncnn::Mat predicted;
        if (!_denoise(scaled, hidden, step.timestep, width, height, predicted, problem_out)) {
            return Ref<Image>();
        }
        if (guidance > 1.0f) {
            ncnn::Mat unguided;
            if (!_denoise(scaled, blank, step.timestep, width, height, unguided, problem_out)) {
                return Ref<Image>();
            }
            for (int c = 0; c < latent_channels; c++) {
                float *write = predicted.channel(c);
                const float *read = unguided.channel(c);
                for (int i = 0; i < lw * lh; i++) {
                    write[i] = read[i] + guidance * (write[i] - read[i]);
                }
            }
        }
        denoise_ms += now_ms() - at;

        if (predicted.w != lw || predicted.h != lh || predicted.c != latent_channels) {
            problem_out = String("Govorilka: the denoiser answered {0}x{1}x{2} where the latent "
                                 "is {3}x{4}x{5}.")
                                  .format(Array::make(predicted.w, predicted.h, predicted.c,
                                          lw, lh, latent_channels));
            return Ref<Image>();
        }

        const bool last = s + 1 == schedule.size();
        for (int c = 0; c < latent_channels; c++) {
            _apply_step(latent.channel(c), predicted.channel(c), lw * lh, scheduler, step, last,
                    noise);
        }
    }

    doing("decoding the latent");
    at = now_ms();
    ncnn::Mat rgb;
    if (!_decode_latent(latent, width, height, rgb, problem_out)) {
        return Ref<Image>();
    }
    decode_ms = now_ms() - at;

    if (rgb.w != width || rgb.h != height || rgb.c != 3) {
        problem_out = String("Govorilka: the decoder answered {0}x{1}x{2} where a {3}x{4} "
                             "picture was asked for.")
                              .format(Array::make(rgb.w, rgb.h, rgb.c, width, height));
        return Ref<Image>();
    }

    // Flattened channel-first out of the runtime's own layout, which pads each channel to an
    // alignment, and then through the one conversion a test can reach without a model.
    PackedFloat32Array flat;
    flat.resize(width * height * 3);
    float *write = flat.ptrw();
    for (int c = 0; c < 3; c++) {
        memcpy(write + (size_t)c * width * height, rgb.channel(c),
                (size_t)width * height * sizeof(float));
    }
    ran = true;
    return to_image(flat, width, height);
}

// The decoder writes [-1, 1] per channel, which is the range this family is trained in. Clamped
// rather than scaled: a picture normalised to its own extremes changes colour from one prompt to
// the next, and nothing above says the extremes are wrong.
Ref<Image> NcnnT2I::to_image(const PackedFloat32Array &pixels, int width, int height) {
    if (width <= 0 || height <= 0 || pixels.size() != width * height * 3) {
        return Ref<Image>();
    }
    PackedByteArray bytes;
    bytes.resize(width * height * 3);
    uint8_t *write = bytes.ptrw();
    const float *read = pixels.ptr();
    for (int c = 0; c < 3; c++) {
        const float *plane = read + (size_t)c * width * height;
        for (int i = 0; i < width * height; i++) {
            const float value = std::min(std::max(plane[i] * 0.5f + 0.5f, 0.0f), 1.0f);
            write[i * 3 + c] = (uint8_t)(value * 255.0f + 0.5f);
        }
    }
    return Image::create_from_data(width, height, false, Image::FORMAT_RGB8, bytes);
}

PackedFloat32Array NcnnT2I::draw_noise(int64_t seed, int count) const {
    PackedFloat32Array out;
    if (count <= 0) {
        return out;
    }
    out.resize(count);
    float *write = out.ptrw();
    T2INoise noise((uint64_t)seed);
    for (int i = 0; i < count; i++) {
        write[i] = noise.normal();
    }
    return out;
}

PackedFloat32Array NcnnT2I::advance(const PackedFloat32Array &latent,
        const PackedFloat32Array &predicted, const Dictionary &step, bool last,
        int64_t seed) const {
    // Assigned rather than duplicated: the array copies itself the moment ptrw() is taken, so
    // the caller's latent is never the one written through.
    PackedFloat32Array out = latent;
    if (out.size() != predicted.size() || out.is_empty()) {
        return PackedFloat32Array();
    }
    Scheduler road = scheduler;
    if (step.has("scheduler")) {
        road = String(step["scheduler"]) == "lcm" ? SCHEDULER_LCM : SCHEDULER_EULER;
    }
    T2INoise noise((uint64_t)seed);
    _apply_step(out.ptrw(), predicted.ptr(), out.size(), road, _step_of(step), last, noise);
    return out;
}

// The worker's whole life: run, then hand the picture to the main thread. Emitting from here
// instead would put a signal on a thread the engine's listeners are not written for.
void NcnnT2I::work(String prompt, uint64_t seed, int width, int height, int64_t at) {
    // Fenced again out here, around everything the worker does and not only the model: an
    // exception that escapes a thread body is std::terminate, and the process then ends with no
    // line anywhere. Whatever happens, the delivery is still posted.
    try {
        pending_picture = run(prompt, seed, width, height, pending_problem);
    } catch (const std::exception &thrown) {
        pending_picture = Ref<Image>();
        pending_problem = _faulted(ncnn_report::describe(thrown));
    } catch (...) {
        pending_picture = Ref<Image>();
        pending_problem = _faulted(ncnn_report::describe_unknown());
    }
    answer_ready.store(true);
    callable_mp(this, &NcnnT2I::deliver).call_deferred(at);
}

// One sentence about a fault, with everything somebody would otherwise have to ask for: which
// class, what it was in the middle of, and what was thrown.
String NcnnT2I::_faulted(const String &thrown) const {
    return String("Govorilka: {0} faulted while {1}: {2}")
            .format(Array::make(get_class(), ncnn_report::last_note(), thrown));
}

// The delivery, on whichever thread got here first -- a frame, or a caller draining by hand.
// The flag is given back either way, because a picture that was cut off has to free the graphs
// too, and only a turn still current says anything.
void NcnnT2I::deliver(int64_t at) {
    if (answer_taken.exchange(true)) {
        return;
    }
    const bool current = at == epoch.load();
    if (owed.load() && owed_at.load() == at) {
        owed.store(false);
        owed_at.store(-1);
        busy.store(false);
    }
    if (!current) {
        pending_picture.unref();
        return;
    }
    problem = pending_problem;
    Ref<Image> picture = pending_picture;
    pending_picture.unref();
    if (picture.is_null()) {
        emit_signal("failed", pending_problem);
        return;
    }
    emit_signal("picture_ready", picture);
}

void NcnnT2I::join_worker() {
    if (worker.joinable()) {
        worker.join();
    }
}

void NcnnT2I::doing(const String &what) const {
    ncnn_report::note(get_class() + " " + what);
}


ncnn::Mat NcnnT2I::owned(const float *source, int w, int h) {
    return ncnn_util::owned(source, w, h);
}

ncnn::Mat NcnnT2I::owned_indices(const int *source, int count) {
    return ncnn_util::owned_indices(source, count);
}

String NcnnT2I::pick(const PackedStringArray &files, const String &mark, const String &suffix) {
    return ncnn_util::pick(files, mark, suffix);
}

double NcnnT2I::now_ms() {
    return ncnn_util::now_ms();
}

void NcnnT2I::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load", "model_dir", "num_threads"), &NcnnT2I::load);
    ClassDB::bind_method(D_METHOD("generate", "prompt", "seed", "width", "height"),
            &NcnnT2I::generate);
    ClassDB::bind_method(D_METHOD("generate_async", "prompt", "seed", "width", "height"),
            &NcnnT2I::generate_async);
    ClassDB::bind_method(D_METHOD("cancel"), &NcnnT2I::cancel);
    ClassDB::bind_method(D_METHOD("deliver_pending"), &NcnnT2I::deliver_pending);
    ClassDB::bind_method(D_METHOD("wait_for_picture", "timeout_ms"), &NcnnT2I::wait_for_picture);
    ClassDB::bind_method(D_METHOD("is_busy"), &NcnnT2I::is_busy);
    ClassDB::bind_method(D_METHOD("is_loaded"), &NcnnT2I::is_loaded);
    ClassDB::bind_method(D_METHOD("unload"), &NcnnT2I::unload);
    ClassDB::bind_method(D_METHOD("last_problem"), &NcnnT2I::last_problem);
    ClassDB::bind_method(D_METHOD("tokenize", "prompt"), &NcnnT2I::tokenize);
    ClassDB::bind_method(D_METHOD("draw_noise", "seed", "count"), &NcnnT2I::draw_noise);
    ClassDB::bind_method(D_METHOD("advance", "latent", "predicted", "step", "last", "seed"),
            &NcnnT2I::advance);
    ClassDB::bind_static_method("NcnnT2I", D_METHOD("to_image", "pixels", "width", "height"),
            &NcnnT2I::to_image);
    ClassDB::bind_method(D_METHOD("last_timings"), &NcnnT2I::last_timings);
    ClassDB::bind_method(D_METHOD("offered_sizes"), &NcnnT2I::offered_sizes);
    ClassDB::bind_method(D_METHOD("describe"), &NcnnT2I::describe);

    BIND_ENUM_CONSTANT(SCHEDULER_EULER);
    BIND_ENUM_CONSTANT(SCHEDULER_LCM);

    ADD_SIGNAL(MethodInfo("picture_ready",
            PropertyInfo(Variant::OBJECT, "picture", PROPERTY_HINT_RESOURCE_TYPE, "Image")));
    ADD_SIGNAL(MethodInfo("failed", PropertyInfo(Variant::STRING, "message")));
}
