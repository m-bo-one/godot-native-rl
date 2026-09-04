#include "register_types.h"

#include "gigaam_asr.h"
#include "ncnn_asr.h"
#include "ncnn_report.h"
#include "ncnn_runner.h"
#include "ncnn_t2i.h"
#include "ncnn_tts.h"
#include "piper_tts.h"
#include "sd1_t2i.h"
#include "sdxs_t2i.h"
#include "whisper_asr.h"

#include <godot_cpp/godot.hpp>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace godot;

// The OpenMP runtime this library is built against stays loaded for the life of the process.
// The engine frees the extension at shutdown while the runtime's worker threads are still
// parked in their spin, and a runtime unloaded under a spinning thread is an access violation
// on the way out of an otherwise clean exit. Pinned once here; a build without OpenMP has no
// such module and the call finds nothing to pin.
static void pin_openmp_runtime() {
#ifdef _WIN32
    HMODULE runtime = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, L"vcomp140.dll", &runtime);
#endif
}

void initialize_ncnn_runner_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    pin_openmp_runtime();
    // Before any class of this extension exists, so that a fault in one of them has somewhere
    // to be reported from. Without it a fault inside a worker thread ends the process with no
    // line in the log and no line on the terminal, which is how the defect this was written
    // for was found: by exit code, twice.
    ncnn_report::install_handlers();
    GDREGISTER_CLASS(NcnnRunner);
    // The recogniser every family extends is registered so a script can ask for it by name
    // and hold any family as one type; it has no instances of its own.
    GDREGISTER_ABSTRACT_CLASS(NcnnASR);
    GDREGISTER_CLASS(WhisperASR);
    GDREGISTER_CLASS(GigaAMASR);
    // The synthesiser every family extends, registered for the same reason as the recogniser:
    // a script asks for it by name and holds any family as one type. No instances of its own.
    GDREGISTER_ABSTRACT_CLASS(NcnnTTS);
    GDREGISTER_CLASS(PiperTTS);
    // The picture maker every family extends, registered for the same reason as the other two:
    // a script asks for it by name and holds any family as one type. No instances of its own.
    GDREGISTER_ABSTRACT_CLASS(NcnnT2I);
    GDREGISTER_CLASS(SdxsT2I);
    GDREGISTER_CLASS(Sd1T2I);
}

void uninitialize_ncnn_runner_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
}

extern "C" {
GDExtensionBool GDE_EXPORT ncnn_runner_library_init(
    GDExtensionInterfaceGetProcAddress p_get_proc_address,
    const GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization *r_initialization
) {
    GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);
    init_obj.register_initializer(initialize_ncnn_runner_module);
    init_obj.register_terminator(uninitialize_ncnn_runner_module);
    init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

    return init_obj.init();
}
}
