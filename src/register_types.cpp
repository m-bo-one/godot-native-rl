#include "register_types.h"

#include "ncnn_asr.h"
#include "ncnn_runner.h"
#include "whisper_asr.h"

#include <godot_cpp/godot.hpp>

using namespace godot;

void initialize_ncnn_runner_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    ClassDB::register_class<NcnnRunner>();
    // The recogniser every family extends is registered so a script can ask for it by name
    // and hold any family as one type; it has no instances of its own.
    GDREGISTER_ABSTRACT_CLASS(NcnnASR);
    GDREGISTER_CLASS(WhisperASR);
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
