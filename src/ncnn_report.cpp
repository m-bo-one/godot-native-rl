#include "ncnn_report.h"

#include <godot_cpp/variant/utility_functions.hpp>

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

using namespace godot;

namespace {

// Room for the class and the operation and nothing else. Written and read without a lock on
// purpose -- see the header -- so it is a fixed array rather than anything that allocates.
constexpr int NOTE_SIZE = 256;
char last[NOTE_SIZE] = "nothing yet";

std::terminate_handler previous = nullptr;

// Both roads out. push_error is what reaches Godot's log file and the editor's panel; stderr
// is what reaches a terminal when the engine is already too far gone to print. A dying
// process is not the place to choose between them.
void say(const char *line) {
    UtilityFunctions::push_error(String(line));
    fputs(line, stderr);
    fputc('\n', stderr);
    fflush(stderr);
}

void on_terminate() {
    char line[NOTE_SIZE + 160];
    snprintf(line, sizeof(line),
            "Govorilka: the ncnn extension is ending the process. The last thing it started "
            "was: %s. This is a fault inside the extension, not a setting.", last);
    say(line);
    if (previous != nullptr) {
        previous();
    }
    abort();
}

#ifdef _WIN32
LONG WINAPI on_unhandled(EXCEPTION_POINTERS *info) {
    char line[NOTE_SIZE + 160];
    snprintf(line, sizeof(line),
            "Govorilka: the ncnn extension faulted (code 0x%08lX). The last thing it started "
            "was: %s.",
            info != nullptr ? info->ExceptionRecord->ExceptionCode : 0UL, last);
    say(line);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

} // namespace

void ncnn_report::note(const String &what) {
    const CharString bytes = what.utf8();
    const int length = bytes.length() < NOTE_SIZE - 1 ? bytes.length() : NOTE_SIZE - 1;
    memcpy(last, bytes.get_data(), (size_t)length);
    last[length] = 0;
}

String ncnn_report::last_note() {
    return String(last);
}

String ncnn_report::describe(const std::exception &thrown) {
    const char *what = thrown.what();
    return String(what != nullptr && what[0] != 0 ? what : "an exception with nothing to say");
}

String ncnn_report::describe_unknown() {
    return String("something that is not a standard exception, so it carries no message");
}

// Once, at initialisation. Installing this from a worker would leave every thread started
// before it uninstrumented, and installing it twice would chain the handler to itself.
void ncnn_report::install_handlers() {
    static bool installed = false;
    if (installed) {
        return;
    }
    installed = true;
    previous = std::set_terminate(on_terminate);
#ifdef _WIN32
    // It catches an access violation and it does NOT catch std::terminate: MSVC ends that with
    // __fastfail, which is not a Windows exception and reaches no filter. The terminate
    // handler above is the one that covers the failure this was written for; this covers the
    // other kind.
    SetUnhandledExceptionFilter(on_unhandled);
#endif
}
