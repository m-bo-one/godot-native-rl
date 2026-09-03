#ifndef NCNN_REPORT_H
#define NCNN_REPORT_H

#include <godot_cpp/variant/string.hpp>

#include <exception>

namespace godot {

// What this extension says when it dies, and what it leaves behind so it can say anything at
// all. A fault inside a worker thread used to end the process with no line anywhere: MSVC's
// std::terminate calls __fastfail, which writes nothing, unwinds nothing, and is not a
// Windows exception either -- so a structured handler never sees it and the log's last line
// is whatever was printed before. The only place left to speak from is std::terminate itself.
namespace ncnn_report {

// The last thing the extension started doing, kept so the handler below has something to name
// when it runs. Written on whatever thread is working and read on whatever thread is dying:
// a fixed buffer with no lock, because a handler that blocks on a mutex held by a thread that
// is already gone prints nothing at all. A torn read is a garbled hint, which beats silence.
void note(const String &what);
String last_note();

// The text of whatever was thrown, as a sentence rather than a type name. Anything that is
// not a std::exception has none, and says so.
String describe(const std::exception &thrown);
String describe_unknown();

// Installed once, at library initialisation. From then on a fault that would have ended the
// process in silence prints one line first: the extension, the class, and what it was in the
// middle of. It does not stop the process -- nothing can, past this point -- it makes the
// crash reportable.
void install_handlers();

} // namespace ncnn_report

} // namespace godot

#endif // NCNN_REPORT_H
