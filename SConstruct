#!/usr/bin/env python
import os
import subprocess

from SCons.Script import ARGUMENTS, Default, Exit, File, Glob, SConscript

# The speech recognisers fence a decode in a try and answer a worker thread that could not be
# started with false; godot-cpp's own default, disable_exceptions=yes, compiles both fences
# away and lets either fault unwind into the engine. The default is turned here, before the
# bindings read it, and src/ncnn_asr.cpp refuses to compile without exceptions either way.
ARGUMENTS.setdefault("disable_exceptions", "no")

# Pulls in godot-cpp's build env. By default this also compiles the bindings library; pass
# `build_library=no` to link an already-built godot-cpp/bin/libgodot-cpp.*.a instead (CI does
# this to avoid recompiling the bindings on every extension build — see ci.yml / issue #85).
env = SConscript("godot-cpp/SConstruct")
sources = Glob("src/*.cpp")
env.Append(CPPPATH=["src"])
requested_arch = str(ARGUMENTS.get("arch", env.get("arch", "")))

# The ncnn revision this project is built and measured against. It is not master: measured,
# a 2026-09-02 master faults with an integer divide by zero part-way through a Whisper decode
# that the same sources decode correctly on this tag. docs/dev/building.md says how that was
# established; the string a built library reports is the build DATE and not this.
NCNN_PIN = "tag 20260526 (commit e54f7b1f88434e1d844ea0551b880a1cfb079ce1)"

project_dir = os.path.abspath(".")
ncnn_root = os.path.join(project_dir, "thirdparty", "ncnn")
if not os.path.isdir(ncnn_root):
    print("Error: ncnn root was not found at {}".format(ncnn_root))
    print("It is a submodule at {} -- run `git submodule update --init`.".format(NCNN_PIN))
    Exit(1)

# Per-target cross/native build dir (e.g. build-linux-x86_64, build-windows-x86_64), so a
# cross-compiled ncnn is preferred over the host build/install (which holds the host arch).
_target_arch = requested_arch or str(env.get("arch", ""))
ncnn_target_build = os.path.join(ncnn_root, "build-{}-{}".format(env["platform"], _target_arch))

ncnn_include_candidates = [
    os.path.join(ncnn_target_build, "install", "include"),
    os.path.join(ncnn_target_build, "install", "include", "ncnn"),
    os.path.join(ncnn_root, "include"),
    os.path.join(ncnn_root, "src"),
    os.path.join(ncnn_root, "build", "src"),
    os.path.join(ncnn_root, "build-arm64", "src"),
    os.path.join(ncnn_root, "build-x86_64", "src"),
    os.path.join(ncnn_root, "install", "include"),
    os.path.join(ncnn_root, "install", "include", "ncnn"),
    os.path.join(ncnn_root, "install-arm64", "include"),
    os.path.join(ncnn_root, "install-arm64", "include", "ncnn"),
    os.path.join(ncnn_root, "install-x86_64", "include"),
    os.path.join(ncnn_root, "install-x86_64", "include", "ncnn"),
    os.path.join(ncnn_root, "build", "install", "include"),
    os.path.join(ncnn_root, "build", "install", "include", "ncnn"),
]
ncnn_include_paths = []
for path in ncnn_include_candidates:
    if os.path.isdir(path) and path not in ncnn_include_paths:
        ncnn_include_paths.append(path)
if not ncnn_include_paths:
    print("Error: no ncnn include directories found under {}".format(ncnn_root))
    Exit(1)
env.Append(CPPPATH=ncnn_include_paths)

if env["platform"] == "windows":
    ncnn_static_candidates = [
        os.path.join(ncnn_target_build, "install", "lib", "ncnn.lib"),
        os.path.join(ncnn_target_build, "install", "lib", "libncnn.a"),
        os.path.join(ncnn_root, "build", "install", "lib", "ncnn.lib"),
        os.path.join(ncnn_root, "build", "install", "lib", "libncnn.a"),
    ]
else:
    ncnn_static_candidates = [
        os.path.join(ncnn_target_build, "install", "lib", "libncnn.a"),
        os.path.join(ncnn_root, "build", "install", "lib", "libncnn.a"),
        os.path.join(ncnn_root, "build", "src", "libncnn.a"),
    ]

ncnn_static_lib = next((path for path in ncnn_static_candidates if os.path.isfile(path)), None)
if ncnn_static_lib is None:
    print("Error: could not find static ncnn library. Looked in:")
    for path in ncnn_static_candidates:
        print("  - {}".format(path))
    print("Build it from {} -- see docs/dev/building.md.".format(NCNN_PIN))
    Exit(1)

if env["platform"] == "macos":
    try:
        lipo_info = subprocess.check_output(["lipo", "-info", ncnn_static_lib], text=True).strip()
    except Exception:
        lipo_info = ""

    ncnn_arches = set()
    if " are: " in lipo_info:
        ncnn_arches.update(lipo_info.split(" are: ", 1)[1].split())
    elif " architecture: " in lipo_info:
        ncnn_arches.add(lipo_info.split(" architecture: ", 1)[1].strip())

    if requested_arch == "universal":
        missing = {"arm64", "x86_64"} - ncnn_arches
        if missing:
            print("Error: building macOS universal, but ncnn static lib is missing architectures: {}.".format(", ".join(sorted(missing))))
            print("ncnn lib architectures detected: {}".format(", ".join(sorted(ncnn_arches)) if ncnn_arches else "(unknown)"))
            print("Fix options:")
            print("  1) Build extension single-arch: scons platform=macos arch=arm64 target=template_debug")
            print("  2) Build arm64 and x86_64 ncnn separately, then merge libncnn.a with lipo (see README).")
            Exit(1)
    elif requested_arch in ("arm64", "x86_64") and ncnn_arches and requested_arch not in ncnn_arches:
        print("Error: macOS arch={} requested, but ncnn static lib has architectures: {}.".format(
            requested_arch, ", ".join(sorted(ncnn_arches))
        ))
        print("Rebuild ncnn for {} or build with matching arch.".format(requested_arch))
        Exit(1)

# Link ncnn statically into the extension.
env.Append(LIBS=[File(ncnn_static_lib)])

# An ncnn built with NCNN_VULKAN=ON compiles its compute shaders through glslang, which it builds
# from its own submodule and installs beside ncnn.lib. Those archives have to be linked here or
# every ncnn::create_pipeline reference is undefined; a Vulkan ncnn with no glslang beside it is
# refused by name rather than left to the linker. NCNN_VULKAN is read off the installed
# platform.h, which is the header the extension will actually compile against.
_ncnn_lib_dir = os.path.dirname(ncnn_static_lib)
_ncnn_vulkan = False
for _include in ncnn_include_paths:
    _platform_h = os.path.join(_include, "platform.h")
    if os.path.isfile(_platform_h):
        with open(_platform_h, "r", encoding="utf-8", errors="replace") as _reading:
            _ncnn_vulkan = "#define NCNN_VULKAN 1" in _reading.read()
        break

if _ncnn_vulkan:
    # glslang's archives in dependency order, dependents first: GNU ld reads an archive once and
    # takes only what is undefined at that point, so SPIRV after glslang leaves every
    # GlslangToSpv reference unresolved. Alphabetical order is what a listing gives and it is
    # wrong. Names that a given glslang folds into another are simply not there and are skipped;
    # anything else the install left is appended after, where its order cannot matter.
    _glslang_order = [
        "SPIRV", "glslang-default-resource-limits", "glslang",
        "MachineIndependent", "GenericCodeGen", "OSDependent",
    ]
    # The two the linker cannot do without. A guard that only asked for "some archive" passes on
    # an install that has ncnn's own leftovers in it and nothing of the shader compiler.
    _glslang_needed = ["SPIRV", "glslang"]
    _suffix = ".lib" if env["platform"] == "windows" else ".a"
    _prefix = "" if env["platform"] == "windows" else "lib"

    def _archive_named(stem):
        path = os.path.join(_ncnn_lib_dir, _prefix + stem + _suffix)
        return path if os.path.isfile(path) else None

    _missing = [stem for stem in _glslang_needed if _archive_named(stem) is None]
    if _missing:
        print("Error: ncnn at {} is built with Vulkan and {} is not beside it in {}.".format(
            ncnn_static_lib, " and ".join(_missing), _ncnn_lib_dir))
        print("Fetch ncnn's own submodules (`git submodule update --init` inside thirdparty/ncnn)")
        print("and build ncnn again -- see docs/dev/building.md.")
        Exit(1)

    _beside = [_prefix + stem + _suffix for stem in _glslang_order if _archive_named(stem)]
    _beside += sorted(name for name in os.listdir(_ncnn_lib_dir)
                      if name.endswith(_suffix) and name not in _beside
                      and os.path.join(_ncnn_lib_dir, name) != ncnn_static_lib)
    env.Append(LIBS=[File(os.path.join(_ncnn_lib_dir, name)) for name in _beside])
    print("ncnn is built with Vulkan; linking {} beside it".format(", ".join(_beside)))

# An ncnn built with OpenMP makes libncnn.a reference the GNU OpenMP runtime (GOMP_parallel)
# and pthreads. Those must be linked AFTER libncnn.a — so the linker's default --as-needed
# keeps libgomp in DT_NEEDED — or the extension fails to load on Linux with
# "undefined symbol: GOMP_parallel". (macOS resolves its own OpenMP runtime, so scope to Linux.)
# Since #103 the project's Linux builds (build_linux_native.sh, ci.yml, the zig cross-compile)
# all build ncnn with NCNN_OPENMP=OFF and pass ncnn_openmp=no, shipping a self-contained .so
# with no libgomp runtime dependency. The default stays "yes" so a plain `scons platform=linux`
# against a stock OpenMP ncnn still links correctly — set ncnn_openmp=no to match an
# NCNN_OPENMP=OFF static lib.
#
# On Windows the same flag means the same thing and is spelled differently: MSVC picks its
# OpenMP runtime from the /openmp switch, so an ncnn built with NCNN_OPENMP=ON needs it here
# or the link fails on _vcomp symbols. The default is "no" there because the self-contained
# flavour is an NCNN_OPENMP=OFF static lib, and passing /openmp for a library that never calls
# OpenMP would add a vcomp140.dll dependency for nothing.
_openmp_default = "no" if env["platform"] == "windows" else "yes"
_ncnn_openmp = str(ARGUMENTS.get("ncnn_openmp", _openmp_default)).lower() not in ("0", "no", "false")
if env["platform"] == "linux":
    env.Append(LIBS=(["gomp", "pthread"] if _ncnn_openmp else ["pthread"]))
elif env["platform"] == "windows" and _ncnn_openmp:
    env.Append(CCFLAGS=["/openmp"])

# ncnn's Android backend pulls in the platform asset-manager (AAsset_*, in libandroid) and logging
# (__android_log_print, in liblog). Link them so they land in the extension's own DT_NEEDED — must
# come after libncnn.a. Without this the symbols are left to the host, so a plain dlopen() of the
# .so fails to load with "cannot locate symbol AAsset_seek": bionic doesn't resolve a dlopen'd
# library against the host executable's libs the way glibc does. Self-contained = loads anywhere (#102).
if env["platform"] == "android":
    env.Append(LIBS=["android", "log"])

# When cross-compiling from a macOS host, SCons keeps host-derived suffixes: the shared
# library would be named ".dylib" (not ".so") and shared objects ".os" (which clang/zig
# reject as an "unrecognized file extension"). Pin the target's own suffixes. godot-cpp's
# own objects are already built at this point, so changing SHOBJSUFFIX is safe.
if env["platform"] == "linux":
    env["SHLIBSUFFIX"] = ".so"
    env["SHOBJSUFFIX"] = ".o"
elif env["platform"] == "windows":
    env["SHLIBSUFFIX"] = ".dll"
    env["SHOBJSUFFIX"] = ".o"
elif env["platform"] == "android":
    # SHLIBSUFFIX is already ".so" (set by godot-cpp's android tool); just fix the
    # shared-object extension the NDK clang would otherwise reject on a macOS host.
    env["SHOBJSUFFIX"] = ".o"

library = env.SharedLibrary(
    target=os.path.join("addons", "godot_native_rl", "bin", "libncnn_runner{}{}".format(env["suffix"], env["SHLIBSUFFIX"])),
    source=sources,
)

Default(library)
