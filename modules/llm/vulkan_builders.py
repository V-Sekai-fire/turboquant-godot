"""vulkan_builders.py — SCons actions for compiling Vulkan shaders into the module.

Mirrors what llama.cpp's CMakeLists.txt does via vulkan-shaders-gen + glslc,
but as a Godot-style Python SCons action — no cmake, no ExternalProject.

Steps:
  1. Compile vulkan-shaders-gen.cpp → native host binary (once, cached).
  2. Run it once per .comp file: glslc compiles GLSL → SPIR-V, gen writes
     a per-shader .cpp and the shared ggml-vulkan-shaders.gen.hpp.
  3. The generated .cpp files + ggml-vulkan.cpp are compiled into the module.
"""

from __future__ import annotations

import glob
import os
import subprocess


def _host_cxx():
    """Return a usable host C++ compiler."""
    which_cmd = "where" if os.name == "nt" else "which"
    for cxx in ("clang++", "g++", "c++", "cl"):
        if subprocess.run([which_cmd, cxx], capture_output=True).returncode == 0:
            return cxx
    raise RuntimeError("No host C++ compiler found for vulkan-shaders-gen")


def _build_shaders_gen(shaders_dir: str, out_binary: str) -> None:
    """Compile vulkan-shaders-gen from source if missing or stale."""
    src = os.path.join(shaders_dir, "vulkan-shaders-gen.cpp")
    if os.path.exists(out_binary):
        if os.path.getmtime(out_binary) > os.path.getmtime(src):
            return  # up to date
    os.makedirs(os.path.dirname(out_binary), exist_ok=True)
    cxx = _host_cxx()
    cmd = [cxx, "-std=c++17", "-O2", src, "-o", out_binary]
    subprocess.check_call(cmd)


def _glslc_path(env_glslc: str = "") -> str:
    """Resolve glslc from GLSLC env var, VULKAN_SDK/bin, or PATH."""
    if env_glslc:
        return env_glslc
    glslc_env = os.environ.get("GLSLC", "")
    if glslc_env:
        return glslc_env
    sdk = os.environ.get("VULKAN_SDK", "")
    if sdk:
        # Windows SDK uses Bin\, Linux/macOS use bin/; .exe suffix on Windows.
        for subdir in ("bin", "Bin"):
            for exe in ("glslc", "glslc.exe"):
                candidate = os.path.join(sdk, subdir, exe)
                if os.path.exists(candidate):
                    return candidate
    # Fall back to PATH (works on Linux/macOS with shaderc installed)
    which_cmd = ["where", "glslc"] if os.name == "nt" else ["which", "glslc"]
    result = subprocess.run(which_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip().splitlines()[0]
    raise RuntimeError(
        "glslc not found. Options:\n"
        "  export VULKAN_SDK=/path/to/VulkanSDK/x.y.z\n"
        "  export GLSLC=/path/to/glslc\n"
        "Install the Vulkan SDK: https://vulkan.lunarg.com/"
    )


def generate_vulkan_shaders(vk_src_dir: str, out_dir: str, hpp_path: str) -> None:
    """Compile all Vulkan .comp shaders and write ggml-vulkan-shaders.hpp.

    Called directly from SCsub during the SCons reading phase so the header
    exists before any C++ compilation starts. Uses mtime to skip if current.
    """
    shaders_dir = os.path.join(vk_src_dir, "vulkan-shaders")
    comp_files = sorted(glob.glob(os.path.join(shaders_dir, "*.comp")))

    # Skip if hpp is newer than all .comp sources.
    if os.path.exists(hpp_path):
        hpp_mtime = os.path.getmtime(hpp_path)
        gen_src = os.path.join(shaders_dir, "vulkan-shaders-gen.cpp")
        newest_src = max(os.path.getmtime(f) for f in comp_files + [gen_src])
        if hpp_mtime > newest_src:
            return

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(hpp_path), exist_ok=True)

    # 1. Build vulkan-shaders-gen host binary
    gen_binary = os.path.join(out_dir, "vulkan-shaders-gen")
    _build_shaders_gen(shaders_dir, gen_binary)

    glslc = _glslc_path(os.environ.get("GLSLC", ""))

    # 2. Generate the HPP header skeleton first
    subprocess.check_call([
        gen_binary,
        "--output-dir",
        out_dir,
        "--target-hpp",
        hpp_path,
    ])

    # 3. Compile each .comp → SPIR-V → per-shader .cpp (also updates HPP)
    for comp in comp_files:
        name = os.path.basename(comp)
        target_cpp = os.path.join(out_dir, name + ".cpp")
        subprocess.check_call([
            gen_binary,
            "--glslc",
            glslc,
            "--source",
            comp,
            "--output-dir",
            out_dir,
            "--target-hpp",
            hpp_path,
            "--target-cpp",
            target_cpp,
        ])

    print(f"vulkan_builders: compiled {len(comp_files)} shaders → {hpp_path}")


def get_generated_shader_cpps(spv_dir: str) -> list[str]:
    """Return list of generated per-shader .cpp paths after generation."""
    return sorted(glob.glob(os.path.join(spv_dir, "*.comp.cpp")))


def generate_mingw_vulkan_implib(vulkan_headers_include: str, out_dir: str) -> str:
    """Generate libvulkan-1.dll.a for MinGW cross-compile.

    MinGW has no Vulkan import library in its sysroot. We extract the public
    Vulkan function names from the SDK headers and use dlltool to create a
    stub import library. vulkan-1.dll is guaranteed present on Windows 7+
    with any modern GPU driver, so runtime linking always works.

    Returns the path to the generated libvulkan-1.dll.a.
    """
    out_lib = os.path.join(out_dir, "libvulkan-1.dll.a")
    out_def = os.path.join(out_dir, "vulkan-1.def")

    # Skip if up to date
    vk_core_h = os.path.join(vulkan_headers_include, "vulkan", "vulkan_core.h")
    if (
        os.path.exists(out_lib)
        and os.path.exists(vk_core_h)
        and os.path.getmtime(out_lib) > os.path.getmtime(vk_core_h)
    ):
        return out_lib

    os.makedirs(out_dir, exist_ok=True)

    # Extract all vkXxx public API function names from vulkan_core.h
    import re

    functions = []
    vk_headers_dir = os.path.join(vulkan_headers_include, "vulkan")
    for header in sorted(glob.glob(os.path.join(vk_headers_dir, "vulkan*.h"))):
        with open(header, encoding="utf-8", errors="ignore") as f:
            src = f.read()
        # Match: VKAPI_ATTR ... VKAPI_CALL vkFunctionName (
        for m in re.finditer(r"VKAPI_CALL\s+(vk\w+)\s*\(", src):
            fn = m.group(1)
            if fn not in functions:
                functions.append(fn)

    with open(out_def, "w") as f:
        f.write("LIBRARY vulkan-1\nEXPORTS\n")
        for fn in functions:
            f.write(f"    {fn}\n")

    # Find the MinGW cross dlltool
    dlltool = "x86_64-w64-mingw32-dlltool"
    result = subprocess.run(["which", dlltool], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{dlltool} not found. Install mingw-w64:\n"
            "  brew install mingw-w64   (macOS)\n"
            "  apt install mingw-w64    (Linux)"
        )

    subprocess.check_call([
        dlltool,
        "--dllname",
        "vulkan-1.dll",
        "--def",
        out_def,
        "--output-lib",
        out_lib,
    ])
    print(f"vulkan_builders: generated {out_lib} ({len(functions)} exports)")
    return out_lib
