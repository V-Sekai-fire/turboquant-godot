"""webgpu_builders.py — SCons action for embedding WGSL shaders into the LLM module.

Mirrors what llama.cpp's CMakeLists does via embed_wgsl.py,
but as a Godot-style Python SCons action — no cmake, no ExternalProject.

Steps:
  1. Run embed_wgsl.py against the wgsl-shaders/ directory.
  2. This produces ggml-wgsl-shaders.hpp in the ggml-webgpu/ source dir.
  3. ggml-webgpu.cpp pulls it in via ggml-webgpu-shader-lib.hpp.
"""

from __future__ import annotations

import os
import subprocess
import sys


def generate_webgpu_shaders(wgpu_src_dir: str, hpp_path: str) -> None:
    """Embed all WGSL shaders into ggml-wgsl-shaders.hpp.

    Called directly from SCsub during the SCons reading phase so the header
    exists before any C++ compilation starts. Uses mtime to skip if current.
    """
    shader_dir = os.path.join(wgpu_src_dir, "wgsl-shaders")
    embed_script = os.path.join(shader_dir, "embed_wgsl.py")

    # Skip if hpp is newer than all shader sources.
    if os.path.exists(hpp_path):
        hpp_mtime = os.path.getmtime(hpp_path)
        wgsl_files = [
            os.path.join(shader_dir, f) for f in os.listdir(shader_dir) if f.endswith((".wgsl", ".tmpl", ".py"))
        ]
        if wgsl_files and hpp_mtime > max(os.path.getmtime(f) for f in wgsl_files):
            return

    os.makedirs(os.path.dirname(hpp_path), exist_ok=True)

    subprocess.check_call(
        [
            sys.executable,
            embed_script,
            "--input_dir",
            shader_dir,
            "--output_file",
            hpp_path,
        ]
    )

    print(f"webgpu_builders: embedded WGSL shaders → {hpp_path}")
