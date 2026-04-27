/**************************************************************************/
/*  llm_model.h                                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/object/ref_counted.h"
#include "core/string/ustring.h"

#include "modules/llm/thirdparty/llama_cpp/include/llama.h"

// Loads a GGUF model file into memory.
// GPU acceleration is automatic: Metal on macOS, CUDA on Linux/Windows
// when the module was built with the respective backend (see SCsub).
//
// GDScript:
//   var model = LLMModel.new()
//   model.model_path = "/path/to/Qwen3.5-0.8B-heretic-Q4_K_M.gguf"
//   model.n_gpu_layers = -1  # -1 = all layers on GPU
//   var err = model.load()
class LLMModel : public RefCounted {
	GDCLASS(LLMModel, RefCounted);

	String model_path;
	int n_gpu_layers = -1; // -1 = all layers; 0 = CPU only
	bool use_mmap = true;
	bool use_mlock = false;

	llama_model *model = nullptr;

protected:
	static void _bind_methods();

public:
	void set_model_path(const String &p_path);
	String get_model_path() const;

	void set_n_gpu_layers(int p_layers);
	int get_n_gpu_layers() const;

	void set_use_mmap(bool p_enabled);
	bool get_use_mmap() const;

	void set_use_mlock(bool p_enabled);
	bool get_use_mlock() const;

	Error load();
	void unload();
	bool is_loaded() const;

	// Internal: accessed by LLMContext.
	llama_model *get_llama_model() const { return model; }

	LLMModel();
	~LLMModel();
};
