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

#include <thirdparty/llama_cpp/include/llama.h>

#include "core/object/ref_counted.h"
#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "core/string/ustring.h"

// Loads a GGUF model file into memory asynchronously.
// Call load() then await the loaded or load_failed signal.
//
// GDScript:
//   var model = LLMModel.new()
//   model.model_path = "/path/to/model.gguf"
//   model.loaded.connect(func(): print("ready"))
//   model.load_failed.connect(func(e): push_error(e))
//   model.load()
class LLMModel : public RefCounted {
	GDCLASS(LLMModel, RefCounted);

	String model_path;
	int n_gpu_layers = -1;
	bool use_mmap = true;
	bool use_mlock = false;

	llama_model *model = nullptr;
	Thread worker;
	Mutex worker_mutex;
	bool loading = false;

	static void _load_thread(void *p_userdata);
	void _do_load();

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

	// Async: returns immediately, emits loaded or load_failed when done.
	Error load();
	void unload();
	bool is_loaded() const;
	bool is_loading() const;

	llama_model *get_llama_model() const { return model; }

	LLMModel();
	~LLMModel();
};
