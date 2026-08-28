/**************************************************************************/
/*  llm_model.cpp                                                         */
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

#include "llm_model.h"

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/io/resource_importer.h"
#include "core/object/class_db.h"

#include <thirdparty/llama_cpp/include/llama.h>

#ifdef WEB_ENABLED
#include <cstdio>
#endif

LLMModel::LLMModel() {
}

LLMModel::~LLMModel() {
	if (worker.is_started()) {
		worker.wait_to_finish();
	}
	unload();
}

void LLMModel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_model_path", "path"), &LLMModel::set_model_path);
	ClassDB::bind_method(D_METHOD("get_model_path"), &LLMModel::get_model_path);
	ClassDB::bind_method(D_METHOD("set_n_gpu_layers", "layers"), &LLMModel::set_n_gpu_layers);
	ClassDB::bind_method(D_METHOD("get_n_gpu_layers"), &LLMModel::get_n_gpu_layers);
	ClassDB::bind_method(D_METHOD("set_use_mmap", "enabled"), &LLMModel::set_use_mmap);
	ClassDB::bind_method(D_METHOD("get_use_mmap"), &LLMModel::get_use_mmap);
	ClassDB::bind_method(D_METHOD("set_use_mlock", "enabled"), &LLMModel::set_use_mlock);
	ClassDB::bind_method(D_METHOD("get_use_mlock"), &LLMModel::get_use_mlock);
	ClassDB::bind_method(D_METHOD("set_mmproj_path", "path"), &LLMModel::set_mmproj_path);
	ClassDB::bind_method(D_METHOD("get_mmproj_path"), &LLMModel::get_mmproj_path);
	ClassDB::bind_method(D_METHOD("supports_vision"), &LLMModel::supports_vision);
	ClassDB::bind_method(D_METHOD("supports_audio"), &LLMModel::supports_audio);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "mmproj_path"), "set_mmproj_path", "get_mmproj_path");
	ClassDB::bind_method(D_METHOD("load"), &LLMModel::load);
	ClassDB::bind_method(D_METHOD("load_from_buffer", "data"), &LLMModel::load_from_buffer);
	ClassDB::bind_method(D_METHOD("unload"), &LLMModel::unload);
	ClassDB::bind_method(D_METHOD("is_loaded"), &LLMModel::is_loaded);
	ClassDB::bind_method(D_METHOD("is_loading"), &LLMModel::is_loading);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE, "*.gguf"),
			"set_model_path", "get_model_path");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "n_gpu_layers"), "set_n_gpu_layers", "get_n_gpu_layers");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_mmap"), "set_use_mmap", "get_use_mmap");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_mlock"), "set_use_mlock", "get_use_mlock");

	ADD_SIGNAL(MethodInfo("loaded"));
	ADD_SIGNAL(MethodInfo("load_failed", PropertyInfo(Variant::STRING, "error")));
}

void LLMModel::set_model_path(const String &p_path) {
	model_path = p_path;
}
String LLMModel::get_model_path() const {
	return model_path;
}
void LLMModel::set_n_gpu_layers(int p_layers) {
	n_gpu_layers = p_layers;
}
int LLMModel::get_n_gpu_layers() const {
	return n_gpu_layers;
}
void LLMModel::set_use_mmap(bool p_enabled) {
	use_mmap = p_enabled;
}
bool LLMModel::get_use_mmap() const {
	return use_mmap;
}
void LLMModel::set_use_mlock(bool p_enabled) {
	use_mlock = p_enabled;
}
bool LLMModel::get_use_mlock() const {
	return use_mlock;
}

void LLMModel::_load_thread(void *p_userdata) {
	static_cast<LLMModel *>(p_userdata)->_do_load();
}

void LLMModel::_do_load() {
	String resolved = model_path;
	if (model_path.begins_with("res://")) {
		String imported = ResourceFormatImporter::get_singleton()->get_internal_resource_path(model_path);
		resolved = ProjectSettings::get_singleton()->globalize_path(
				imported.is_empty() ? model_path : imported);
	} else if (model_path.begins_with("user://")) {
		resolved = ProjectSettings::get_singleton()->globalize_path(model_path);
	}

	llama_model_params params = llama_model_default_params();
	params.n_gpu_layers = (n_gpu_layers < 0) ? INT_MAX : n_gpu_layers;
	params.use_mmap = use_mmap;
	params.use_mlock = use_mlock;

	llama_model *loaded = nullptr;

#ifdef WEB_ENABLED
	if (!model_data.is_empty()) {
		// Data came from load_from_buffer() — already in WASM linear memory,
		// accessible from this pthread. Use fmemopen to get a FILE* for llama.cpp.
		params.use_mmap = false;
		FILE *fp = fmemopen(model_data.ptrw(), model_data.size(), "rb");
		if (fp) {
			loaded = llama_model_load_from_file_ptr(fp, params);
			fclose(fp);
		}
		model_data.clear();
	} else {
		loaded = llama_model_load_from_file(resolved.utf8().get_data(), params);
	}
#else
	loaded = llama_model_load_from_file(resolved.utf8().get_data(), params);
#endif

	// The projector needs the text model, so it is built here on the worker
	// thread rather than in load(). A projector failure is not fatal: the model
	// still serves text, and supports_vision() reports false.
#ifdef LLM_HAS_MTMD
	mtmd_context *loaded_mctx = nullptr;
	if (loaded != nullptr && !mmproj_path.is_empty()) {
		String mm = mmproj_path;
		if (mm.begins_with("res://") || mm.begins_with("user://")) {
			mm = ProjectSettings::get_singleton()->globalize_path(mm);
		}
		mtmd_context_params mp = mtmd_context_params_default();
		mp.use_gpu = (n_gpu_layers != 0);
		mp.n_threads = 4;
		mp.print_timings = false;
		loaded_mctx = mtmd_init_from_file(mm.utf8().get_data(), loaded, mp);
		if (loaded_mctx == nullptr) {
			// STUB: vision is not wired up for every projector this tree can be pointed at.
			// Gemma-4's mmproj declares projector type "gemma4uv", which the vendored
			// clip.cpp does not register, so mtmd_init_from_file returns null here -- the
			// file is fine, the loader simply does not know that type yet. Naming that in
			// the message keeps it from being read as a corrupt or missing download.
			//
			// The failure stays non-fatal on purpose: text generation is unaffected and
			// supports_vision() reports false, so callers can branch on it rather than
			// discovering the gap through a crash. Media parts sent to LLMChat are dropped
			// with their own warning.
			ERR_PRINT("LLMModel: could not load the projector at " + mmproj_path +
					" — continuing text-only. If the log above says \"unknown projector type\", "
					"this build has no support for that projector yet; vision is stubbed off "
					"rather than broken.");
		}
	}
#endif

	{
		MutexLock lock(worker_mutex);
		loading = false;
		if (loaded != nullptr) {
			model = loaded;
#ifdef LLM_HAS_MTMD
			mctx = loaded_mctx;
#endif
		}
	}
	if (loaded == nullptr) {
		call_deferred("emit_signal", "load_failed", String("LLMModel: failed to load model from ") + model_path);
	} else {
		call_deferred("emit_signal", "loaded");
	}
}

Error LLMModel::load() {
	ERR_FAIL_COND_V_MSG(model_path.is_empty(), ERR_UNCONFIGURED, "LLMModel: model_path is not set.");
	MutexLock lock(worker_mutex);
	ERR_FAIL_COND_V_MSG(loading, ERR_BUSY, "LLMModel: already loading.");
	ERR_FAIL_COND_V_MSG(model != nullptr, ERR_ALREADY_IN_USE, "LLMModel: model is already loaded.");
	loading = true;
	if (worker.is_started()) {
		worker.wait_to_finish();
	}
	worker.start(_load_thread, this);
	return OK;
}

Error LLMModel::load_from_buffer(const PackedByteArray &p_data) {
	ERR_FAIL_COND_V_MSG(p_data.is_empty(), ERR_INVALID_PARAMETER, "LLMModel: buffer is empty.");
	MutexLock lock(worker_mutex);
	ERR_FAIL_COND_V_MSG(loading, ERR_BUSY, "LLMModel: already loading.");
	ERR_FAIL_COND_V_MSG(model != nullptr, ERR_ALREADY_IN_USE, "LLMModel: model is already loaded.");
	if (worker.is_started()) {
		worker.wait_to_finish();
	}
	model_data.resize(p_data.size());
	memcpy(model_data.ptrw(), p_data.ptr(), p_data.size());
	loading = true;
	if (worker.is_started()) {
		worker.wait_to_finish();
	}
	worker.start(_load_thread, this);
	return OK;
}

void LLMModel::set_mmproj_path(const String &p_path) {
	mmproj_path = p_path;
}

String LLMModel::get_mmproj_path() const {
	return mmproj_path;
}

bool LLMModel::supports_vision() const {
#ifdef LLM_HAS_MTMD
	return mctx != nullptr && mtmd_support_vision(mctx);
#else
	return false;
#endif
}

bool LLMModel::supports_audio() const {
#ifdef LLM_HAS_MTMD
	return mctx != nullptr && mtmd_support_audio(mctx);
#else
	return false;
#endif
}

void LLMModel::unload() {
	MutexLock lock(worker_mutex);
#ifdef LLM_HAS_MTMD
	if (mctx != nullptr) {
		mtmd_free(mctx);
		mctx = nullptr;
	}
#endif
	if (model != nullptr) {
		llama_model_free(model);
		model = nullptr;
	}
}

bool LLMModel::is_loaded() const {
	return model != nullptr;
}
bool LLMModel::is_loading() const {
	return loading;
}
