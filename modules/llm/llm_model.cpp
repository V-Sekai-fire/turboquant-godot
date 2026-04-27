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

#include <thirdparty/llama_cpp/include/llama.h>

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/io/resource_importer.h"
#include "core/object/class_db.h"

LLMModel::LLMModel() {
	llama_backend_init();
}

LLMModel::~LLMModel() {
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
	ClassDB::bind_method(D_METHOD("load"), &LLMModel::load);
	ClassDB::bind_method(D_METHOD("unload"), &LLMModel::unload);
	ClassDB::bind_method(D_METHOD("is_loaded"), &LLMModel::is_loaded);

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

Error LLMModel::load() {
	ERR_FAIL_COND_V_MSG(model_path.is_empty(), ERR_UNCONFIGURED, "LLMModel: model_path is not set.");
	ERR_FAIL_COND_V_MSG(model != nullptr, ERR_ALREADY_IN_USE, "LLMModel: model is already loaded.");

	// Resolve res:// paths through the importer switcharoo so llama.cpp gets
	// a real OS path to the file in .godot/imported/ rather than the virtual source.
	String resolved = model_path;
	if (model_path.begins_with("res://")) {
		String imported = ResourceFormatImporter::get_singleton()->get_internal_resource_path(model_path);
		resolved = ProjectSettings::get_singleton()->globalize_path(
				imported.is_empty() ? model_path : imported);
	}

	llama_model_params params = llama_model_default_params();
	params.n_gpu_layers = (n_gpu_layers < 0) ? INT_MAX : n_gpu_layers;
	params.use_mmap = use_mmap;
	params.use_mlock = use_mlock;

	model = llama_model_load_from_file(resolved.utf8().get_data(), params);
	if (model == nullptr) {
		String err = "LLMModel: failed to load model from " + model_path;
		emit_signal("load_failed", err);
		ERR_FAIL_V_MSG(FAILED, err);
	}

	emit_signal("loaded");
	return OK;
}

void LLMModel::unload() {
	if (model != nullptr) {
		llama_model_free(model);
		model = nullptr;
	}
}

bool LLMModel::is_loaded() const {
	return model != nullptr;
}
