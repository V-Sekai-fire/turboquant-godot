/**************************************************************************/
/*  llm_context.cpp                                                       */
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

#include "llm_context.h"

#include "core/error/error_macros.h"
#include "core/object/class_db.h"

LLMContext::~LLMContext() {
	destroy();
}

ggml_type LLMContext::_parse_cache_type(const String &p_name) {
	if (p_name == "turbo4") {
		return GGML_TYPE_TURBO4_0;
	}
	if (p_name == "turbo3") {
		return GGML_TYPE_TURBO3_0;
	}
	if (p_name == "turbo2") {
		return GGML_TYPE_TURBO2_0;
	}
	if (p_name == "q8_0") {
		return GGML_TYPE_Q8_0;
	}
	if (p_name == "q4_0") {
		return GGML_TYPE_Q4_0;
	}
	return GGML_TYPE_F16; // default / "f16"
}

void LLMContext::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_n_ctx", "n_ctx"), &LLMContext::set_n_ctx);
	ClassDB::bind_method(D_METHOD("get_n_ctx"), &LLMContext::get_n_ctx);
	ClassDB::bind_method(D_METHOD("set_n_threads", "threads"), &LLMContext::set_n_threads);
	ClassDB::bind_method(D_METHOD("get_n_threads"), &LLMContext::get_n_threads);
	ClassDB::bind_method(D_METHOD("set_flash_attn", "enabled"), &LLMContext::set_flash_attn);
	ClassDB::bind_method(D_METHOD("get_flash_attn"), &LLMContext::get_flash_attn);
	ClassDB::bind_method(D_METHOD("set_cache_type_k", "type"), &LLMContext::set_cache_type_k);
	ClassDB::bind_method(D_METHOD("get_cache_type_k"), &LLMContext::get_cache_type_k);
	ClassDB::bind_method(D_METHOD("set_cache_type_v", "type"), &LLMContext::set_cache_type_v);
	ClassDB::bind_method(D_METHOD("get_cache_type_v"), &LLMContext::get_cache_type_v);
	ClassDB::bind_method(D_METHOD("create", "model"), &LLMContext::create);
	ClassDB::bind_method(D_METHOD("destroy"), &LLMContext::destroy);
	ClassDB::bind_method(D_METHOD("is_valid"), &LLMContext::is_valid);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "n_ctx"), "set_n_ctx", "get_n_ctx");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "n_threads"), "set_n_threads", "get_n_threads");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flash_attn"), "set_flash_attn", "get_flash_attn");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "cache_type_k", PROPERTY_HINT_ENUM,
						 "f16,q8_0,q4_0,turbo2,turbo3,turbo4"),
			"set_cache_type_k", "get_cache_type_k");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "cache_type_v", PROPERTY_HINT_ENUM,
						 "f16,q8_0,q4_0,turbo2,turbo3,turbo4"),
			"set_cache_type_v", "get_cache_type_v");
}

void LLMContext::set_n_ctx(int p_n_ctx) {
	n_ctx = p_n_ctx;
}
int LLMContext::get_n_ctx() const {
	return n_ctx;
}
void LLMContext::set_n_threads(int p_threads) {
	n_threads = p_threads;
}
int LLMContext::get_n_threads() const {
	return n_threads;
}
void LLMContext::set_flash_attn(bool p_enabled) {
	flash_attn = p_enabled;
}
bool LLMContext::get_flash_attn() const {
	return flash_attn;
}
void LLMContext::set_cache_type_k(const String &p_type) {
	cache_type_k = p_type;
}
String LLMContext::get_cache_type_k() const {
	return cache_type_k;
}
void LLMContext::set_cache_type_v(const String &p_type) {
	cache_type_v = p_type;
}
String LLMContext::get_cache_type_v() const {
	return cache_type_v;
}

Error LLMContext::create(Ref<LLMModel> p_model) {
	ERR_FAIL_COND_V_MSG(p_model.is_null(), ERR_INVALID_PARAMETER, "LLMContext: model is null.");
	ERR_FAIL_COND_V_MSG(!p_model->is_loaded(), ERR_INVALID_PARAMETER, "LLMContext: model not loaded.");
	ERR_FAIL_COND_V_MSG(ctx != nullptr, ERR_ALREADY_IN_USE, "LLMContext: already created.");

	llama_context_params cparams = llama_context_default_params();
	cparams.n_ctx = n_ctx;
	cparams.n_threads = n_threads;
	cparams.flash_attn_type = flash_attn
			? LLAMA_FLASH_ATTN_TYPE_ENABLED
			: LLAMA_FLASH_ATTN_TYPE_DISABLED;
	cparams.type_k = _parse_cache_type(cache_type_k);
	cparams.type_v = _parse_cache_type(cache_type_v);

	ctx = llama_init_from_model(p_model->get_llama_model(), cparams);
	ERR_FAIL_COND_V_MSG(ctx == nullptr, FAILED, "LLMContext: llama_init_from_model failed.");

	return OK;
}

void LLMContext::destroy() {
	if (ctx != nullptr) {
		llama_free(ctx);
		ctx = nullptr;
	}
}

bool LLMContext::is_valid() const {
	return ctx != nullptr;
}
