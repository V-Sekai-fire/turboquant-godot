/**************************************************************************/
/*  llm_context.h                                                         */
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

#include "llm_model.h"
#include <thirdparty/llama_cpp/ggml/include/ggml.h>
#include <thirdparty/llama_cpp/include/llama.h>

#include "core/object/ref_counted.h"
#include "core/string/ustring.h"

// Inference context wrapping llama_context*.
// KV cache types are set via string: "f16", "q8_0", "q4_0", "turbo2", "turbo3", "turbo4"
//
// GDScript:
//   var ctx = LLMContext.new()
//   ctx.n_ctx = 32768
//   ctx.cache_type_k = "q8_0"
//   ctx.cache_type_v = "turbo4"
//   ctx.create(model)
class LLMContext : public RefCounted {
	GDCLASS(LLMContext, RefCounted);

	int n_ctx = 32768;
	int n_threads = 4;
	bool flash_attn = true;
	String cache_type_k = "q8_0";
	String cache_type_v = "turbo4";

	llama_context *ctx = nullptr;

	static ggml_type _parse_cache_type(const String &p_name);

protected:
	static void _bind_methods();

public:
	void set_n_ctx(int p_n_ctx);
	int get_n_ctx() const;

	void set_n_threads(int p_threads);
	int get_n_threads() const;

	void set_flash_attn(bool p_enabled);
	bool get_flash_attn() const;

	// Valid values: "f16", "q8_0", "q4_0", "turbo2", "turbo3", "turbo4"
	void set_cache_type_k(const String &p_type);
	String get_cache_type_k() const;

	void set_cache_type_v(const String &p_type);
	String get_cache_type_v() const;

	Error create(Ref<LLMModel> p_model);
	void destroy();
	bool is_valid() const;

	llama_context *get_llama_context() const { return ctx; }

	~LLMContext();
};
