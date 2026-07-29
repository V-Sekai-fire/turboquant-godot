/**************************************************************************/
/*  llm_chat.h                                                            */
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

#include "llm_context.h"
#include "llm_model.h"

#include "core/object/ref_counted.h"
#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "core/string/ustring.h"
#include "core/variant/array.h"

#include <thirdparty/llama_cpp/include/llama.h>

#include <atomic>
#include <vector>

// High-level chat completion using in-process llama.cpp inference.
// Runs on a worker thread; emits token_generated per token and
// response_received when complete.
//
// GDScript:
//   var chat = LLMChat.new()
//   chat.setup(model, context)
//   chat.token_generated.connect(func(tok): print(tok, end=""))
//   chat.response_received.connect(func(full): print(full))
//   chat.complete([{"role":"user","content":"Hello"}])
class LLMChat : public RefCounted {
	GDCLASS(LLMChat, RefCounted);

	Ref<LLMModel> model;
	Ref<LLMContext> context;

	int max_tokens = 1024;
	float temperature = 0.7f;
	float top_p = 0.9f;
	int top_k = 40;
	float repeat_penalty = 1.1f;
	bool enable_thinking = false;

	Thread worker;
	Mutex worker_mutex;
	bool busy = false;
	// Erlang-style exit signal: set by cancel(), checked at every token boundary.
	// The worker exits cleanly at the next reduction point rather than being killed.
	std::atomic<bool> abort_flag{ false };

	// Tokens currently prefilled in the KV cache. Persisted across turns so
	// each complete() only prefills the new suffix rather than the full prompt.
	std::vector<llama_token> cached_tokens;

	struct Job {
		Array messages;
		LLMChat *self = nullptr;
	};

	static void _worker_func(void *p_data);
	void _run_inference(const Array &p_messages);
	String _apply_chat_template(const Array &p_messages) const;

protected:
	static void _bind_methods();

public:
	void setup(Ref<LLMModel> p_model, Ref<LLMContext> p_context);

	void set_max_tokens(int p_v);
	int get_max_tokens() const;
	void set_temperature(float p_v);
	float get_temperature() const;
	void set_top_p(float p_v);
	float get_top_p() const;
	void set_top_k(int p_v);
	int get_top_k() const;
	void set_repeat_penalty(float p_v);
	float get_repeat_penalty() const;
	void set_enable_thinking(bool p_v);
	bool get_enable_thinking() const;

	bool is_busy() const;

	// Send an exit signal to the running inference worker (Erlang-style).
	// Returns immediately; the worker checks abort_flag at each token boundary
	// and exits at the next reduction point. After cancel() the worker will
	// clear busy, allowing reset() to be called safely.
	void cancel();

	// Clear KV cache and conversation prefix so the next complete() starts fresh.
	// Must not be called while busy (call cancel() first and wait for is_busy() == false).
	void reset();

	// Async. Emits token_generated(token) and response_received(text).
	void complete(const Array &p_messages);

	~LLMChat();
};
