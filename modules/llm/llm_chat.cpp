/**************************************************************************/
/*  llm_chat.cpp                                                          */
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

#include "llm_chat.h"

#include "core/error/error_macros.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/variant/dictionary.h"

#include "modules/llm/thirdparty/llama_cpp/common/common.h"
#include "modules/llm/thirdparty/llama_cpp/common/sampling.h"
#include "modules/llm/thirdparty/llama_cpp/include/llama.h"

void LLMChat::_bind_methods() {
	ClassDB::bind_method(D_METHOD("setup", "model", "context"), &LLMChat::setup);
	ClassDB::bind_method(D_METHOD("complete", "messages"), &LLMChat::complete);
	ClassDB::bind_method(D_METHOD("is_busy"), &LLMChat::is_busy);

	ClassDB::bind_method(D_METHOD("set_max_tokens", "v"), &LLMChat::set_max_tokens);
	ClassDB::bind_method(D_METHOD("get_max_tokens"), &LLMChat::get_max_tokens);
	ClassDB::bind_method(D_METHOD("set_temperature", "v"), &LLMChat::set_temperature);
	ClassDB::bind_method(D_METHOD("get_temperature"), &LLMChat::get_temperature);
	ClassDB::bind_method(D_METHOD("set_top_p", "v"), &LLMChat::set_top_p);
	ClassDB::bind_method(D_METHOD("get_top_p"), &LLMChat::get_top_p);
	ClassDB::bind_method(D_METHOD("set_top_k", "v"), &LLMChat::set_top_k);
	ClassDB::bind_method(D_METHOD("get_top_k"), &LLMChat::get_top_k);
	ClassDB::bind_method(D_METHOD("set_repeat_penalty", "v"), &LLMChat::set_repeat_penalty);
	ClassDB::bind_method(D_METHOD("get_repeat_penalty"), &LLMChat::get_repeat_penalty);
	ClassDB::bind_method(D_METHOD("set_enable_thinking", "v"), &LLMChat::set_enable_thinking);
	ClassDB::bind_method(D_METHOD("get_enable_thinking"), &LLMChat::get_enable_thinking);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_tokens"), "set_max_tokens", "get_max_tokens");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "temperature"), "set_temperature", "get_temperature");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "top_p"), "set_top_p", "get_top_p");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "top_k"), "set_top_k", "get_top_k");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "repeat_penalty"), "set_repeat_penalty", "get_repeat_penalty");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_thinking"), "set_enable_thinking", "get_enable_thinking");

	ADD_SIGNAL(MethodInfo("token_generated", PropertyInfo(Variant::STRING, "token")));
	ADD_SIGNAL(MethodInfo("response_received", PropertyInfo(Variant::STRING, "text")));
	ADD_SIGNAL(MethodInfo("inference_failed", PropertyInfo(Variant::STRING, "error")));
}

void LLMChat::setup(Ref<LLMModel> p_model, Ref<LLMContext> p_context) {
	model = p_model;
	context = p_context;
}

void LLMChat::set_max_tokens(int p_v) {
	max_tokens = p_v;
}
int LLMChat::get_max_tokens() const {
	return max_tokens;
}
void LLMChat::set_temperature(float p_v) {
	temperature = p_v;
}
float LLMChat::get_temperature() const {
	return temperature;
}
void LLMChat::set_top_p(float p_v) {
	top_p = p_v;
}
float LLMChat::get_top_p() const {
	return top_p;
}
void LLMChat::set_top_k(int p_v) {
	top_k = p_v;
}
int LLMChat::get_top_k() const {
	return top_k;
}
void LLMChat::set_repeat_penalty(float p_v) {
	repeat_penalty = p_v;
}
float LLMChat::get_repeat_penalty() const {
	return repeat_penalty;
}
void LLMChat::set_enable_thinking(bool p_v) {
	enable_thinking = p_v;
}
bool LLMChat::get_enable_thinking() const {
	return enable_thinking;
}
bool LLMChat::is_busy() const {
	return busy;
}

void LLMChat::complete(const Array &p_messages) {
	MutexLock lock(worker_mutex);
	ERR_FAIL_COND_MSG(busy, "LLMChat: inference already in progress.");
	ERR_FAIL_COND_MSG(model.is_null() || !model->is_loaded(), "LLMChat: model not loaded.");
	ERR_FAIL_COND_MSG(context.is_null() || !context->is_valid(), "LLMChat: context not valid.");

	Job *job = memnew(Job);
	job->messages = p_messages;
	job->self = this;
	busy = true;

	worker.start(_worker_func, job);
}

void LLMChat::_worker_func(void *p_data) {
	Job *job = static_cast<Job *>(p_data);
	job->self->_run_inference(job->messages);
	memdelete(job);
}

String LLMChat::_apply_chat_template(const Array &p_messages) const {
	// Build a basic ChatML prompt from the messages array.
	// Each element must be a Dictionary with "role" and "content".
	String prompt;
	for (int i = 0; i < p_messages.size(); i++) {
		Dictionary msg = p_messages[i];
		String role = msg.get("role", "user");
		String content = msg.get("content", "");
		prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
	}
	if (!enable_thinking) {
		prompt += "<|im_start|>assistant\n<think>\n</think>\n";
	} else {
		prompt += "<|im_start|>assistant\n";
	}
	return prompt;
}

void LLMChat::_run_inference(const Array &p_messages) {
	llama_context *lctx = context->get_llama_context();
	llama_model *lmodel = model->get_llama_model();
	const llama_vocab *vocab = llama_model_get_vocab(lmodel);

	String prompt_str = _apply_chat_template(p_messages);
	std::string prompt_std = prompt_str.utf8().get_data();

	// Tokenize prompt.
	std::vector<llama_token> prompt_tokens;
	prompt_tokens.resize(prompt_std.size() + 16);
	int n_prompt = llama_tokenize(vocab, prompt_std.c_str(), prompt_std.size(),
			prompt_tokens.data(), prompt_tokens.size(), true, true);
	if (n_prompt < 0) {
		call_deferred("emit_signal", "inference_failed", "Tokenization failed.");
		MutexLock lock(worker_mutex);
		busy = false;
		return;
	}
	prompt_tokens.resize(n_prompt);

	// Prefill.
	llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);
	if (llama_decode(lctx, batch) != 0) {
		call_deferred("emit_signal", "inference_failed", "Prefill (llama_decode) failed.");
		MutexLock lock(worker_mutex);
		busy = false;
		return;
	}

	// Sampler setup.
	llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
	llama_sampler *sampler = llama_sampler_chain_init(sparams);
	llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
	llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
	llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
	llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

	// Decode loop.
	String full_response;
	const llama_token eos = llama_vocab_eos(vocab);

	for (int i = 0; i < max_tokens; i++) {
		llama_token token_id = llama_sampler_sample(sampler, lctx, -1);
		llama_sampler_accept(sampler, token_id);

		if (token_id == eos) {
			break;
		}

		char piece_buf[256];
		int piece_len = llama_token_to_piece(vocab, token_id, piece_buf, sizeof(piece_buf), 0, true);
		if (piece_len < 0) {
			continue;
		}
		piece_buf[piece_len] = '\0';
		String token_str = String::utf8(piece_buf, piece_len);
		full_response += token_str;

		call_deferred("emit_signal", "token_generated", token_str);

		// Decode the new token.
		llama_batch next = llama_batch_get_one(&token_id, 1);
		if (llama_decode(lctx, next) != 0) {
			break;
		}
	}

	llama_sampler_free(sampler);
	llama_memory_clear(llama_get_memory(lctx), true);

	call_deferred("emit_signal", "response_received", full_response);

	MutexLock lock(worker_mutex);
	busy = false;
}

LLMChat::~LLMChat() {
	if (worker.is_started()) {
		worker.wait_to_finish();
	}
}
