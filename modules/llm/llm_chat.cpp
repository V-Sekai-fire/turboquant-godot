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

#include "core/crypto/crypto_core.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/object/class_db.h"
#include "core/variant/dictionary.h"

#include <thirdparty/llama_cpp/include/llama.h>

#include <cstring>
#include <string>
#include <vector>

void LLMChat::_bind_methods() {
	ClassDB::bind_method(D_METHOD("setup", "model", "context"), &LLMChat::setup);
	ClassDB::bind_method(D_METHOD("complete", "messages"), &LLMChat::complete);
	ClassDB::bind_method(D_METHOD("cancel"), &LLMChat::cancel);
	ClassDB::bind_method(D_METHOD("reset"), &LLMChat::reset);
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

	if (worker.is_started()) {
		worker.wait_to_finish();
	}

	abort_flag.store(false, std::memory_order_relaxed);

	Job *job = memnew(Job);
	job->messages = p_messages;
	job->self = this;
	busy = true;

	worker.start(_worker_func, job);
}

void LLMChat::cancel() {
	// Fire-and-forget exit signal. The worker checks abort_flag at each token
	// boundary and will clear busy before returning, after which reset() is safe.
	abort_flag.store(true, std::memory_order_relaxed);
}

void LLMChat::reset() {
	MutexLock lock(worker_mutex);
	ERR_FAIL_COND_MSG(busy, "LLMChat: cannot reset while inference is in progress.");
	if (context.is_valid() && context->is_valid()) {
		llama_memory_clear(llama_get_memory(context->get_llama_context()), true);
	}
	cached_tokens.clear();
}

void LLMChat::_worker_func(void *p_data) {
	Job *job = static_cast<Job *>(p_data);
	job->self->_run_inference(job->messages);
	memdelete(job);
}

bool LLMChat::_decode_media_ref(const String &p_url, PackedByteArray &r_bytes) {
	if (p_url.begins_with("data:")) {
		const int comma = p_url.find(",");
		if (comma < 0 || !p_url.substr(0, comma).contains("base64")) {
			return false;
		}
		const CharString b64 = p_url.substr(comma + 1).utf8();
		r_bytes.resize(b64.length()); // decoded output is always smaller
		size_t decoded = 0;
		if (CryptoCore::b64_decode(r_bytes.ptrw(), r_bytes.size(), &decoded,
					(const unsigned char *)b64.get_data(), b64.length()) != OK) {
			return false;
		}
		r_bytes.resize((int)decoded);
		return decoded > 0;
	}

	String path = p_url;
	if (path.begins_with("file://")) {
		path = path.substr(7);
	}
	if (path.begins_with("res://") || path.begins_with("user://") || FileAccess::exists(path)) {
		Error err = OK;
		r_bytes = FileAccess::get_file_as_bytes(path, &err);
		return err == OK && !r_bytes.is_empty();
	}
	return false;
}

// Gemma-4 renders turns as <|turn>role ... <turn|> with an explicit thought channel,
// nothing like ChatML. Detect it from the template's own markers rather than from a model
// name, so a retag or a different Gemma-4 quant still routes here.
static bool _is_turn_format(const char *p_tmpl) {
	return p_tmpl != nullptr && strstr(p_tmpl, "<|turn>") != nullptr;
}

// The shipped template is 21 KB because it also renders tool declarations, tool-call and
// tool-response channels, and replayed reasoning. LLMChat passes none of those -- only a
// role and a text body per message -- and for that input the template reduces to what is
// written out below. Reproducing that reduction directly keeps this tree free of a jinja
// engine, which cannot be built here anyway: llama.cpp's vendored one signals every type
// error by throwing, and Godot compiles without exceptions.
//
// Deliberately no bos_token: both llama_tokenize call sites pass add_special = true, so
// the tokenizer emits BOS itself and writing it here as text would produce two.
static String _format_turns(const Vector<String> &p_roles, const Vector<String> &p_texts, bool p_thinking) {
	int first = 0;
	const bool has_system = !p_roles.is_empty() && (p_roles[0] == "system" || p_roles[0] == "developer");

	String out;
	// A system turn is opened when thinking is on even with no system message, because that
	// is where the template puts the <|think|> switch.
	if (p_thinking || has_system) {
		out += "<|turn>system\n";
		if (p_thinking) {
			out += "<|think|>\n";
		}
		if (has_system) {
			out += p_texts[0].strip_edges();
			first = 1;
		}
		out += "<turn|>\n";
	}

	for (int i = first; i < p_roles.size(); i++) {
		const String role = (p_roles[i] == "assistant") ? String("model") : p_roles[i];
		const bool prev_was_model = (i > first) && (p_roles[i - 1] == "assistant");
		const bool next_is_model = (i + 1 < p_roles.size()) && (p_roles[i + 1] == "assistant");

		// Consecutive assistant messages are one continued model turn, not two turns.
		if (!(role == "model" && prev_was_model)) {
			out += "<|turn>" + role + "\n";
		}
		out += p_texts[i].strip_edges();
		out += (role == "model" && next_is_model) ? "\n" : "<turn|>\n";
	}

	out += "<|turn>model\n";
	// With thinking off the thought channel is opened and immediately closed, which is how
	// the template tells the model not to use it. Leaving it out invites the model to narrate
	// its reasoning into the reply.
	if (!p_thinking) {
		out += "<|channel>thought\n<channel|>";
	}
	return out;
}

String LLMChat::_apply_chat_template(const Array &p_messages, Vector<PackedByteArray> &r_media) const {
	// Build a basic ChatML prompt from the messages array.
	// "content" is either a String, or an OpenAI-style Array of parts:
	//   {"type":"text","text":...}
	//   {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
	//   {"type":"input_audio","input_audio":{"data":"<base64>"}}
	// Each decodable media part becomes a media marker in the prompt and its
	// bytes are appended to r_media; mtmd_tokenize() pairs them up by order.
#ifdef LLM_HAS_MTMD
	const char *marker = mtmd_default_marker();
#else
	// No projector on this platform, so a media part has nowhere to go. It is dropped
	// with a warning below rather than silently becoming empty text in the prompt.
	const char *marker = nullptr;
#endif
	Vector<String> roles;
	Vector<String> texts;
	for (int i = 0; i < p_messages.size(); i++) {
		Dictionary msg = p_messages[i];
		String role = msg.get("role", "user");
		Variant content = msg.get("content", "");
		String text;

		if (content.get_type() == Variant::ARRAY) {
			Array parts = content;
			for (int k = 0; k < parts.size(); k++) {
				if (parts[k].get_type() != Variant::DICTIONARY) {
					continue;
				}
				Dictionary part = parts[k];
				const String type = part.get("type", "");
				if (type == "text") {
					text += String(part.get("text", ""));
					continue;
				}

				String url;
				if (type == "image_url") {
					const Variant iv = part.get("image_url", Variant());
					url = (iv.get_type() == Variant::DICTIONARY) ? String(Dictionary(iv).get("url", "")) : String(iv);
				} else if (type == "input_audio") {
					const Variant av = part.get("input_audio", Variant());
					if (av.get_type() == Variant::DICTIONARY) {
						url = String(Dictionary(av).get("data", ""));
						if (!url.begins_with("data:")) {
							url = "data:audio/wav;base64," + url;
						}
					}
				}
				if (url.is_empty()) {
					continue;
				}

				PackedByteArray bytes;
				if (marker != nullptr && _decode_media_ref(url, bytes)) {
					r_media.push_back(bytes);
					text += String::utf8(marker);
				} else {
					WARN_PRINT("LLMChat: could not decode a media part; skipping it.");
				}
			}
		} else {
			text = content;
		}

		roles.push_back(role);
		texts.push_back(text);
	}
	const char *tmpl = llama_model_chat_template(model->get_llama_model(), nullptr);
	if (_is_turn_format(tmpl)) {
		return _format_turns(roles, texts, enable_thinking);
	}

	// ChatML. Correct for models that use it, and this is now a decision rather than the
	// unconditional assumption it used to be.
	String prompt;
	for (int i = 0; i < roles.size(); i++) {
		prompt += "<|im_start|>" + roles[i] + "\n" + texts[i] + "<|im_end|>\n";
	}
	prompt += "<|im_start|>assistant\n";
	return prompt;
}

void LLMChat::_run_inference(const Array &p_messages) {
	llama_context *lctx = context->get_llama_context();
	llama_model *lmodel = model->get_llama_model();
	const llama_vocab *vocab = llama_model_get_vocab(lmodel);
	const uint32_t n_ctx = llama_n_ctx(lctx);

	Vector<PackedByteArray> media;
	String prompt_str = _apply_chat_template(p_messages, media);
	std::string prompt_std = prompt_str.utf8().get_data();

	// ---- multimodal prefill -------------------------------------------------
	// When the request carries media and a projector is loaded, mtmd owns
	// tokenization and prefill: it interleaves text tokens with encoded image or
	// audio embeddings. The text-only prefix cache cannot be reused here, so the
	// KV cache is cleared first.
#ifdef LLM_HAS_MTMD
	mtmd_context *mctx = model->get_mtmd_context();
	if (!media.is_empty() && mctx != nullptr) {
		llama_memory_clear(llama_get_memory(lctx), true);
		cached_tokens.clear();

		std::vector<mtmd_bitmap *> bitmaps;
		bitmaps.reserve(media.size());
		bool decoded_all = true;
		for (int i = 0; i < media.size(); i++) {
			const PackedByteArray &b = media[i];
			mtmd_bitmap *bm = mtmd_helper_bitmap_init_from_buf(mctx, b.ptr(), (size_t)b.size());
			if (bm == nullptr) {
				decoded_all = false;
				break;
			}
			bitmaps.push_back(bm);
		}

		int32_t rc = -1;
		llama_pos new_n_past = 0;
		if (decoded_all) {
			mtmd_input_text txt;
			txt.text = prompt_std.c_str();
			txt.add_special = true;
			txt.parse_special = true;

			mtmd_input_chunks *chunks = mtmd_input_chunks_init();
			rc = mtmd_tokenize(mctx, chunks, &txt,
					(const mtmd_bitmap **)bitmaps.data(), bitmaps.size());
			if (rc == 0) {
				rc = mtmd_helper_eval_chunks(mctx, lctx, chunks,
						/*n_past*/ 0, /*seq_id*/ 0, /*n_batch*/ (int32_t)llama_n_batch(lctx),
						/*logits_last*/ true, &new_n_past);
			}
			mtmd_input_chunks_free(chunks);
		}

		for (mtmd_bitmap *bm : bitmaps) {
			mtmd_bitmap_free(bm);
		}

		if (!decoded_all || rc != 0) {
			const String why = !decoded_all
					? String("Could not decode media (unsupported format?).")
					: vformat("Multimodal prefill failed (mtmd error %d).", rc);
			call_deferred("emit_signal", "inference_failed", why);
			MutexLock lock(worker_mutex);
			busy = false;
			return;
		}

		_generate(lctx, vocab, (int)new_n_past);
		return;
	}
#endif
	// ---- text-only path continues below ------------------------------------

	if (!media.is_empty()) {
		WARN_PRINT("LLMChat: media supplied but no mmproj is loaded; ignoring it.");
	}

	// Tokenize full prompt.
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

	// If the prompt exceeds the context window, clear the cache and start fresh.
	if ((uint32_t)n_prompt >= n_ctx) {
		llama_memory_clear(llama_get_memory(lctx), true);
		cached_tokens.clear();
		call_deferred("emit_signal", "inference_failed", "Prompt exceeds context window.");
		MutexLock lock(worker_mutex);
		busy = false;
		return;
	}

	// Find the longest common prefix between what is already in the KV cache
	// and the new prompt, so we only prefill the novel suffix.
	int n_match = 0;
	int n_cached = (int)cached_tokens.size();
	while (n_match < n_cached && n_match < n_prompt &&
			cached_tokens[n_match] == prompt_tokens[n_match]) {
		n_match++;
	}

	// No token record means the cache holds something we cannot match against
	// (e.g. a previous multimodal turn) — drop it wholesale.
	if (n_cached == 0) {
		llama_memory_clear(llama_get_memory(lctx), true);
	}

	// Trim KV cache entries beyond the common prefix if the prompt diverged.
	if (n_match < n_cached) {
		llama_memory_seq_rm(llama_get_memory(lctx), 0, n_match, -1);
		cached_tokens.resize(n_match);
	}

	// Prefill only the new tokens.
	if (n_match < n_prompt) {
		llama_batch batch = llama_batch_get_one(prompt_tokens.data() + n_match, n_prompt - n_match);
		if (llama_decode(lctx, batch) != 0) {
			call_deferred("emit_signal", "inference_failed", "Prefill (llama_decode) failed.");
			MutexLock lock(worker_mutex);
			busy = false;
			return;
		}
	}
	cached_tokens = prompt_tokens;

	_generate(lctx, vocab, n_prompt);
}

void LLMChat::_generate(llama_context *lctx, const llama_vocab *vocab, int p_n_past) {
	// p_n_past is informational: llama tracks sequence position internally.
	(void)p_n_past;
	// Sampler setup.
	llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
	llama_sampler *sampler = llama_sampler_chain_init(sparams);
	llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, repeat_penalty, 0.0f, 0.0f));
	llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
	llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
	llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
	llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

	// Decode loop.
	String full_response;
	// Buffer incomplete UTF-8 byte sequences that may span token boundaries.
	PackedByteArray utf8_buf;

	// n_ctx was a local of _run_inference before the sampling loop moved here, and it
	// is not worth threading through a parameter: the context owns the number.
	const uint32_t n_ctx = llama_n_ctx(lctx);

	// max_tokens <= 0 means "no limit": generate until EOS or the context window
	// is full. The remaining space is n_ctx minus tokens already in the KV cache.
	const int token_budget = (max_tokens > 0)
			? max_tokens
			: (int)n_ctx - (int)cached_tokens.size();

	for (int i = 0; i < token_budget; i++) {
		// Reduction point: check for an exit signal before sampling the next token.
		if (abort_flag.load(std::memory_order_relaxed)) {
			break;
		}

		llama_token token_id = llama_sampler_sample(sampler, lctx, -1);
		llama_sampler_accept(sampler, token_id);

		// is_eog, not == eos. A model can end a turn with any of several tokens, and
		// Gemma-4 ends with <turn|> rather than the single id llama_vocab_eos reports --
		// so comparing against that one id let the turn marker through into the reply.
		if (llama_vocab_is_eog(vocab, token_id)) {
			break;
		}

		char piece_buf[256];
		int piece_len = llama_token_to_piece(vocab, token_id, piece_buf, sizeof(piece_buf), 0, true);
		if (piece_len < 0) {
			continue;
		}

		for (int j = 0; j < piece_len; j++) {
			utf8_buf.push_back((uint8_t)piece_buf[j]);
		}

		// Emit only complete UTF-8 sequences. A leading byte tells us the
		// expected sequence length; if the buffer doesn't hold the full
		// sequence yet, wait for the next token.
		int emit_up_to = 0;
		for (int j = 0; j < utf8_buf.size();) {
			uint8_t b = utf8_buf[j];
			int seq_len;
			if (b < 0x80) {
				seq_len = 1;
			} else if ((b & 0xE0) == 0xC0) {
				seq_len = 2;
			} else if ((b & 0xF0) == 0xE0) {
				seq_len = 3;
			} else if ((b & 0xF8) == 0xF0) {
				seq_len = 4;
			} else {
				// Invalid leading byte — skip it.
				j++;
				emit_up_to = j;
				continue;
			}
			if (j + seq_len <= utf8_buf.size()) {
				j += seq_len;
				emit_up_to = j;
			} else {
				break; // Incomplete sequence — wait for more bytes.
			}
		}

		if (emit_up_to > 0) {
			String token_str = String::utf8((const char *)utf8_buf.ptr(), emit_up_to);
			full_response += token_str;
			call_deferred("emit_signal", "token_generated", token_str);
			utf8_buf = utf8_buf.slice(emit_up_to, utf8_buf.size());
		}

		// Track the generated token in the cache, then decode it.
		cached_tokens.push_back(token_id);
		llama_batch next = llama_batch_get_one(&token_id, 1);
		if (llama_decode(lctx, next) != 0) {
			break;
		}
	}

	// Flush any remaining bytes (best-effort).
	if (!utf8_buf.is_empty()) {
		String remainder = String::utf8((const char *)utf8_buf.ptr(), utf8_buf.size());
		if (!remainder.is_empty()) {
			full_response += remainder;
			call_deferred("emit_signal", "token_generated", remainder);
		}
	}

	llama_sampler_free(sampler);
	// KV cache is intentionally kept for the next turn (prefix reuse).

	call_deferred("emit_signal", "response_received", full_response);

	MutexLock lock(worker_mutex);
	busy = false;
}

LLMChat::~LLMChat() {
	if (worker.is_started()) {
		worker.wait_to_finish();
	}
}
