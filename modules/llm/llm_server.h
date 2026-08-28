/**************************************************************************/
/*  llm_server.h                                                          */
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

#include "llm_chat.h"
#include "llm_context.h"
#include "llm_model.h"

#include "core/io/stream_peer_tcp.h"
#include "core/io/tcp_server.h"
#include "core/os/main_loop.h"
#include "core/templates/local_vector.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"

#include "modules/jsonrpc/jsonrpc.h"

// OpenAI-compatible HTTP front end for in-process llama.cpp inference.
//
// Runs as a MainLoop so it needs no SceneTree:
//
//   godot --headless --main-loop-type=LLMServer
//
// Configuration is read from the environment at initialize():
//   LLM_MODEL   (required)  path to the GGUF weights
//   LLM_MMPROJ              path to the mmproj GGUF (enables vision/audio)
//   LLM_HOST                bind address        (default 127.0.0.1)
//   LLM_PORT                listen port         (default 8080)
//   LLM_ALIAS               served model id     (default gemma-4-12b)
//   LLM_CTX                 context window      (default 32768)
//   LLM_CTK / LLM_CTV       KV cache types      (default q8_0 / turbo4)
//   LLM_NGL                 GPU layers          (default -1, all)
//   LLM_THREADS             CPU threads         (default 8)
//
// Routes:
//   GET  /health              -> {"status":"ok"}
//   GET  /v1/models           -> OpenAI model list
//   POST /v1/chat/completions -> completion, streaming when {"stream":true}
//   POST /rpc                 -> JSON-RPC surface (JSONRPC dispatch)
class LLMServer : public MainLoop {
	GDCLASS(LLMServer, MainLoop);

	enum ClientState {
		CLIENT_READING,
		CLIENT_WAITING, // queued behind an in-flight completion
		CLIENT_STREAMING,
		CLIENT_DONE,
	};

	struct Client {
		Ref<StreamPeerTCP> peer;
		Vector<uint8_t> inbuf;
		ClientState state = CLIENT_READING;
		bool stream = false;
		bool sent_header = false;
		Array pending_messages;
		String id;
		uint64_t queued_at = 0;
	};

	Ref<TCPServer> server;
	LocalVector<Client *> clients;

	Ref<LLMModel> model;
	Ref<LLMContext> context;
	Ref<LLMChat> chat;
	// JSONRPC derives from Object, not RefCounted, so it cannot be held in a Ref --
	// Ref<T> calls init_ref() and Object has no such member. Owned by hand instead.
	JSONRPC *rpc = nullptr;

	String alias = "gemma-4-12b";
	int port = 8080;
	String host = "127.0.0.1";

	// Index into `clients` of the request currently being generated, or -1.
	int active = -1;
	bool model_ready = false;
	bool quit = false;

	// --- plumbing ---
	void _accept_new();
	void _poll_client(uint32_t p_index);
	bool _try_dispatch(); // start the next queued completion, if idle
	void _drop(uint32_t p_index);

	// --- HTTP ---
	static bool _request_complete(const Vector<uint8_t> &p_buf, int &r_header_end, int &r_content_length);
	void _handle_request(uint32_t p_index, const String &p_method, const String &p_path, const String &p_body);
	void _send(Client *p_c, int p_code, const String &p_content_type, const String &p_body);
	void _send_json(Client *p_c, int p_code, const Dictionary &p_obj);
	void _begin_stream(Client *p_c);
	void _send_chunk(Client *p_c, const String &p_delta, bool p_final);

	// --- payloads ---
	Dictionary _completion_payload(const String &p_id, const String &p_text) const;
	Dictionary _models_payload() const;

	// --- LLMChat signal handlers (arrive on the main thread via call_deferred) ---
	void _on_token(const String &p_token);
	void _on_response(const String &p_text);
	void _on_failed(const String &p_error);

protected:
	static void _bind_methods();

public:
	virtual void initialize() override;
	virtual bool process(double p_delta) override;
	virtual void finalize() override;

	LLMServer();
	~LLMServer();
};
