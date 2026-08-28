/**************************************************************************/
/*  llm_server.cpp                                                        */
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

#include "llm_server.h"

#include "core/io/json.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "core/os/time.h"
#include "core/string/print_string.h"

namespace {

int env_int(const char *p_name, int p_default) {
	const String v = OS::get_singleton()->get_environment(p_name);
	return v.is_empty() ? p_default : v.to_int();
}

String env_str(const char *p_name, const String &p_default) {
	const String v = OS::get_singleton()->get_environment(p_name);
	return v.is_empty() ? p_default : v;
}

const char *status_text(int p_code) {
	switch (p_code) {
		case 200:
			return "OK";
		case 400:
			return "Bad Request";
		case 404:
			return "Not Found";
		case 500:
			return "Internal Server Error";
		case 503:
			return "Service Unavailable";
		default:
			return "Unknown";
	}
}

} // namespace

LLMServer::LLMServer() {}

LLMServer::~LLMServer() {
	for (uint32_t i = 0; i < clients.size(); i++) {
		memdelete(clients[i]);
	}
	clients.clear();
	if (rpc) {
		memdelete(rpc);
		rpc = nullptr;
	}
}

void LLMServer::_bind_methods() {}

void LLMServer::initialize() {
	const String model_path = env_str("LLM_MODEL", String());
	if (model_path.is_empty()) {
		ERR_PRINT("LLM_MODEL is not set — nothing to serve.");
		quit = true;
		return;
	}

	alias = env_str("LLM_ALIAS", "gemma-4-12b");
	host = env_str("LLM_HOST", "127.0.0.1");
	port = env_int("LLM_PORT", 8080);

	model.instantiate();
	model->set_model_path(model_path);
	model->set_n_gpu_layers(env_int("LLM_NGL", -1));
	const String mmproj = env_str("LLM_MMPROJ", String());
	if (!mmproj.is_empty()) {
		model->set_mmproj_path(mmproj);
	}

	context.instantiate();
	context->set_n_ctx(env_int("LLM_CTX", 32768));
	context->set_n_threads(env_int("LLM_THREADS", 8));
	context->set_cache_type_k(env_str("LLM_CTK", "q8_0"));
	context->set_cache_type_v(env_str("LLM_CTV", "turbo4"));
	context->set_flash_attn(true); // required by the turbo V codecs

	chat.instantiate();
	chat->connect("token_generated", callable_mp(this, &LLMServer::_on_token));
	chat->connect("response_received", callable_mp(this, &LLMServer::_on_response));
	chat->connect("inference_failed", callable_mp(this, &LLMServer::_on_failed));

	rpc = memnew(JSONRPC);

	server.instantiate();
	const Error err = server->listen(port, IPAddress(host));
	if (err != OK) {
		ERR_PRINT(vformat("Could not listen on %s:%d (error %d).", host, port, err));
		quit = true;
		return;
	}

	print_line(vformat("llm-server: listening on http://%s:%d", host, port));
	print_line(vformat("llm-server: model=%s ngl=%d ctx=%d k=%s v=%s",
			model_path.get_file(), model->get_n_gpu_layers(),
			context->get_n_ctx(), context->get_cache_type_k(), context->get_cache_type_v()));

	model->load(); // async; readiness is polled in process()
}

bool LLMServer::process(double p_delta) {
	if (quit) {
		return true;
	}

	// Bring the model and context up, in order, without blocking the loop.
	if (!model_ready) {
		if (model->is_loaded() && !context->is_valid() && !context->is_creating()) {
			context->create(model);
		} else if (context->is_valid()) {
			chat->setup(model, context);
			model_ready = true;
			print_line(vformat("llm-server: model ready (vision=%s audio=%s)",
					model->supports_vision() ? "yes" : "no",
					model->supports_audio() ? "yes" : "no"));
		}
	}

	_accept_new();

	for (uint32_t i = 0; i < clients.size();) {
		Client *c = clients[i];
		c->peer->poll();
		if (c->peer->get_status() != StreamPeerTCP::STATUS_CONNECTED || c->state == CLIENT_DONE) {
			// Never drop the request currently generating; let it finish first.
			if ((int)i == active && c->state != CLIENT_DONE) {
				i++;
				continue;
			}
			_drop(i);
			continue;
		}
		if (c->state == CLIENT_READING) {
			_poll_client(i);
		}
		i++;
	}

	_try_dispatch();
	return false;
}

void LLMServer::finalize() {
	if (server.is_valid()) {
		server->stop();
	}
}

void LLMServer::_accept_new() {
	while (server.is_valid() && server->is_connection_available()) {
		Client *c = memnew(Client);
		c->peer = server->take_connection();
		clients.push_back(c);
	}
}

void LLMServer::_drop(uint32_t p_index) {
	Client *c = clients[p_index];
	if ((int)p_index == active) {
		active = -1;
	} else if (active > (int)p_index) {
		active--;
	}
	if (c->peer.is_valid()) {
		c->peer->disconnect_from_host();
	}
	memdelete(c);
	clients.remove_at(p_index);
}

bool LLMServer::_request_complete(const Vector<uint8_t> &p_buf, int &r_header_end, int &r_content_length) {
	const String text = String::utf8((const char *)p_buf.ptr(), p_buf.size());
	const int end = text.find("\r\n\r\n");
	if (end < 0) {
		return false;
	}
	r_header_end = end + 4;
	r_content_length = 0;
	const Vector<String> lines = text.substr(0, end).split("\r\n");
	for (int i = 1; i < lines.size(); i++) {
		const String l = lines[i].to_lower();
		if (l.begins_with("content-length:")) {
			r_content_length = l.get_slice(":", 1).strip_edges().to_int();
		}
	}
	return p_buf.size() >= r_header_end + r_content_length;
}

void LLMServer::_poll_client(uint32_t p_index) {
	Client *c = clients[p_index];

	const int avail = c->peer->get_available_bytes();
	if (avail > 0) {
		Vector<uint8_t> tmp;
		tmp.resize(avail);
		int got = 0;
		if (c->peer->get_partial_data(tmp.ptrw(), avail, got) == OK && got > 0) {
			const int base = c->inbuf.size();
			c->inbuf.resize(base + got);
			memcpy(c->inbuf.ptrw() + base, tmp.ptr(), got);
		}
	}

	int header_end = 0;
	int content_length = 0;
	if (!_request_complete(c->inbuf, header_end, content_length)) {
		return;
	}

	const String raw = String::utf8((const char *)c->inbuf.ptr(), c->inbuf.size());
	const String request_line = raw.get_slice("\r\n", 0);
	const String method = request_line.get_slice(" ", 0);
	const String path = request_line.get_slice(" ", 1);
	const String body = raw.substr(header_end, content_length);

	_handle_request(p_index, method, path, body);
}

void LLMServer::_handle_request(uint32_t p_index, const String &p_method, const String &p_path, const String &p_body) {
	Client *c = clients[p_index];

	if (p_method == "GET" && p_path.begins_with("/health")) {
		Dictionary d;
		d["status"] = model_ready ? "ok" : "loading";
		_send_json(c, 200, d);
		c->state = CLIENT_DONE;
		return;
	}

	if (p_method == "GET" && p_path.begins_with("/v1/models")) {
		_send_json(c, 200, _models_payload());
		c->state = CLIENT_DONE;
		return;
	}

	if (p_method == "POST" && p_path.begins_with("/rpc")) {
		const String out = rpc->process_string(p_body);
		_send(c, 200, "application/json", out);
		c->state = CLIENT_DONE;
		return;
	}

	if (p_method == "POST" && p_path.begins_with("/v1/chat/completions")) {
		if (!model_ready) {
			Dictionary e;
			e["error"] = "model still loading";
			_send_json(c, 503, e);
			c->state = CLIENT_DONE;
			return;
		}

		const Variant parsed = JSON::parse_string(p_body);
		if (parsed.get_type() != Variant::DICTIONARY) {
			Dictionary e;
			e["error"] = "body must be a JSON object";
			_send_json(c, 400, e);
			c->state = CLIENT_DONE;
			return;
		}
		const Dictionary req = parsed;
		if (!req.has("messages")) {
			Dictionary e;
			e["error"] = "missing 'messages'";
			_send_json(c, 400, e);
			c->state = CLIENT_DONE;
			return;
		}

		c->pending_messages = req["messages"];
		c->stream = req.get("stream", false);
		c->id = "chatcmpl-" + itos(Time::get_singleton()->get_ticks_usec());
		c->queued_at = OS::get_singleton()->get_ticks_msec();
		c->state = CLIENT_WAITING;
		c->inbuf.clear();

		// Per-request sampling overrides; absent keys keep the current value.
		if (req.has("temperature")) {
			chat->set_temperature(req["temperature"]);
		}
		if (req.has("top_k")) {
			chat->set_top_k(req["top_k"]);
		}
		if (req.has("max_tokens")) {
			chat->set_max_tokens(req["max_tokens"]);
		}
		return; // dispatched by _try_dispatch()
	}

	Dictionary e;
	e["error"] = "not found";
	_send_json(c, 404, e);
	c->state = CLIENT_DONE;
}

bool LLMServer::_try_dispatch() {
	if (active >= 0 || !model_ready || chat->is_busy()) {
		return false;
	}
	for (uint32_t i = 0; i < clients.size(); i++) {
		Client *c = clients[i];
		if (c->state != CLIENT_WAITING) {
			continue;
		}
		// Re-read the body we stored on the client at parse time.
		active = (int)i;
		if (c->stream) {
			_begin_stream(c);
			c->state = CLIENT_STREAMING;
		}
		chat->complete(c->pending_messages);
		return true;
	}
	return false;
}

void LLMServer::_on_token(const String &p_token) {
	if (active < 0) {
		return;
	}
	Client *c = clients[active];
	if (c->state == CLIENT_STREAMING) {
		_send_chunk(c, p_token, false);
	}
}

void LLMServer::_on_response(const String &p_text) {
	if (active < 0) {
		return;
	}
	Client *c = clients[active];
	if (c->state == CLIENT_STREAMING) {
		_send_chunk(c, String(), true);
	} else {
		_send_json(c, 200, _completion_payload(c->id, p_text));
	}
	c->state = CLIENT_DONE;
	active = -1;
}

void LLMServer::_on_failed(const String &p_error) {
	if (active < 0) {
		return;
	}
	Client *c = clients[active];
	if (c->state == CLIENT_STREAMING) {
		_send_chunk(c, String(), true);
	} else {
		Dictionary e;
		e["error"] = p_error;
		_send_json(c, 500, e);
	}
	c->state = CLIENT_DONE;
	active = -1;
}

void LLMServer::_send(Client *p_c, int p_code, const String &p_content_type, const String &p_body) {
	const CharString payload = p_body.utf8();
	const String head = vformat(
			"HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %d\r\nConnection: close\r\n\r\n",
			p_code, status_text(p_code), p_content_type, payload.length());
	const CharString head_cs = head.utf8();
	p_c->peer->put_data((const uint8_t *)head_cs.get_data(), head_cs.length());
	if (payload.length() > 0) {
		p_c->peer->put_data((const uint8_t *)payload.get_data(), payload.length());
	}
}

void LLMServer::_send_json(Client *p_c, int p_code, const Dictionary &p_obj) {
	_send(p_c, p_code, "application/json", JSON::stringify(p_obj));
}

void LLMServer::_begin_stream(Client *p_c) {
	const String head =
			"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
			"Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
	const CharString cs = head.utf8();
	p_c->peer->put_data((const uint8_t *)cs.get_data(), cs.length());
	p_c->sent_header = true;
}

void LLMServer::_send_chunk(Client *p_c, const String &p_delta, bool p_final) {
	Dictionary delta;
	if (!p_final) {
		delta["content"] = p_delta;
	}

	Dictionary choice;
	choice["index"] = 0;
	choice["delta"] = delta;
	choice["finish_reason"] = p_final ? Variant("stop") : Variant();

	Array choices;
	choices.push_back(choice);

	Dictionary obj;
	obj["id"] = p_c->id;
	obj["object"] = "chat.completion.chunk";
	obj["created"] = (int64_t)Time::get_singleton()->get_unix_time_from_system();
	obj["model"] = alias;
	obj["choices"] = choices;

	String frame = "data: " + JSON::stringify(obj) + "\n\n";
	if (p_final) {
		frame += "data: [DONE]\n\n";
	}
	const CharString cs = frame.utf8();
	p_c->peer->put_data((const uint8_t *)cs.get_data(), cs.length());
}

Dictionary LLMServer::_completion_payload(const String &p_id, const String &p_text) const {
	Dictionary message;
	message["role"] = "assistant";
	message["content"] = p_text;

	Dictionary choice;
	choice["index"] = 0;
	choice["message"] = message;
	choice["finish_reason"] = "stop";

	Array choices;
	choices.push_back(choice);

	Dictionary obj;
	obj["id"] = p_id;
	obj["object"] = "chat.completion";
	obj["created"] = (int64_t)Time::get_singleton()->get_unix_time_from_system();
	obj["model"] = alias;
	obj["choices"] = choices;
	return obj;
}

Dictionary LLMServer::_models_payload() const {
	Dictionary m;
	m["id"] = alias;
	m["object"] = "model";
	m["created"] = (int64_t)Time::get_singleton()->get_unix_time_from_system();
	m["owned_by"] = "local";

	Array data;
	data.push_back(m);

	Dictionary obj;
	obj["object"] = "list";
	obj["data"] = data;
	return obj;
}
