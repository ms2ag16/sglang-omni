// SPDX-License-Identifier: Apache-2.0
#include "common/tts_service.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>

#include "common/http.h"

namespace omni_server {

namespace {

constexpr int kMaximumNewTokens = 4096;

} // namespace

TTSService::TTSService(std::string model_name, std::string default_voice,
                       std::set<std::string> voices, SynthesisWorker &worker)
    : model_name_(std::move(model_name)),
      default_voice_(std::move(default_voice)), voices_(std::move(voices)),
      worker_(worker) {}

void TTSService::Register(mg_context *context) {
  mg_set_request_handler(context, "/v1/audio/speech$", HandleSpeech, this);
}

int TTSService::HandleSpeech(mg_connection *connection, void *data) {
  auto *service = static_cast<TTSService *>(data);
  return service->Synthesize(connection);
}

int TTSService::Synthesize(mg_connection *connection) {
  if (!IsPost(connection)) {
    return WriteJson(connection, 405, {{"detail", "Method Not Allowed"}});
  } else {
  }
  Json body;
  try {
    body = Json::parse(ReadBody(connection));
  } catch (const Json::exception &) {
    return BadRequest(connection, "request body must be valid JSON");
  }
  if (!body.is_object()) {
    return BadRequest(connection, "request body must be a JSON object");
  } else {
  }
  const auto text_value = body.find("input");
  if (text_value == body.end() || !text_value->is_string()) {
    return BadRequest(connection, "input must be a string");
  } else {
  }
  const std::string text = text_value->get<std::string>();
  if (text.empty() ||
      std::all_of(text.begin(), text.end(), [](unsigned char character) {
        return std::isspace(character) != 0;
      })) {
    return BadRequest(connection,
                      "input must contain non-whitespace characters");
  } else {
  }
  const auto model_value = body.find("model");
  if (model_value != body.end() &&
      (!model_value->is_string() ||
       model_value->get<std::string>() != model_name_)) {
    return WriteJson(
        connection, 404,
        {{"detail", "Unknown model: " + (model_value->is_string()
                                             ? model_value->get<std::string>()
                                             : std::string())}});
  } else {
  }
  const auto voice_value = body.find("voice");
  if (voice_value != body.end() && !voice_value->is_string()) {
    return BadRequest(connection, "voice must be a string");
  } else {
  }
  const std::string voice = voice_value == body.end()
                                ? default_voice_
                                : voice_value->get<std::string>();
  if (!voices_.contains(voice)) {
    return BadRequest(connection, "unknown voice: " + voice);
  } else {
  }
  const auto format_value = body.find("response_format");
  if (format_value != body.end() &&
      (!format_value->is_string() ||
       format_value->get<std::string>() != "wav")) {
    return BadRequest(connection, "response_format must be wav");
  } else {
  }
  int max_new_tokens = kMaximumNewTokens;
  const auto max_tokens_value = body.find("max_new_tokens");
  if (max_tokens_value != body.end()) {
    if (!max_tokens_value->is_number_integer()) {
      return BadRequest(connection, "max_new_tokens must be an integer");
    } else {
    }
    max_new_tokens = max_tokens_value->get<int>();
    if (max_new_tokens < 1 || max_new_tokens > kMaximumNewTokens) {
      return BadRequest(connection,
                        "max_new_tokens is outside the supported range");
    } else {
    }
  } else {
  }
  uint64_t seed = std::mt19937_64(std::random_device{}())();
  const auto seed_value = body.find("seed");
  if (seed_value != body.end()) {
    if (!seed_value->is_number_unsigned() &&
        (!seed_value->is_number_integer() || seed_value->get<int64_t>() < 0)) {
      return BadRequest(connection, "seed must be a nonnegative integer");
    } else {
    }
    seed = seed_value->get<uint64_t>();
  } else {
  }

  std::future<SynthesisResult> result_future;
  try {
    result_future = worker_.Submit({text, voice, max_new_tokens, seed});
  } catch (const std::length_error &) {
    return WriteJson(connection, 429,
                     {{"detail", "TTS request queue is full"}});
  } catch (const std::exception &error) {
    std::cerr << "cannot enqueue TTS request: " << error.what() << "\n";
    return WriteJson(connection, 503,
                     {{"detail", "TTS service is unavailable"}});
  }
  try {
    const SynthesisResult result = result_future.get();
    const std::string finish_reason =
        result.finish_reason == "length" ? "length" : "stop";
    mg_printf(connection,
              "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\n"
              "Content-Disposition: attachment; filename=\"speech.wav\"\r\n"
              "X-Speech-Finish-Reason: %s\r\nContent-Length: %zu\r\n"
              "Connection: close\r\n\r\n",
              finish_reason.c_str(), result.wav_bytes.size());
    mg_write(connection, result.wav_bytes.data(), result.wav_bytes.size());
    return 200;
  } catch (const std::exception &error) {
    std::cerr << "TTS request failed: " << error.what() << "\n";
    return WriteJson(connection, 500,
                     {{"detail", "TTS speech generation failed"}});
  }
}

} // namespace omni_server
