// SPDX-License-Identifier: Apache-2.0
// Native C++ MLX Voxtral TTS server.
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/audio.h"
#include "common/http.h"
#include "common/supervisor.h"
#include "common/synthesis.h"
#include "common/tts_service.h"
#include "voxtral_tts/config.h"
#include "voxtral_tts/model.h"
#include "voxtral_tts/speech_tokenizer.h"

namespace {

constexpr int kMaximumQueuedRequests = 2;
constexpr const char *kModelKind = "voxtral_tts";
constexpr const char *kDefaultVoice = "cheerful_female";

struct ServerArguments {
  std::filesystem::path model_directory;
  std::string model_name;
  std::string host = "127.0.0.1";
  int port = 8000;
  bool supervised = false;
};

ServerArguments ParseArguments(int argument_count, char **argument_values) {
  ServerArguments arguments;
  for (int argument_index = 1; argument_index < argument_count;
       ++argument_index) {
    const std::string flag = argument_values[argument_index];
    const auto read_value = [&]() {
      if (argument_index + 1 >= argument_count) {
        throw std::invalid_argument(flag + " needs a value");
      } else {
      }
      return std::string(argument_values[++argument_index]);
    };
    if (flag == "--model-path" || flag == "--model-directory") {
      arguments.model_directory = read_value();
    } else if (flag == "--model-name") {
      arguments.model_name = read_value();
    } else if (flag == "--host") {
      arguments.host = read_value();
    } else if (flag == "--port") {
      arguments.port = std::stoi(read_value());
    } else if (flag == "--supervised") {
      arguments.supervised = true;
    } else if (flag == "--model-kind") {
      if (read_value() != kModelKind) {
        throw std::invalid_argument("only --model-kind voxtral_tts is served");
      } else {
      }
    } else if (flag == "--startup-timeout-s") {
      read_value();
    } else {
      throw std::invalid_argument("unknown argument " + flag);
    }
  }
  if (arguments.model_directory.empty()) {
    throw std::invalid_argument("--model-path is required");
  } else if (arguments.port < 0 || arguments.port > 65535) {
    throw std::invalid_argument("--port must be between 0 and 65535");
  } else {
  }
  if (arguments.model_name.empty()) {
    arguments.model_name =
        std::string("voxt-") + kModelKind + "-" + omni_server::RandomHex(12);
  } else {
  }
  return arguments;
}

} // namespace

int main(int argument_count, char **argument_values) {
  omni_server::BlockTerminationSignals();
  omni_server::IgnoreBrokenPipe();
  ServerArguments arguments;
  try {
    arguments = ParseArguments(argument_count, argument_values);
  } catch (const std::exception &error) {
    std::cerr << argument_values[0] << ": " << error.what() << "\n";
    return 2;
  }

  omni_server::StopSignal stop;
  std::thread(omni_server::WaitForTerminationSignal, std::ref(stop)).detach();
  if (arguments.supervised) {
    std::thread(omni_server::WatchShutdownCommands, std::ref(stop)).detach();
  } else {
  }

  const auto startup_started = std::chrono::steady_clock::now();
  std::unique_ptr<voxtral_tts::VoxtralConfig> config;
  std::unique_ptr<voxtral_tts::VoxtralModel> model;
  std::unique_ptr<voxtral_tts::SpeechTokenizer> tokenizer;
  std::map<std::string, mlx::core::array> voice_embeddings;
  try {
    config = std::make_unique<voxtral_tts::VoxtralConfig>(
        voxtral_tts::VoxtralConfig::Load(arguments.model_directory));
  } catch (const std::exception &error) {
    if (arguments.supervised) {
      omni_server::EmitControlEvent(
          {{"event", "failed"},
           {"reason", std::string("model load failed: ") + error.what()}});
    } else {
      std::cerr << "model load failed: " << error.what() << "\n";
    }
    return 1;
  }

  std::vector<std::string> voices;
  voices.reserve(config->voices.size());
  for (const auto &voice_entry : config->voices) {
    voices.push_back(voice_entry.first);
  }
  if (voices.empty()) {
    throw std::runtime_error("Voxtral checkpoint has no configured voices");
  } else {
  }
  const std::string default_voice =
      config->voices.contains(kDefaultVoice) ? kDefaultVoice : voices.front();
  omni_server::SynthesisWorker worker(
      [&model, &tokenizer,
       &voice_embeddings](const omni_server::SynthesisRequest &request,
                          const std::atomic<bool> &cancelled) {
        const std::vector<int> prompt_token_ids =
            tokenizer->EncodeSpeechRequest(request.text, request.voice);
        const voxtral_tts::GeneratedCodes codes = model->Generate(
            prompt_token_ids, voice_embeddings.at(request.voice),
            request.max_new_tokens, request.seed, cancelled);
        const std::vector<float> samples = model->DecodeAudio(codes.frames);
        return omni_server::SynthesisResult{
            qwen3_asr::EncodeWav(
                samples, model->config().audio_tokenizer.sampling_rate_hz),
            codes.finish_reason};
      },
      [&arguments, &config, &model, &tokenizer, &voice_embeddings]() {
        model = std::make_unique<voxtral_tts::VoxtralModel>(*config);
        tokenizer = std::make_unique<voxtral_tts::SpeechTokenizer>(
            arguments.model_directory);
        for (const auto &voice_entry : config->voices) {
          voice_embeddings.emplace(
              voice_entry.first,
              model->LoadVoiceEmbedding(voice_entry.first));
        }
      },
      kMaximumQueuedRequests);
  try {
    worker.WaitUntilReady();
  } catch (const std::exception &error) {
    if (arguments.supervised) {
      omni_server::EmitControlEvent(
          {{"event", "failed"},
           {"reason", std::string("model load failed: ") + error.what()}});
    } else {
      std::cerr << "model load failed: " << error.what() << "\n";
    }
    return 1;
  }

  omni_server::InfoRoutes info_routes(
      arguments.model_name, [&worker]() { return worker.RequestStates(); });
  omni_server::TTSService speech_service(
      arguments.model_name, default_voice,
      std::set<std::string>(voices.begin(), voices.end()), worker);
  const omni_server::HttpServer http_server =
      omni_server::StartHttpServer(arguments.host, arguments.port);
  if (http_server.context == nullptr) {
    const std::string endpoint =
        arguments.host + ":" + std::to_string(arguments.port);
    if (arguments.supervised) {
      omni_server::EmitControlEvent(
          {{"event", "failed"}, {"reason", "cannot listen on " + endpoint}});
    } else {
      std::cerr << "cannot listen on " << endpoint << "\n";
    }
    return 1;
  } else {
  }
  info_routes.Register(http_server.context);
  speech_service.Register(http_server.context);
  const double startup_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    startup_started)
          .count();
  if (arguments.supervised) {
    omni_server::EmitControlEvent(
        {{"event", "ready"},
         {"host", arguments.host},
         {"port", http_server.port},
         {"model_name", arguments.model_name},
         {"server_pid", static_cast<int>(getpid())},
         {"startup_s", std::round(startup_seconds * 1000) / 1000}});
  } else {
    std::cerr << "serving " << arguments.model_name << " on " << arguments.host
              << ":" << http_server.port << " after " << startup_seconds
              << " s\n";
  }

  const std::string stop_reason = stop.Wait();
  worker.CancelAll();
  mg_stop(http_server.context);
  mg_exit_library();
  if (stop_reason == "shutdown") {
    omni_server::EmitControlEvent({{"event", "stopped"}});
  } else {
  }
  return 0;
}
