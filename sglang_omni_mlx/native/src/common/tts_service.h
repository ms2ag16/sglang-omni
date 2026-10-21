// SPDX-License-Identifier: Apache-2.0
// Shared OpenAI-style TTS HTTP endpoint.
#pragma once

#include <set>
#include <string>

#include "civetweb.h"
#include "common/synthesis.h"

namespace omni_server {

class TTSService {
public:
  TTSService(std::string default_voice, std::set<std::string> voices,
             SynthesisWorker &worker);
  void Register(mg_context *context);

private:
  static int HandleSpeech(mg_connection *connection, void *data);
  int Synthesize(mg_connection *connection);

  std::string default_voice_;
  std::set<std::string> voices_;
  SynthesisWorker &worker_;
};

} // namespace omni_server
