#include "FlitEventLogger.h"

#include <iostream>
#include <map>
#include <sstream> // for std::ostringstream
#include <stdexcept>
#include <string>

namespace flit {

FlitEventLogger *logger = nullptr;

FlitEventLogger::FlitEventLogger() : _out(nullptr) {
  if (logger != nullptr) {
    throw std::runtime_error("FlitEventLogger: can only make one");
  }
  logger = this;
}

void FlitEventLogger::log_event(
    const std::string &name,
    FlitEventLogger::EventType type,
    const std::map<std::string, std::string> &properties)
{
  // do nothing if we don't have anywhere to output
  if (_out == nullptr) {
    return;
  }

  switch (type) {
    case FlitEventLogger::START:
      // TODO: do something with it
      break;
    case FlitEventLogger::STOP:
      // TODO: do something with it
      break;
    default:
      throw std::runtime_error("unrecognized event type: "
                               + std::to_string(int(type)));
  }

  // TODO: implement
  throw std::runtime_error("unimplemented log_event()");

  std::ostringstream msg_builder;
  msg_builder << "{";
  bool first = true;
  for (const auto &kv : properties) {
    if (!first) {
      msg_builder << ", ";
    }
    first = false;
    msg_builder << '"' << kv.first << "\":\"" << kv.second << '"';
  }
  msg_builder << "}";
  msg_builder.str();
}

} // end of namespace flit
