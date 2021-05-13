#include <flit/FlitEventLogger.h>
#include <flit/flit.h>

#include <chrono>
#include <iomanip>   // for std::put_time()
#include <iostream>
#include <map>
#include <sstream> // for std::ostringstream
#include <stdexcept>
#include <string>

#include <cstdint> // for int64_t

namespace flit {

FlitEventLogger *logger = nullptr;

FlitEventLogger::FlitEventLogger() : _prog_name(), _out(nullptr) {
  if (logger != nullptr) {
    throw std::runtime_error("FlitEventLogger: can only make one");
  }
  logger = this;
}

void FlitEventLogger::log_event(
    const std::string &name,
    FlitEventLogger::EventType type,
    std::map<std::string, std::string> properties)
{
  // do nothing if we don't have anywhere to output
  if (_out == nullptr) {
    // TODO: Using this to toggle logging for now. Does this work well?
    //throw std::runtime_error("no outfile for logging");
    return;
  }

  if (type != START && type != STOP) {
    throw std::runtime_error("unrecognized event type: "
                             + std::to_string(int(type)));
  }

  // Add properties to all events.
  properties["Program Name"] = _prog_name;
  properties["Host"]         = FLIT_HOST;
  properties["Compiler"]     = FLIT_COMPILER;
  properties["Optl"]         = FLIT_OPTL;
  properties["Switches"]     = FLIT_SWITCHES;
  properties["Filename"]     = FLIT_FILENAME;

  // Convert properties to a json object
  std::ostringstream prop_builder;
  prop_builder << "{";
  bool first = true;
  for (const auto &kv : properties) {
    if (!first) {
      prop_builder << ", ";
    }
    first = false;
    prop_builder << '"' << kv.first << "\":\"" << kv.second << '"';
  }
  prop_builder << "}";
  std::string prop_json = prop_builder.str();

  // Setup clock
  std::chrono::duration<int64_t, std::nano> nanos_since_epoch(
      std::chrono::steady_clock::now().time_since_epoch());
  auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

  // Construct full event json object
  std::ostream &outref = *_out;
  outref << "{"
         << "\"date\":"        << "\""
           << std::put_time(std::localtime(&tt), "%a %d %b %Y %r %Z") << "\","
         << "\"time\":"        << nanos_since_epoch.count() << ","
         << "\"name\":\""      << name << "\","
         << "\"type\":"  << '"'
           << (type == START ? "start" : "stop")
           << "\", "
         << "\"properties\":"  << prop_json
         << "}\n";
}

} // end of namespace flit
