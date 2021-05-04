#ifndef FLIT_EVENT_LOGGER_H
#define FLIT_EVENT_LOGGER_H

#include <iostream>
#include <stdexcept>
#include <string>
#include <map>

namespace flit {

class FlitEventLogger;
extern FlitEventLogger *logger;

/** Event logging class (as a singleton)
 *
 * Intended to be created early on in the application and sets the flit::logger
 * singleton pointer.  Only one should be made.
 *
 * Example:
 *
 *   int main() {
 *     ...
 *     std::ofstream log_out("events.log");
 *     flit::FlitEventLogger logger(log_out);
 *     ...
 *   }
 *
 *   // called some time after logger is created in main()
 *   void some_function() {
 *     ...
 *     flit::logger->log_event("EventName", flit::FlitEventLogger::START,
 *                             {{"test", "mytest"},
 *                              {"compiler", "gcc"}});
 *     ...
 *   }
 */
class FlitEventLogger {
public:

  enum EventType {
    START = 1,
    STOP  = 2,
  };

  FlitEventLogger();

  /** Set where to log events */
  void set_stream(std::ostream &out) { _out = &out; }

  /** Log event to the logging out stream
   *
   * @param name: name of the event
   * @param type: START if this is the start of the event, else STOP 
   * @param properties: additional payload properties to encode in the event
   *     log
   */
  void log_event(const std::string &name,
                 EventType type,
                 std::map<std::string, std::string> properties);

private:
  std::ostream *_out = nullptr;

}; // end of class FlitEventLogger

} // end of namespace flit

#endif // FLIT_EVENT_LOGGER_H
