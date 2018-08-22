#include "FlitCsv.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace flit {

std::string const& CsvRow::operator[](std::string col) const {
  if (m_header == nullptr) {
    throw std::logic_error("No header defined");
  }
  auto iter = std::find(m_header->begin(), m_header->end(), col);
  if (iter == m_header->end()) {
    std::stringstream message;
    message << "No column named " << col;
    throw std::invalid_argument(message.str());
  }
  auto idx = iter - m_header->begin();
  return this->at(idx);
}

CsvRow CsvReader::parseRow(std::istream &in) {
  enum class State {
    DEFAULT,
    IN_STRING,
  };
  State state = State::DEFAULT;

  char quote_char = '"';
  char separator = ',';
  char line_end = '\n';

  auto transition_state = [&state](State newstate) {
    //std::cout << "State transitioned from "
    //          << static_cast<int>(state)
    //          << " to "
    //          << static_cast<int>(newstate)
    //          << std::endl;
    state = newstate;
  };

  CsvRow row;
  char current;
  std::ostringstream running;
  int running_size = 0;
  while (in.get(current)) {
    if (state == State::DEFAULT) {
      if (running_size == 0 && current == quote_char) {
        transition_state(State::IN_STRING);
      } else if (current == separator) {
        row.emplace_back(running.str());
        running.str("");
        running_size = 0;
      } else if (current == line_end) {
        row.emplace_back(running.str());
        running.str("");
        running_size = 0;
        break; // break out of the while loop
      } else {
        running << current;
        running_size++;
      }
    } else if (state == State::IN_STRING) {
      if (current == quote_char) {
        transition_state(State::DEFAULT);
      } else {
        running << current;
        running_size++;
      }
    } else {
      throw std::runtime_error(
        "Please contact Michael Bentley, this shouldn't happen...");
    }
  }

  // We should not be within a STRING when we exit the while loop
  if (state != State::DEFAULT) {
    throw std::runtime_error("Error parsing CSV file");
  }

  // If we stopped because we reached the end of file...
  // (i.e. ignore empty last rows)
  if (!in && !row.empty()) {
    row.emplace_back(running.str());
  }

  return row;
}

} // end of namespace flit
