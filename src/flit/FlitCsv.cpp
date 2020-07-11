/* -- LICENSE BEGIN --
 *
 * Copyright (c) 2015-2020, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * Written by
 *   Michael Bentley (mikebentley15@gmail.com),
 *   Geof Sawaya (fredricflinstone@gmail.com),
 *   and Ian Briggs (ian.briggs@utah.edu)
 * under the direction of
 *   Ganesh Gopalakrishnan
 *   and Dong H. Ahn.
 *
 * LLNL-CODE-743137
 *
 * All rights reserved.
 *
 * This file is part of FLiT. For details, see
 *   https://pruners.github.io/flit
 * Please also read
 *   https://github.com/PRUNERS/FLiT/blob/main/LICENSE
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the disclaimer below.
 *
 * - Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the disclaimer
 *   (as noted below) in the documentation and/or other materials
 *   provided with the distribution.
 *
 * - Neither the name of the LLNS/LLNL nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
 * SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Additional BSD Notice
 *
 * 1. This notice is required to be provided under our contract
 *    with the U.S. Department of Energy (DOE). This work was
 *    produced at Lawrence Livermore National Laboratory under
 *    Contract No. DE-AC52-07NA27344 with the DOE.
 *
 * 2. Neither the United States Government nor Lawrence Livermore
 *    National Security, LLC nor any of their employees, makes any
 *    warranty, express or implied, or assumes any liability or
 *    responsibility for the accuracy, completeness, or usefulness of
 *    any information, apparatus, product, or process disclosed, or
 *    represents that its use would not infringe privately-owned
 *    rights.
 *
 * 3. Also, reference herein to any specific commercial products,
 *    process, or services by trade name, trademark, manufacturer or
 *    otherwise does not necessarily constitute or imply its
 *    endorsement, recommendation, or favoring by the United States
 *    Government or Lawrence Livermore National Security, LLC. The
 *    views and opinions of authors expressed herein do not
 *    necessarily state or reflect those of the United States
 *    Government or Lawrence Livermore National Security, LLC, and
 *    shall not be used for advertising or product endorsement
 *    purposes.
 *
 * -- LICENSE END --
 */

#include <flit/FlitCsv.h>

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
