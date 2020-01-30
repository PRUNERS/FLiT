/* -- LICENSE BEGIN --
 *
 * Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
 *   https://github.com/PRUNERS/FLiT/blob/master/LICENSE
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

#ifndef FLIT_CSV_H
#define FLIT_CSV_H

#include <flit/flitHelpers.h>

#include <istream>
#include <string>
#include <vector>

namespace flit {

/** Helper class for CsvReader.
 *
 * Represents a single row either indexed by number or by column name.
 */
class CsvRow : public std::vector<std::string> {
public:
  // Inherit base class constructors
  using std::vector<std::string>::vector;

  const CsvRow* header() const { return m_header; }
  void setHeader(CsvRow* head) { m_header = head; }

  using std::vector<std::string>::operator[];
  std::string const& operator[](std::string col) const;

  bool operator!=(const CsvRow &other) {
    // Check the header
    if (this->header() == nullptr && other.header() != nullptr) {
      return true;
    }
    if (this->header() != nullptr && other.header() == nullptr) {
      return true;
    }
    if (this->header() != nullptr &&
        other.header() != nullptr &&
        *this->header() != *other.header())
    {
      return true;
    }

    // Check row contents
    if (this->size() != other.size()) {
      return true;
    }
    for (CsvRow::size_type i = 0; i < this->size(); i++) {
      if (this->at(i) != other.at(i)) {
        return true;
      }
    }

    return false;
  }

  bool operator==(const CsvRow &other) { return !(*this != other); }

private:
  CsvRow* m_header {nullptr};  // not owned by this class
};

/** Class for parsing csv files */
class CsvReader {
public:
  CsvReader(std::istream &in)
    : m_header(CsvReader::parseRow(in)), m_in(in), m_is_done(!bool(in)) {}

  operator bool() const { return !m_is_done; }
  CsvRow* header() { return &m_header; }
  std::istream& stream() { return m_in; }

  CsvReader& operator>> (CsvRow& row) {
    row = CsvReader::parseRow(m_in);
    row.setHeader(this->header());
    if (row.empty()) {
      m_is_done = true;
    }
    return *this;
  }

private:
  static CsvRow parseRow(std::istream &in);

private:
  CsvRow m_header;
  std::istream &m_in;
  bool m_is_done;
};

class CsvWriter {
public:
  explicit CsvWriter(std::ostream &out)
    : m_out(out), m_is_line_beginning(true) {}
  virtual ~CsvWriter() {}

  void new_row() {
    this->m_out << std::endl;
    this->m_is_line_beginning = true;
  }

  template <typename T>
  void write_row (const std::vector<T>& row) {
    if (!this->m_is_line_beginning) {
      throw std::runtime_error("Cannot write a row to a partially created "
                               "row.  Call CsvWriter::new_row() first");
    }
    for (const T &elem : row) {
      *this << elem;
    }
    this->new_row();
  }

#define FLIT_CSVWRITER_OPERATOR(type) \
  CsvWriter& operator<< (const type val) { \
    this->append(val); \
    return *this; \
  }

  FLIT_CSVWRITER_OPERATOR(int);
  FLIT_CSVWRITER_OPERATOR(long);
  FLIT_CSVWRITER_OPERATOR(long long);
  FLIT_CSVWRITER_OPERATOR(unsigned int);
  FLIT_CSVWRITER_OPERATOR(unsigned long);
  FLIT_CSVWRITER_OPERATOR(unsigned long long);
  FLIT_CSVWRITER_OPERATOR(unsigned __int128);
  FLIT_CSVWRITER_OPERATOR(float);
  FLIT_CSVWRITER_OPERATOR(double);
  FLIT_CSVWRITER_OPERATOR(long double);

#undef FLIT_CSVWRITER_OPERATOR

  // handle std::string separately
  CsvWriter& operator<< (const std::string &val) {
    // if there are offending characters in val, then quote the field
    if (val.find_first_of(",\r\n") != std::string::npos) {
      this->append('"' + val + '"');
    } else {
      this->append(val);
    }
    return *this;
  }

private:
  template <typename T>
  void append(const T &val) {
    if (!this->m_is_line_beginning) {
      this->m_out << ',';
    }
    this->m_is_line_beginning = false;
    this->m_out << val;
  }

private:
  std::ostream &m_out;
  bool m_is_line_beginning;
};



} // end of namespace flit

#endif // FLIT_CSV_H
