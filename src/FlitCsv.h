#ifndef FLIT_CSV_H
#define FLIT_CSV_H

#include "flitHelpers.h"

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

private:
  CsvRow* m_header {nullptr};  // not owned by this class
};

/** Class for parsing csv files */
class CsvReader {
public:
  CsvReader(std::istream &in)
    : m_header(CsvReader::parseRow(in))
    , m_in(in)
  {
    m_header.setHeader(&m_header);
  }

  operator bool() const { return static_cast<bool>(m_in); }
  CsvRow header() { return m_header; }
  std::istream& stream() { return m_in; }

  CsvReader& operator>> (CsvRow& row) {
    row = CsvReader::parseRow(m_in);
    row.setHeader(&m_header);
    return *this;
  }

private:
  static CsvRow parseRow(std::istream &in);

private:
  CsvRow m_header;
  std::istream &m_in;
};

class CsvWriter {
public:
  CsvWriter(std::ostream &out) : m_out(out), m_is_line_beginning(true) {}

  template <typename T>
  CsvWriter& operator<< (const T &val) {
    this->write_val(val);
    return *this;
  }

  template <>
  CsvWriter& operator<< (const std::vector<std::string> &row) {
    this->write_row(row);
    return *this;
  }

  template <typename T>
  void write_val (const T val) {
    if (!this->m_is_line_beginning) {
      this->m_out << ',';
    }
    this->m_is_line_beginning = false;
    this->m_out << val;
  }

  template <>
  void write_val (const char* val) { write_val<std::string>(val); }
  
  template <>
  void write_val (const std::string &val) {
    if (!this->m_is_line_beginning) {
      this->m_out << ',';
    }
    this->m_is_line_beginning = false;
    // if there are offending characters in val, then quote the field
    if (val.find_first_of(",\r\n") != std::string::npos) {
      this->m_out << '"' << val << '"';
    } else {
      this->m_out << val;
    }
  }

  void new_row() {
    this->m_out << std::endl;
    this->m_is_line_beginning = true;
  }

  void write_row (const std::vector<std::string>& row) {
    if (!this->m_is_line_beginning) {
      throw std::runtime_error("Cannot write a row to a partially created "
                               "row.  Call CsvWriter::new_row() first");
    }
    for (auto &elem : row) {
      this->write_val(elem);
    }
    this->new_row();
  }

private:
  std::ostream &m_out;
  bool m_is_line_beginning;
};

} // end of namespace flit

#endif // FLIT_CSV_H
