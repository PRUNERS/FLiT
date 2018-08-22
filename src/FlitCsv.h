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
  CsvReader(std::istream &in) : m_header(CsvReader::parseRow(in)), m_in(in) {}

  operator bool() const { return static_cast<bool>(m_in); }
  CsvRow* header() { return &m_header; }
  std::istream& stream() { return m_in; }

  CsvReader& operator>> (CsvRow& row) {
    row = CsvReader::parseRow(m_in);
    row.setHeader(this->header());
    return *this;
  }

  class Iterator {
  public:
    Iterator() : m_reader(nullptr) {}
    Iterator(CsvReader* reader) : m_reader(reader) { *reader >> row; }

    Iterator& operator++() {
      if (m_reader == nullptr) {
        throw std::out_of_range("Went beyond the CSV file");
      }
      *m_reader >> row;
      if (row.empty()) {
        m_reader = nullptr;  // mark the iterator as reaching the end
      }
      return *this;
    }

    bool operator != (const Iterator& other) const {
      return this->m_reader != other.m_reader;
    }

    CsvRow& operator*() { return row; }

  private:
    CsvReader *m_reader;
    CsvRow row;
  };

  CsvReader::Iterator begin() { return Iterator(this); };
  CsvReader::Iterator end() { return Iterator(); };

private:
  static CsvRow parseRow(std::istream &in);

private:
  CsvRow m_header;
  std::istream &m_in;
};

class CsvWriter {
public:
  explicit CsvWriter(std::ostream &out)
    : m_out(out), m_is_line_beginning(true) {}
  virtual ~CsvWriter() {}

  template <typename T>
  CsvWriter& operator<< (const T &val) {
    this->write_val(val);
    return *this;
  }

  template <typename T>
  void write_val (const T val) { append(val); }

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

private:
  template <typename T>
  CsvWriter& append(const T &val) {
    if (!this->m_is_line_beginning) {
      this->m_out << ',';
    }
    this->m_is_line_beginning = false;
    this->m_out << val;
    return *this;
  }

private:
  std::ostream &m_out;
  bool m_is_line_beginning;
};

#define CSVWRITER_OPERATOR(type) \
  inline CsvWriter& operator<< (CsvWriter& out, const type val) { \
    out.write_val(val); \
    return out; \
  }

CSVWRITER_OPERATOR(int);
CSVWRITER_OPERATOR(long);
CSVWRITER_OPERATOR(long long);
CSVWRITER_OPERATOR(unsigned int);
CSVWRITER_OPERATOR(unsigned long);
CSVWRITER_OPERATOR(unsigned long long);
CSVWRITER_OPERATOR(unsigned __int128);
CSVWRITER_OPERATOR(float);
CSVWRITER_OPERATOR(double);
CSVWRITER_OPERATOR(long double);

#undef CSVWRITER_OPERATOR

// handle std::string separately
inline CsvWriter& operator<< (CsvWriter& out, const std::string &val) {
  // if there are offending characters in val, then quote the field
  if (val.find_first_of(",\r\n") != std::string::npos) {
    out.write_val('"' + val + '"');
  } else {
    out.write_val(val);
  }
  return out;
}

} // end of namespace flit

#endif // FLIT_CSV_H
