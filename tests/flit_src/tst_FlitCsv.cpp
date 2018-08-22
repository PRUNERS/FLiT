#include "test_harness.h"

#include "FlitCsv.h"

namespace {

std::ostream& operator<<(std::ostream& out, std::vector<std::string> vec) {
  bool first = true;
  out << "[";
  for (std::string &val : vec) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << '"' << val << '"';
  }
  out << "]";
  return out;
}

std::ostream& operator<<(std::ostream& out, flit::CsvRow& row) {
  out << "CsvRow:\n";
  if (row.header() == nullptr) {
    out << "  Header:  NULL\n";
  } else {
    out << "  Header:  "
        << static_cast<std::vector<std::string>>(*row.header())
        << std::endl;
  }
  out << "  Values:  " << static_cast<std::vector<std::string>>(row)
      << std::endl;
  return out;
}

} // end of unnamed namespace

namespace tst_CsvRow {
void tst_CsvRow_header() {
  flit::CsvRow row {"1", "2", "3", "4"};
  TH_EQUAL(row.header(), nullptr);
  row.setHeader(&row);
  TH_EQUAL(row.header(), &row);
}
TH_REGISTER(tst_CsvRow_header);

void tst_CsvRow_operator_brackets_string() {
  flit::CsvRow row {"1", "2", "3", "4"};
  flit::CsvRow header {"a", "b", "c", "d"};
  row.setHeader(&header);
  TH_EQUAL(row["a"], "1");
  TH_EQUAL(row["b"], "2");
  TH_EQUAL(row["c"], "3");
  TH_EQUAL(row["d"], "4");
  TH_THROWS(row["Mike"], std::invalid_argument);

  // Row missing elements
  header.emplace_back("e");
  TH_THROWS(row["e"], std::out_of_range);

  // null header
  row.setHeader(nullptr);
  TH_THROWS(row["a"], std::logic_error);
}
TH_REGISTER(tst_CsvRow_operator_brackets_string);

void tst_CsvRow_operator_brackets_int() {
  flit::CsvRow row {"1", "2", "3", "4"};
  flit::CsvRow header {"a", "b", "c", "d"};
  row.setHeader(&header);
  TH_EQUAL(row[0], "1");
  TH_EQUAL(row[1], "2");
  TH_EQUAL(row[2], "3");
  TH_EQUAL(row[3], "4");
  TH_THROWS(row.at(4), std::out_of_range);

  // Row missing elements
  header.emplace_back("e");
  TH_EQUAL(row.at(3), "4");
  TH_THROWS(row.at(4), std::out_of_range);

  // null header
  row.setHeader(nullptr);
  TH_EQUAL(row.at(2), "3");
  TH_THROWS(row.at(4), std::out_of_range);
}
TH_REGISTER(tst_CsvRow_operator_brackets_int);
} // end of namespace tst_CsvRow

namespace tst_CsvReader {
void tst_CsvReader_oneRowAtATime() {
  std::istringstream in(
      "first,second,third,fourth\n"   // header row
      "a, b,c,\n"
      "1,2,3,4,5,6,7\n"
      "\n"
      "hello,\"there,my\",friends,\"newline \n"
      "in quotes\","
      );

  flit::CsvReader reader(in);
  flit::CsvRow expected_header {"first", "second", "third", "fourth"};
  TH_EQUAL(*reader.header(), expected_header);

  flit::CsvRow row;
  reader >> row;
  flit::CsvRow expected_row;
  expected_row = {"a", " b", "c", ""};
  expected_row.setHeader(&expected_header);
  TH_EQUAL(row, expected_row);
  TH_VERIFY(reader);

  reader >> row;
  expected_row = {"1", "2", "3", "4", "5", "6", "7"};
  expected_row.setHeader(&expected_header);
  TH_EQUAL(row, expected_row);
  TH_VERIFY(reader);

  reader >> row;
  expected_row = {""};
  expected_row.setHeader(&expected_header);
  TH_EQUAL(row, expected_row);
  TH_VERIFY(reader);

  reader >> row;
  expected_row = {"hello", "there,my", "friends", "newline \nin quotes", ""};
  expected_row.setHeader(&expected_header);
  TH_EQUAL(row, expected_row);
  TH_VERIFY(!reader);

  reader >> row;
  expected_row = {};
  expected_row.setHeader(&expected_header);
  TH_VERIFY(row.empty());
  TH_EQUAL(row, expected_row);
}
TH_REGISTER(tst_CsvReader_oneRowAtATime);
} // end of namespace tst_CsvReader


namespace tst_CsvWriter {

/// Tests that the CsvWriter will end the file in a newline (if writing rows)
void tst_CsvWriter_write_row_addsNewline() {
  std::istringstream in(
      "first,second,third,fourth\n"   // header row
      "a, b,c,"
      );
  std::ostringstream out;

  flit::CsvReader reader(in);
  flit::CsvWriter writer(out);

  writer.write_row(*reader.header());
  while (reader) {
    flit::CsvRow row;
    reader >> row;
    if (row.empty()) {
      break;
    }
    writer.write_row(row);
  }

  TH_EQUAL(out.str().back(), '\n');
  TH_EQUAL(in.str() + '\n', out.str());

  // Try the range-based for loop
  std::istringstream in2(in.str());
  out.str("");
  flit::CsvReader reader2(in2);

  writer.write_row(*reader2.header());
  for (auto &row : reader2) { writer.write_row(row); }

  TH_EQUAL(out.str().back(), '\n');
  TH_EQUAL(in2.str() + '\n', out.str());
}
TH_REGISTER(tst_CsvWriter_write_row_addsNewline);

/// Tests that CsvWriter can write out the exact same csv read in
void tst_CsvWriter_write_row_exactly() {
  std::istringstream in(
      "first,second,third,fourth\n"   // header row
      "a, b,c,\n"
      "1,2,3,4,5,6,7\n"
      "\n"
      "hello,\"there,my\",friends,\"newline \n"
      "in quotes\",\n"
      );
  std::ostringstream out;

  flit::CsvReader reader(in);
  flit::CsvWriter writer(out);

  writer.write_row(*reader.header());
  while (reader) {
    flit::CsvRow row;
    reader >> row;
    if (row.empty()) {
      break;
    }
    writer.write_row(row);
  }

  TH_EQUAL(in.str(), out.str());

  // Try the range-based for loop
  std::istringstream in2(in.str());
  out.str("");
  flit::CsvReader reader2(in2);

  writer.write_row(*reader2.header());
  for (auto &row : reader2) { writer.write_row(row); }

  TH_EQUAL(in.str(), out.str());
}
TH_REGISTER(tst_CsvWriter_write_row_exactly);

} // end of namespace tst_CsvWriter

