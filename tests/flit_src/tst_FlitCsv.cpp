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

#include "test_harness.h"

#include <flit/FlitCsv.h>
#include <flit/FlitCsv.cpp>

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
TH_TEST(tst_CsvRow_header) {
  flit::CsvRow row {"1", "2", "3", "4"};
  TH_EQUAL(row.header(), nullptr);
  row.setHeader(&row);
  TH_EQUAL(row.header(), &row);
}

TH_TEST(tst_CsvRow_operator_brackets_string) {
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

TH_TEST(tst_CsvRow_operator_brackets_int) {
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
} // end of namespace tst_CsvRow

namespace tst_CsvReader {
TH_TEST(tst_CsvReader_oneRowAtATime) {
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
  TH_VERIFY(reader);

  reader >> row;
  expected_row = {};
  expected_row.setHeader(&expected_header);
  TH_VERIFY(row.empty());
  TH_EQUAL(row, expected_row);
  TH_VERIFY(!reader);
}

TH_TEST(tst_CsvReader_createRowVector) {
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

  std::vector<flit::CsvRow> expected_rows = {
    {"a", " b", "c", ""},
    {"1", "2", "3", "4", "5", "6", "7"},
    {""},
    {"hello", "there,my", "friends", "newline \nin quotes", ""},
  };
  for (auto &row : expected_rows) { row.setHeader(&expected_header); }
  decltype(expected_rows) rows;
  for (flit::CsvRow row; reader >> row;) { rows.emplace_back(row); }
  TH_EQUAL(rows, expected_rows);
}

} // end of namespace tst_CsvReader


namespace tst_CsvWriter {

/// Tests that the CsvWriter will end the file in a newline (if writing rows)
TH_TEST(tst_CsvWriter_write_row_addsNewline) {
  std::istringstream in(
      "first,second,third,fourth\n"   // header row
      "a, b,c,"
      );
  std::ostringstream out;

  flit::CsvReader reader(in);
  flit::CsvWriter writer(out);

  writer.write_row(*reader.header());
  flit::CsvRow row;
  while (reader >> row) { writer.write_row(row); }

  TH_EQUAL(out.str().back(), '\n');
  TH_EQUAL(in.str() + '\n', out.str());
}

/// Tests that CsvWriter can write out the exact same csv read in
TH_TEST(tst_CsvWriter_write_row_exactly) {
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
  flit::CsvRow row;
  while (reader >> row) { writer.write_row(row); }

  TH_EQUAL(in.str(), out.str());
}

} // end of namespace tst_CsvWriter

