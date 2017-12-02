#include "test_harness.h"

#include "flit.h"
#include "flit.cpp"

void tst_CsvRow_header() {
  CsvRow row {"1", "2", "3", "4"};
  TH_EQUAL(row.header(), nullptr);
  row.setHeader(&row);
  TH_EQUAL(row.header(), &row);
}
TH_REGISTER(tst_CsvRow_header);

void tst_CsvRow_operator_brackets() {
  CsvRow row {"1", "2", "3", "4"};
  CsvRow header {"a", "b", "c", "d"};
  row.setHeader(&header);
  TH_EQUAL(row["a"], "1");
  TH_EQUAL(row["b"], "2");
  TH_EQUAL(row["c"], "3");
  TH_EQUAL(row["d"], "4");
}
TH_REGISTER(tst_CsvRow_operator_brackets);







