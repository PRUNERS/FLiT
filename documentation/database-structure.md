# Database Structure

[Prev](benchmarks.md)
|
[Table of Contents](README.md)
|
[Next](analyze-results.md)

There are currently only two tables in the created SQLite3 database (although
there may be more added later).  The schema for this database can be found in
`data/db/tables-sqlite.sql`, or in the install located at
`<PREFIX>/share/flit/data/db/tables-sqlite.sql`.  Alternatively, you can see
the schema of the SQLite3 database by querying for it:

```sqlite3
$ sqlite3 results.sqlite
SQLite version 3.13.0 2016-05-18 10:57:30
Enter ".help" for usage hints.
sqlite> .tables
runs   tests
sqlite> .schema
CREATE TABLE runs (
  -- The run id used in tests table
  id             integer    primary key autoincrement     not null,

  -- timestamp is supported in python if you do the following:
  --   conn = sqlite3.connect("flit.sqlite",
  --                          detect_types=sqlite3.PARSE_DECLTYPES)
  -- The secret sauce is in the "detect_types" that allows python to intercept
  -- it and convert it to a sqlite3 basic type and back.
  rdate          timestamp,

  -- The label for the run describing what it is about
  label          text
  );
CREATE TABLE tests (
  id             integer    primary key autoincrement     not null,
  run            integer,   -- run index from runs table
  name           varchar,   -- name of the test case
  host           varchar,   -- name of computer that ran the test
  compiler       varchar,   -- compiler name
  optl           varchar,   -- optimization level (e.g. "-O2")
  switches       varchar,   -- compiler flag(s) (e.g. "-ffast-math")
  precision      varchar,   -- precision (f = float, d = double, e = extended)
  comparison_hex varchar,   -- metric of comparison - hex value
  comparison     real,      -- metric of comparison of result vs ground truth
  file           varchar,   -- filename of test executable
  nanosec        integer    check(nanosec >= 0),  -- timing for the function

  foreign key(run) references runs(id)
  );
sqlite> .quit
.quit
```

This output is as of this writing.  You can execute those same commands to see
the exact schema used in your version of FLiT.

The `runs` table only stores information about each executed full run, the id,
datetime and user-specified label for the run (called `label`).

The `tests` table contains the actual test results.  Each row has a run number
that matches the `id` field of the `runs` table, so you can do things like:

```sqlite3
select * from tests where run = 3;
```

Information about each compilation is stored there as well as the comparison
value.  The comparison value is what is returned from the `compare()` method in
your test class that compares a test result against the ground truth result.
Only this value is stored in the database for storage space reasons.  It also
contains timing information in the `nanosec` column which can be used to find
the fastest runtime execution.

[Prev](test-executable.md)
|
[Table of Contents](README.md)
|
[Next](analyze-results.md)
