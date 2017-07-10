--
-- Commands used to create the tables for flit in an sqlite3 database.
--

-- The foreign key support is off by default.  We want this functionality.
-- According to the documentation, it needs to be turned on every time at
-- runtime.
PRAGMA foreign_keys = ON;

--
-- Table: runs
--
-- Stores the id of the run and a message to identify which run it is as well
-- as the date and time of when it started.
--
CREATE TABLE IF NOT EXISTS runs (
  -- The run id used in tests table
  id             integer    primary key autoincrement     not null,

  -- timestamp is supported in python if you do the following:
  --   conn = sqlite3.connect("flit.sqlite",
  --                          detect_types=sqlite3.PARSE_DECLTYPES)
  -- The secret sauce is in the "detect_types" that allows python to intercept
  -- it and convert it to a sqlite3 basic type and back.
  rdate          timestamp,

  -- The message describing what this run is all about
  label          text
  );

--
-- Table: tests
--
-- Stores the information of compilation and results of FLiT tests as well as
-- timing information.
--
CREATE TABLE IF NOT EXISTS tests (
  id             integer    primary key autoincrement     not null,
  run            integer,   -- run index from runs table
  name           varchar,   -- name of the test case
  host           varchar,   -- name of computer that ran the test
  compiler       varchar,   -- compiler name
  optl           varchar,   -- optimization level (e.g. "-O2")
  switches       varchar,   -- compiler flag(s) (e.g. "-ffast-math")
  precision      varchar,   -- precision (f = float, d = double, e = extended)
  comparison     varchar,   -- metric of comparison - hex value
  comparison_d   real,      -- metric of comparison of result vs ground truth
  file           varchar,   -- filename of test executable
  nanosec        integer    check(nanosec >= 0),  -- timing for the function

  foreign key(run) references runs(id)
  );

-- Tables not created:
-- * clusters
-- * op_counts
-- * opcodes
-- * skipped_pin
-- * switch_conv
-- * switch_desc
-- Do we need these tables?  I don't know.
