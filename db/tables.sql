--
-- PostgreSQL database dump
--

-- Dumped from database version 9.5.5
-- Dumped by pg_dump version 9.5.5

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


--
-- Name: plpython3u; Type: PROCEDURAL LANGUAGE; Schema: -; Owner: -
--

CREATE OR REPLACE PROCEDURAL LANGUAGE plpython3u;


SET search_path = public, pg_catalog;

--
-- Name: breakdowntest(text, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION breakdowntest(name text, run integer) RETURNS integer
    LANGUAGE plpython3u
    AS $$

   quer = ("select distinct trunc(score0d, 15) as score, precision, " +
           "compiler, optl, array(select distinct switches from tests " +
           "where name = t1.name and score0 = t1.score0 and precision " +
           "= t1.precision and compiler = t1.compiler and run = t1.run " +
           "and optl = t1.optl)  from tests as t1 where name = '" +
           name + "' and run = " + str(run) + " order by score, compiler")
   res = plpy.execute(quer)
   return res.nrows()
$$;


--
-- Name: cleanupresults(integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION cleanupresults(run integer DEFAULT '-1'::integer) RETURNS integer[]
    LANGUAGE plpython3u
    AS $$
global run
rn = run
if rn == -1:
    r = ("SELECT MAX(index)as index from runs;")
    res = plpy.execute(r)
    rn = res[0]["index"]

s = ("update tests set compiler = 'icpc' where compiler ~ " +
     "'.*icpc.*' and run = " + str(rn))
res = plpy.execute(s)
s = ("update tests set host = 'kingspeak' where host ~ " +
     "'.*kingspeak.*' and run = " + str(rn))
res2 = plpy.execute(s)
s = ("update tests set switches=trim(switches)")
res3 = plpy.execute(s)
s = ("update tests set compiler=trim(compiler)")
res4 = plpy.execute(s)
s = ("update tests set compiler='clang++' where compiler='clang++-3.6'")
return [res.nrows(), res2.nrows(), res3.nrows(), res4.nrows()]
$$;


--
-- Name: dofullflitimport(text, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION dofullflitimport(path text, notes text) RETURNS integer[]
    LANGUAGE plpython3u
    AS $$
import datetime

query = ("INSERT INTO runs (rdate, notes) "
         "VALUES ('" + str(datetime.datetime.now())  +
         "','" + notes + "')")
plpy.execute(query)
query = ("SELECT MAX(index) from runs")
res = plpy.execute(query)
run = res[0]['max']
query = ("SELECT importflitresults2('" + path + "', " +
         str(run) + ")")
res = plpy.execute(query)
query = ("SELECT importopcoderesults('" + path + "/pins'," +
         str(run) + ")")
res2 = plpy.execute(query)

return [res[0]['importflitresults2'][0],res[0]['importflitresults2'][1],
        res2[0]['importopcoderesults'][0],res2[0]['importopcoderesults'][1]]

$$;


--
-- Name: importopcoderesults(text, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION importopcoderesults(path text, run integer) RETURNS integer[]
    LANGUAGE plpython3u
    AS $$
import glob
from plpy import spiexceptions
import os
count = 0
skipped = 0
for f in glob.iglob(path + '/*'):
    fels = os.path.basename(f).split('_')
    if len(fels) != 6:
        continue
    if fels[0] == 'INTEL':
        compiler = 'icpc'
    elif fels[0] == 'GCC':
        compiler = 'g++'
    elif fels[0] == 'CLANG':
        compiler = 'clang++'
    dynamic = False
    host = fels[1]
    flags = fels[2]
    optl = '-' + fels[3]
    precision = fels[4]
    name = fels[5]
    tq = ("SELECT index from tests where " +
          "name = '" + name + "' and " +
          "host = '" + host + "' and " +
          "precision = '" + precision + "' and " +
      "optl = '" + optl + "' and " +
          "compiler = '" + compiler + "' and " +
          "switches = (select switches from switch_conv where abbrev = '" + flags + "') and " +
          "run = " + str(run))
    res = plpy.execute(tq)
    if res.nrows() != 1:
        dup = res.nrows() > 1
        skq = ("insert into skipped_pin (name, host, precision, optl, " +
               "compiler, switches, run, dup)" +
               " select '" + name + "','" + host + "','" + precision + "','" +
           optl + "','" + compiler + "',switch_conv.switches," + str(run) +
           "," + str(dup) + " from switch_conv where abbrev = '" + flags + "'")
        plpy.execute(skq)
        skipped = skipped + 1
        continue
    tindx = res[0]["index"]
    with open(f) as inf:
        for line in inf:
            l = line.split()
            if len(line.lstrip()) > 0 and line.lstrip()[0] == '#':
                if 'dynamic' in line:
                    dynamic = True
                continue
            if len(l) < 4:
                continue
            opq = ("INSERT INTO opcodes VALUES(" +
                   str(l[0]) + ", '" + l[1] +"')")
            try:
                plpy.execute(opq)
            except spiexceptions.UniqueViolation:
                pass

            cntq = ("INSERT INTO op_counts (test_id, opcode, " +
                    "count, pred_count, dynamic) "+
                    "VALUES(" + str(tindx) + ", " + str(l[0]) +
                    ", " + str(l[2]) + ", " + str(l[3]) + ", " + str(dynamic) + ")")
            plpy.execute(cntq)
            count = count + 1
return [count, skipped]
$$;


--
-- Name: importflitresults(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION importflitresults(path text) RETURNS integer
    LANGUAGE plpython3u
    AS $$
   r = ("SELECT MAX(index)as index from runs;")
   res = plpy.execute(r)
   run = res[0]["index"]
   s = ("COPY tests " +
                "(host, switches, optl, compiler, precision, sort, " +
                "score0d, score0, score1d, score1, name, file) " +
                "FROM '" +
                path +
                "' (DELIMITER ',')")
   plpy.execute(s)
   s = ("UPDATE tests SET run = " + str(run) + " WHERE run IS NULL;")
   res = plpy.execute(s)
   return res.nrows()
$$;


--
-- Name: importflitresults2(text, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION importflitresults2(path text, run integer) RETURNS integer[]
    LANGUAGE plpython3u
    AS $$
import glob
from plpy import spiexceptions
import os
count = 0
skipped = 0
for f in glob.iglob(path + '/*_out_'):
    with open(f) as inf:
        for line in inf:
            elms = line.split(',')
            host = elms[0].strip()
            switches = elms[1].strip()
            optl = elms[2].strip()
            compiler = elms[3].strip()
            prec = elms[4].strip()
            sort = elms[5].strip()
            score0d = elms[6].strip()
            score0 = elms[7].strip()
            score1d = elms[8].strip()
            score1 = elms[9].strip()
            name = elms[10].strip()
            filen = elms[11].strip()
            quer = ("insert into tests "
                 "(host, switches, optl, compiler, precision, sort, "
                 "score0d, score0, score1d, score1, name, file, run) "
                 "VALUES ('" +
                 host + "','" +
                 switches + "','" +
                 optl + "','" +
                 compiler + "','" +
                 prec + "','" +
                 sort + "'," +
                 score0d + ",'" +
                 score0 + "'," +
                 score1d + ",'" +
                 score1 + "','" +
                 name + "','" +
                 filen + "'," +
                 str(run) + ")")
            try:
                plpy.execute(quer)
            except (spiexceptions.InvalidTextRepresentation,
                    spiexceptions.UndefinedColumn):
                quer = ("insert into tests "
                        "(host, switches, optl, compiler, precision, sort, "
                        "score0d, score0, score1d, score1, name, file, run) "
                        "VALUES ('" +
                        host + "','" +
                        switches + "','" +
                        optl + "','" +
                        compiler + "','" +
                        prec + "','" +
                        sort + "'," +
                        str(0) + ",'" +
                        score0 + "'," +
                        str(0) + ",'" +
                        score1 + "','" +
                        name + "','" +
                        filen + "'," +
                        str(run) + ")")
                #try:
                plpy.execute(quer)
                #except:
                #    skipped = skipped + 1
                #    continue
            count = count + 1
return [count, skipped]
$$;


--
-- Name: importswitches(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION importswitches(path text) RETURNS integer
    LANGUAGE plpython3u
    AS $$
with open(path) as inf:
    count = 0
    for line in inf:
        spc = line.find(' ')
        if spc == -1:
            abbrev = line
            swts = ''
        else:
            abbrev = line[0:spc]
            swts = line[spc+1:-1]
        q = ("INSERT INTO switch_conv VALUES " +
             "('" + abbrev + "', '" + swts + "')")
        plpy.execute(q)
        count = count + 1
return count
$$;


SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: clusters; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE clusters (
    testid integer NOT NULL,
    number integer
);


--
-- Name: op_counts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE op_counts (
    test_id integer NOT NULL,
    opcode integer NOT NULL,
    count integer,
    pred_count integer,
    dynamic boolean NOT NULL
);


--
-- Name: opcodes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE opcodes (
    index integer NOT NULL,
    name text
);


--
-- Name: runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE runs (
    index integer NOT NULL,
    rdate timestamp without time zone,
    notes text
);


--
-- Name: run_index_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE run_index_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: run_index_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE run_index_seq OWNED BY runs.index;


--
-- Name: skipped_pin; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE skipped_pin (
    switches character varying(512),
    "precision" character varying(1),
    sort character varying(2),
    score0 character varying(32),
    score0d numeric(1000,180),
    host character varying(50),
    compiler character varying(50),
    name character varying(255),
    index integer,
    score1 character varying(32),
    score1d numeric(1000,180),
    run integer,
    file character varying(512),
    optl character varying(10),
    dup boolean
);


--
-- Name: switch_conv; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE switch_conv (
    abbrev text NOT NULL,
    switches text
);


--
-- Name: tests; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE tests (
    switches character varying(512),
    "precision" character varying(1),
    sort character varying(2),
    score0 character varying(32),
    score0d numeric(1000,180),
    host character varying(50),
    compiler character varying(50),
    name character varying(255),
    index integer NOT NULL,
    score1 character varying(32),
    score1d numeric(1000,180),
    run integer,
    file character varying(512),
    optl character varying(10)
);


--
-- Name: tests_colname_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE tests_colname_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: tests_colname_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE tests_colname_seq OWNED BY tests.index;


--
-- Name: index; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY runs ALTER COLUMN index SET DEFAULT nextval('run_index_seq'::regclass);


--
-- Name: index; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY tests ALTER COLUMN index SET DEFAULT nextval('tests_colname_seq'::regclass);


--
-- Name: clusters_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY clusters
    ADD CONSTRAINT clusters_pkey PRIMARY KEY (testid);


--
-- Name: op_counts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY op_counts
    ADD CONSTRAINT op_counts_pkey PRIMARY KEY (test_id, opcode, dynamic);


--
-- Name: opcodes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY opcodes
    ADD CONSTRAINT opcodes_pkey PRIMARY KEY (index);


--
-- Name: runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY runs
    ADD CONSTRAINT runs_pkey PRIMARY KEY (index);


--
-- Name: switch_conv_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY switch_conv
    ADD CONSTRAINT switch_conv_pkey PRIMARY KEY (abbrev);


--
-- Name: tests_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY tests
    ADD CONSTRAINT tests_pkey PRIMARY KEY (index);


--
-- Name: op_counts_opcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY op_counts
    ADD CONSTRAINT op_counts_opcode_fkey FOREIGN KEY (opcode) REFERENCES opcodes(index);


--
-- Name: op_counts_test_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY op_counts
    ADD CONSTRAINT op_counts_test_id_fkey FOREIGN KEY (test_id) REFERENCES tests(index);


--
-- Name: tests_run_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY tests
    ADD CONSTRAINT tests_run_fkey FOREIGN KEY (run) REFERENCES runs(index);


--
-- PostgreSQL database dump complete
--

