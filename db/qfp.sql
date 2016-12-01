--
-- PostgreSQL database cluster dump
--

SET default_transaction_read_only = off;

SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;

--
-- Roles
--

CREATE ROLE ganesh;
ALTER ROLE ganesh WITH SUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION;
CREATE ROLE mbentley;
ALTER ROLE mbentley WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION PASSWORD 'md5f6c4c032e0a908ec4d99fbb07d2b01c1';
CREATE ROLE postgres;
ALTER ROLE postgres WITH SUPERUSER INHERIT CREATEROLE CREATEDB LOGIN REPLICATION PASSWORD 'md5c102ed37df8880351a99451cec0cc0e2';
CREATE ROLE qfp;
ALTER ROLE qfp WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION PASSWORD 'md55ef1f8b8031d047bdffd21c02b9933e6';
CREATE ROLE sawaya;
ALTER ROLE sawaya WITH SUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION;






--
-- Database creation
--

CREATE DATABASE gauss_wiki WITH TEMPLATE = template0 OWNER = postgres;
CREATE DATABASE qfp WITH TEMPLATE = template0 OWNER = sawaya;
REVOKE ALL ON DATABASE qfp FROM PUBLIC;
REVOKE ALL ON DATABASE qfp FROM sawaya;
GRANT ALL ON DATABASE qfp TO sawaya;
GRANT CONNECT,TEMPORARY ON DATABASE qfp TO PUBLIC;
GRANT ALL ON DATABASE qfp TO qfp;
GRANT ALL ON DATABASE qfp TO mbentley;
CREATE DATABASE qfp_pretrunc WITH TEMPLATE = template0 OWNER = sawaya;
REVOKE ALL ON DATABASE template1 FROM PUBLIC;
REVOKE ALL ON DATABASE template1 FROM postgres;
GRANT ALL ON DATABASE template1 TO postgres;
GRANT CONNECT ON DATABASE template1 TO PUBLIC;


\connect gauss_wiki

SET default_transaction_read_only = off;

--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: mediawiki; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA mediawiki;


ALTER SCHEMA mediawiki OWNER TO postgres;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = mediawiki, pg_catalog;

--
-- Name: media_type; Type: TYPE; Schema: mediawiki; Owner: postgres
--

CREATE TYPE media_type AS ENUM (
    'UNKNOWN',
    'BITMAP',
    'DRAWING',
    'AUDIO',
    'VIDEO',
    'MULTIMEDIA',
    'OFFICE',
    'TEXT',
    'EXECUTABLE',
    'ARCHIVE'
);


ALTER TYPE media_type OWNER TO postgres;

--
-- Name: add_interwiki(text, integer, smallint); Type: FUNCTION; Schema: mediawiki; Owner: postgres
--

CREATE FUNCTION add_interwiki(text, integer, smallint) RETURNS integer
    LANGUAGE sql
    AS $_$
 INSERT INTO interwiki (iw_prefix, iw_url, iw_local) VALUES ($1,$2,$3);
 SELECT 1;
 $_$;


ALTER FUNCTION mediawiki.add_interwiki(text, integer, smallint) OWNER TO postgres;

--
-- Name: page_deleted(); Type: FUNCTION; Schema: mediawiki; Owner: postgres
--

CREATE FUNCTION page_deleted() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
 BEGIN
 DELETE FROM recentchanges WHERE rc_namespace = OLD.page_namespace AND rc_title = OLD.page_title;
 RETURN NULL;
 END;
 $$;


ALTER FUNCTION mediawiki.page_deleted() OWNER TO postgres;

--
-- Name: ts2_page_text(); Type: FUNCTION; Schema: mediawiki; Owner: postgres
--

CREATE FUNCTION ts2_page_text() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
 BEGIN
 IF TG_OP = 'INSERT' THEN
 NEW.textvector = to_tsvector(NEW.old_text);
 ELSIF NEW.old_text != OLD.old_text THEN
 NEW.textvector := to_tsvector(NEW.old_text);
 END IF;
 RETURN NEW;
 END;
 $$;


ALTER FUNCTION mediawiki.ts2_page_text() OWNER TO postgres;

--
-- Name: ts2_page_title(); Type: FUNCTION; Schema: mediawiki; Owner: postgres
--

CREATE FUNCTION ts2_page_title() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
 BEGIN
 IF TG_OP = 'INSERT' THEN
 NEW.titlevector = to_tsvector(REPLACE(NEW.page_title,'/',' '));
 ELSIF NEW.page_title != OLD.page_title THEN
 NEW.titlevector := to_tsvector(REPLACE(NEW.page_title,'/',' '));
 END IF;
 RETURN NEW;
 END;
 $$;


ALTER FUNCTION mediawiki.ts2_page_title() OWNER TO postgres;

--
-- Name: archive_ar_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE archive_ar_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE archive_ar_id_seq OWNER TO postgres;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: archive; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE archive (
    ar_id integer DEFAULT nextval('archive_ar_id_seq'::regclass) NOT NULL,
    ar_namespace smallint NOT NULL,
    ar_title text NOT NULL,
    ar_text text,
    ar_page_id integer,
    ar_parent_id integer,
    ar_sha1 text DEFAULT ''::text NOT NULL,
    ar_comment text,
    ar_user integer,
    ar_user_text text NOT NULL,
    ar_timestamp timestamp with time zone NOT NULL,
    ar_minor_edit smallint DEFAULT 0 NOT NULL,
    ar_flags text,
    ar_rev_id integer,
    ar_text_id integer,
    ar_deleted smallint DEFAULT 0 NOT NULL,
    ar_len integer,
    ar_content_model text,
    ar_content_format text
);


ALTER TABLE archive OWNER TO postgres;

--
-- Name: category_cat_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE category_cat_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE category_cat_id_seq OWNER TO postgres;

--
-- Name: category; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE category (
    cat_id integer DEFAULT nextval('category_cat_id_seq'::regclass) NOT NULL,
    cat_title text NOT NULL,
    cat_pages integer DEFAULT 0 NOT NULL,
    cat_subcats integer DEFAULT 0 NOT NULL,
    cat_files integer DEFAULT 0 NOT NULL,
    cat_hidden smallint DEFAULT 0 NOT NULL
);


ALTER TABLE category OWNER TO postgres;

--
-- Name: categorylinks; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE categorylinks (
    cl_from integer NOT NULL,
    cl_to text NOT NULL,
    cl_sortkey text,
    cl_timestamp timestamp with time zone NOT NULL,
    cl_sortkey_prefix text DEFAULT ''::text NOT NULL,
    cl_collation text DEFAULT 0 NOT NULL,
    cl_type text DEFAULT 'page'::text NOT NULL
);


ALTER TABLE categorylinks OWNER TO postgres;

--
-- Name: change_tag; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE change_tag (
    ct_rc_id integer,
    ct_log_id integer,
    ct_rev_id integer,
    ct_tag text NOT NULL,
    ct_params text
);


ALTER TABLE change_tag OWNER TO postgres;

--
-- Name: externallinks_el_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE externallinks_el_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE externallinks_el_id_seq OWNER TO postgres;

--
-- Name: externallinks; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE externallinks (
    el_id integer DEFAULT nextval('externallinks_el_id_seq'::regclass) NOT NULL,
    el_from integer NOT NULL,
    el_to text NOT NULL,
    el_index text NOT NULL
);


ALTER TABLE externallinks OWNER TO postgres;

--
-- Name: filearchive_fa_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE filearchive_fa_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE filearchive_fa_id_seq OWNER TO postgres;

--
-- Name: filearchive; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE filearchive (
    fa_id integer DEFAULT nextval('filearchive_fa_id_seq'::regclass) NOT NULL,
    fa_name text NOT NULL,
    fa_archive_name text,
    fa_storage_group text,
    fa_storage_key text,
    fa_deleted_user integer,
    fa_deleted_timestamp timestamp with time zone NOT NULL,
    fa_deleted_reason text,
    fa_size integer NOT NULL,
    fa_width integer NOT NULL,
    fa_height integer NOT NULL,
    fa_metadata bytea DEFAULT '\x'::bytea NOT NULL,
    fa_bits smallint,
    fa_media_type text,
    fa_major_mime text DEFAULT 'unknown'::text,
    fa_minor_mime text DEFAULT 'unknown'::text,
    fa_description text NOT NULL,
    fa_user integer,
    fa_user_text text NOT NULL,
    fa_timestamp timestamp with time zone,
    fa_deleted smallint DEFAULT 0 NOT NULL,
    fa_sha1 text DEFAULT ''::text NOT NULL
);


ALTER TABLE filearchive OWNER TO postgres;

--
-- Name: image; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE image (
    img_name text NOT NULL,
    img_size integer NOT NULL,
    img_width integer NOT NULL,
    img_height integer NOT NULL,
    img_metadata bytea DEFAULT '\x'::bytea NOT NULL,
    img_bits smallint,
    img_media_type text,
    img_major_mime text DEFAULT 'unknown'::text,
    img_minor_mime text DEFAULT 'unknown'::text,
    img_description text NOT NULL,
    img_user integer,
    img_user_text text NOT NULL,
    img_timestamp timestamp with time zone,
    img_sha1 text DEFAULT ''::text NOT NULL
);


ALTER TABLE image OWNER TO postgres;

--
-- Name: imagelinks; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE imagelinks (
    il_from integer NOT NULL,
    il_from_namespace integer DEFAULT 0 NOT NULL,
    il_to text NOT NULL
);


ALTER TABLE imagelinks OWNER TO postgres;

--
-- Name: interwiki; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE interwiki (
    iw_prefix text NOT NULL,
    iw_url text NOT NULL,
    iw_local smallint NOT NULL,
    iw_trans smallint DEFAULT 0 NOT NULL,
    iw_api text DEFAULT ''::text NOT NULL,
    iw_wikiid text DEFAULT ''::text NOT NULL
);


ALTER TABLE interwiki OWNER TO postgres;

--
-- Name: ipblocks_ipb_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE ipblocks_ipb_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE ipblocks_ipb_id_seq OWNER TO postgres;

--
-- Name: ipblocks; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE ipblocks (
    ipb_id integer DEFAULT nextval('ipblocks_ipb_id_seq'::regclass) NOT NULL,
    ipb_address text,
    ipb_user integer,
    ipb_by integer NOT NULL,
    ipb_by_text text DEFAULT ''::text NOT NULL,
    ipb_reason text NOT NULL,
    ipb_timestamp timestamp with time zone NOT NULL,
    ipb_auto smallint DEFAULT 0 NOT NULL,
    ipb_anon_only smallint DEFAULT 0 NOT NULL,
    ipb_create_account smallint DEFAULT 1 NOT NULL,
    ipb_enable_autoblock smallint DEFAULT 1 NOT NULL,
    ipb_expiry timestamp with time zone NOT NULL,
    ipb_range_start text,
    ipb_range_end text,
    ipb_deleted smallint DEFAULT 0 NOT NULL,
    ipb_block_email smallint DEFAULT 0 NOT NULL,
    ipb_allow_usertalk smallint DEFAULT 0 NOT NULL,
    ipb_parent_block_id integer
);


ALTER TABLE ipblocks OWNER TO postgres;

--
-- Name: iwlinks; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE iwlinks (
    iwl_from integer DEFAULT 0 NOT NULL,
    iwl_prefix text DEFAULT ''::text NOT NULL,
    iwl_title text DEFAULT ''::text NOT NULL
);


ALTER TABLE iwlinks OWNER TO postgres;

--
-- Name: job_job_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE job_job_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE job_job_id_seq OWNER TO postgres;

--
-- Name: job; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE job (
    job_id integer DEFAULT nextval('job_job_id_seq'::regclass) NOT NULL,
    job_cmd text NOT NULL,
    job_namespace smallint NOT NULL,
    job_title text NOT NULL,
    job_timestamp timestamp with time zone,
    job_params text NOT NULL,
    job_random integer DEFAULT 0 NOT NULL,
    job_attempts integer DEFAULT 0 NOT NULL,
    job_token text DEFAULT ''::text NOT NULL,
    job_token_timestamp timestamp with time zone,
    job_sha1 text DEFAULT ''::text NOT NULL
);


ALTER TABLE job OWNER TO postgres;

--
-- Name: l10n_cache; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE l10n_cache (
    lc_lang text NOT NULL,
    lc_key text NOT NULL,
    lc_value bytea NOT NULL
);


ALTER TABLE l10n_cache OWNER TO postgres;

--
-- Name: langlinks; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE langlinks (
    ll_from integer NOT NULL,
    ll_lang text,
    ll_title text
);


ALTER TABLE langlinks OWNER TO postgres;

--
-- Name: log_search; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE log_search (
    ls_field text NOT NULL,
    ls_value text NOT NULL,
    ls_log_id integer DEFAULT 0 NOT NULL
);


ALTER TABLE log_search OWNER TO postgres;

--
-- Name: logging_log_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE logging_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE logging_log_id_seq OWNER TO postgres;

--
-- Name: logging; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE logging (
    log_id integer DEFAULT nextval('logging_log_id_seq'::regclass) NOT NULL,
    log_type text NOT NULL,
    log_action text NOT NULL,
    log_timestamp timestamp with time zone NOT NULL,
    log_user integer,
    log_namespace smallint NOT NULL,
    log_title text NOT NULL,
    log_comment text,
    log_params text,
    log_deleted smallint DEFAULT 0 NOT NULL,
    log_user_text text DEFAULT ''::text NOT NULL,
    log_page integer
);


ALTER TABLE logging OWNER TO postgres;

--
-- Name: module_deps; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE module_deps (
    md_module text NOT NULL,
    md_skin text NOT NULL,
    md_deps text NOT NULL
);


ALTER TABLE module_deps OWNER TO postgres;

--
-- Name: msg_resource; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE msg_resource (
    mr_resource text NOT NULL,
    mr_lang text NOT NULL,
    mr_blob text NOT NULL,
    mr_timestamp timestamp with time zone NOT NULL
);


ALTER TABLE msg_resource OWNER TO postgres;

--
-- Name: msg_resource_links; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE msg_resource_links (
    mrl_resource text NOT NULL,
    mrl_message text NOT NULL
);


ALTER TABLE msg_resource_links OWNER TO postgres;

--
-- Name: user_user_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE user_user_id_seq
    START WITH 0
    INCREMENT BY 1
    MINVALUE 0
    NO MAXVALUE
    CACHE 1;


ALTER TABLE user_user_id_seq OWNER TO postgres;

--
-- Name: mwuser; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE mwuser (
    user_id integer DEFAULT nextval('user_user_id_seq'::regclass) NOT NULL,
    user_name text NOT NULL,
    user_real_name text,
    user_password text,
    user_newpassword text,
    user_newpass_time timestamp with time zone,
    user_token text,
    user_email text,
    user_email_token text,
    user_email_token_expires timestamp with time zone,
    user_email_authenticated timestamp with time zone,
    user_touched timestamp with time zone,
    user_registration timestamp with time zone,
    user_editcount integer,
    user_password_expires timestamp with time zone
);


ALTER TABLE mwuser OWNER TO postgres;

--
-- Name: objectcache; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE objectcache (
    keyname text,
    value bytea DEFAULT '\x'::bytea NOT NULL,
    exptime timestamp with time zone NOT NULL
);


ALTER TABLE objectcache OWNER TO postgres;

--
-- Name: oldimage; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE oldimage (
    oi_name text NOT NULL,
    oi_archive_name text NOT NULL,
    oi_size integer NOT NULL,
    oi_width integer NOT NULL,
    oi_height integer NOT NULL,
    oi_bits smallint,
    oi_description text,
    oi_user integer,
    oi_user_text text NOT NULL,
    oi_timestamp timestamp with time zone,
    oi_metadata bytea DEFAULT '\x'::bytea NOT NULL,
    oi_media_type text,
    oi_major_mime text DEFAULT 'unknown'::text,
    oi_minor_mime text DEFAULT 'unknown'::text,
    oi_deleted smallint DEFAULT 0 NOT NULL,
    oi_sha1 text DEFAULT ''::text NOT NULL
);


ALTER TABLE oldimage OWNER TO postgres;

--
-- Name: page_page_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE page_page_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE page_page_id_seq OWNER TO postgres;

--
-- Name: page; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE page (
    page_id integer DEFAULT nextval('page_page_id_seq'::regclass) NOT NULL,
    page_namespace smallint NOT NULL,
    page_title text NOT NULL,
    page_restrictions text,
    page_is_redirect smallint DEFAULT 0 NOT NULL,
    page_is_new smallint DEFAULT 0 NOT NULL,
    page_random numeric(15,14) DEFAULT random() NOT NULL,
    page_touched timestamp with time zone,
    page_links_updated timestamp with time zone,
    page_latest integer NOT NULL,
    page_len integer NOT NULL,
    page_content_model text,
    page_lang text,
    titlevector tsvector
);


ALTER TABLE page OWNER TO postgres;

--
-- Name: page_props; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE page_props (
    pp_page integer NOT NULL,
    pp_propname text NOT NULL,
    pp_value text NOT NULL,
    pp_sortkey double precision
);


ALTER TABLE page_props OWNER TO postgres;

--
-- Name: page_restrictions_pr_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE page_restrictions_pr_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE page_restrictions_pr_id_seq OWNER TO postgres;

--
-- Name: page_restrictions; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE page_restrictions (
    pr_id integer DEFAULT nextval('page_restrictions_pr_id_seq'::regclass) NOT NULL,
    pr_page integer NOT NULL,
    pr_type text NOT NULL,
    pr_level text NOT NULL,
    pr_cascade smallint NOT NULL,
    pr_user integer,
    pr_expiry timestamp with time zone
);


ALTER TABLE page_restrictions OWNER TO postgres;

--
-- Name: text_old_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE text_old_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE text_old_id_seq OWNER TO postgres;

--
-- Name: pagecontent; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE pagecontent (
    old_id integer DEFAULT nextval('text_old_id_seq'::regclass) NOT NULL,
    old_text text,
    old_flags text,
    textvector tsvector
);


ALTER TABLE pagecontent OWNER TO postgres;

--
-- Name: pagelinks; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE pagelinks (
    pl_from integer NOT NULL,
    pl_from_namespace integer DEFAULT 0 NOT NULL,
    pl_namespace smallint NOT NULL,
    pl_title text NOT NULL
);


ALTER TABLE pagelinks OWNER TO postgres;

--
-- Name: profiling; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE profiling (
    pf_count integer DEFAULT 0 NOT NULL,
    pf_time double precision DEFAULT 0 NOT NULL,
    pf_memory double precision DEFAULT 0 NOT NULL,
    pf_name text NOT NULL,
    pf_server text
);


ALTER TABLE profiling OWNER TO postgres;

--
-- Name: protected_titles; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE protected_titles (
    pt_namespace smallint NOT NULL,
    pt_title text NOT NULL,
    pt_user integer,
    pt_reason text,
    pt_timestamp timestamp with time zone NOT NULL,
    pt_expiry timestamp with time zone,
    pt_create_perm text DEFAULT ''::text NOT NULL
);


ALTER TABLE protected_titles OWNER TO postgres;

--
-- Name: querycache; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE querycache (
    qc_type text NOT NULL,
    qc_value integer NOT NULL,
    qc_namespace smallint NOT NULL,
    qc_title text NOT NULL
);


ALTER TABLE querycache OWNER TO postgres;

--
-- Name: querycache_info; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE querycache_info (
    qci_type text,
    qci_timestamp timestamp with time zone
);


ALTER TABLE querycache_info OWNER TO postgres;

--
-- Name: querycachetwo; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE querycachetwo (
    qcc_type text NOT NULL,
    qcc_value integer DEFAULT 0 NOT NULL,
    qcc_namespace integer DEFAULT 0 NOT NULL,
    qcc_title text DEFAULT ''::text NOT NULL,
    qcc_namespacetwo integer DEFAULT 0 NOT NULL,
    qcc_titletwo text DEFAULT ''::text NOT NULL
);


ALTER TABLE querycachetwo OWNER TO postgres;

--
-- Name: recentchanges_rc_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE recentchanges_rc_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE recentchanges_rc_id_seq OWNER TO postgres;

--
-- Name: recentchanges; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE recentchanges (
    rc_id integer DEFAULT nextval('recentchanges_rc_id_seq'::regclass) NOT NULL,
    rc_timestamp timestamp with time zone NOT NULL,
    rc_cur_time timestamp with time zone,
    rc_user integer,
    rc_user_text text NOT NULL,
    rc_namespace smallint NOT NULL,
    rc_title text NOT NULL,
    rc_comment text,
    rc_minor smallint DEFAULT 0 NOT NULL,
    rc_bot smallint DEFAULT 0 NOT NULL,
    rc_new smallint DEFAULT 0 NOT NULL,
    rc_cur_id integer,
    rc_this_oldid integer NOT NULL,
    rc_last_oldid integer NOT NULL,
    rc_type smallint DEFAULT 0 NOT NULL,
    rc_source text NOT NULL,
    rc_patrolled smallint DEFAULT 0 NOT NULL,
    rc_ip cidr,
    rc_old_len integer,
    rc_new_len integer,
    rc_deleted smallint DEFAULT 0 NOT NULL,
    rc_logid integer DEFAULT 0 NOT NULL,
    rc_log_type text,
    rc_log_action text,
    rc_params text
);


ALTER TABLE recentchanges OWNER TO postgres;

--
-- Name: redirect; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE redirect (
    rd_from integer NOT NULL,
    rd_namespace smallint NOT NULL,
    rd_title text NOT NULL,
    rd_interwiki text,
    rd_fragment text
);


ALTER TABLE redirect OWNER TO postgres;

--
-- Name: revision_rev_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE revision_rev_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE revision_rev_id_seq OWNER TO postgres;

--
-- Name: revision; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE revision (
    rev_id integer DEFAULT nextval('revision_rev_id_seq'::regclass) NOT NULL,
    rev_page integer,
    rev_text_id integer,
    rev_comment text,
    rev_user integer NOT NULL,
    rev_user_text text NOT NULL,
    rev_timestamp timestamp with time zone NOT NULL,
    rev_minor_edit smallint DEFAULT 0 NOT NULL,
    rev_deleted smallint DEFAULT 0 NOT NULL,
    rev_len integer,
    rev_parent_id integer,
    rev_sha1 text DEFAULT ''::text NOT NULL,
    rev_content_model text,
    rev_content_format text
);


ALTER TABLE revision OWNER TO postgres;

--
-- Name: site_identifiers; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE site_identifiers (
    si_site integer NOT NULL,
    si_type text NOT NULL,
    si_key text NOT NULL
);


ALTER TABLE site_identifiers OWNER TO postgres;

--
-- Name: site_stats; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE site_stats (
    ss_row_id integer NOT NULL,
    ss_total_edits integer DEFAULT 0,
    ss_good_articles integer DEFAULT 0,
    ss_total_pages integer DEFAULT (-1),
    ss_users integer DEFAULT (-1),
    ss_active_users integer DEFAULT (-1),
    ss_admins integer DEFAULT (-1),
    ss_images integer DEFAULT 0
);


ALTER TABLE site_stats OWNER TO postgres;

--
-- Name: sites_site_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE sites_site_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE sites_site_id_seq OWNER TO postgres;

--
-- Name: sites; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE sites (
    site_id integer DEFAULT nextval('sites_site_id_seq'::regclass) NOT NULL,
    site_global_key text NOT NULL,
    site_type text NOT NULL,
    site_group text NOT NULL,
    site_source text NOT NULL,
    site_language text NOT NULL,
    site_protocol text NOT NULL,
    site_domain text NOT NULL,
    site_data text NOT NULL,
    site_forward smallint NOT NULL,
    site_config text NOT NULL
);


ALTER TABLE sites OWNER TO postgres;

--
-- Name: tag_summary; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE tag_summary (
    ts_rc_id integer,
    ts_log_id integer,
    ts_rev_id integer,
    ts_tags text NOT NULL
);


ALTER TABLE tag_summary OWNER TO postgres;

--
-- Name: templatelinks; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE templatelinks (
    tl_from integer NOT NULL,
    tl_from_namespace integer DEFAULT 0 NOT NULL,
    tl_namespace smallint NOT NULL,
    tl_title text NOT NULL
);


ALTER TABLE templatelinks OWNER TO postgres;

--
-- Name: transcache; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE transcache (
    tc_url text NOT NULL,
    tc_contents text NOT NULL,
    tc_time timestamp with time zone NOT NULL
);


ALTER TABLE transcache OWNER TO postgres;

--
-- Name: updatelog; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE updatelog (
    ul_key text NOT NULL,
    ul_value text
);


ALTER TABLE updatelog OWNER TO postgres;

--
-- Name: uploadstash_us_id_seq; Type: SEQUENCE; Schema: mediawiki; Owner: postgres
--

CREATE SEQUENCE uploadstash_us_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE uploadstash_us_id_seq OWNER TO postgres;

--
-- Name: uploadstash; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE uploadstash (
    us_id integer DEFAULT nextval('uploadstash_us_id_seq'::regclass) NOT NULL,
    us_user integer,
    us_key text,
    us_orig_path text,
    us_path text,
    us_props bytea,
    us_source_type text,
    us_timestamp timestamp with time zone,
    us_status text,
    us_chunk_inx integer,
    us_size integer,
    us_sha1 text,
    us_mime text,
    us_media_type media_type,
    us_image_width integer,
    us_image_height integer,
    us_image_bits smallint
);


ALTER TABLE uploadstash OWNER TO postgres;

--
-- Name: user_former_groups; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE user_former_groups (
    ufg_user integer,
    ufg_group text NOT NULL
);


ALTER TABLE user_former_groups OWNER TO postgres;

--
-- Name: user_groups; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE user_groups (
    ug_user integer,
    ug_group text NOT NULL
);


ALTER TABLE user_groups OWNER TO postgres;

--
-- Name: user_newtalk; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE user_newtalk (
    user_id integer NOT NULL,
    user_ip text,
    user_last_timestamp timestamp with time zone
);


ALTER TABLE user_newtalk OWNER TO postgres;

--
-- Name: user_properties; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE user_properties (
    up_user integer,
    up_property text NOT NULL,
    up_value text
);


ALTER TABLE user_properties OWNER TO postgres;

--
-- Name: valid_tag; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE valid_tag (
    vt_tag text NOT NULL
);


ALTER TABLE valid_tag OWNER TO postgres;

--
-- Name: watchlist; Type: TABLE; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE TABLE watchlist (
    wl_user integer NOT NULL,
    wl_namespace smallint DEFAULT 0 NOT NULL,
    wl_title text NOT NULL,
    wl_notificationtimestamp timestamp with time zone
);


ALTER TABLE watchlist OWNER TO postgres;

--
-- Name: archive_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY archive
    ADD CONSTRAINT archive_pkey PRIMARY KEY (ar_id);


--
-- Name: category_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY category
    ADD CONSTRAINT category_pkey PRIMARY KEY (cat_id);


--
-- Name: externallinks_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY externallinks
    ADD CONSTRAINT externallinks_pkey PRIMARY KEY (el_id);


--
-- Name: filearchive_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY filearchive
    ADD CONSTRAINT filearchive_pkey PRIMARY KEY (fa_id);


--
-- Name: image_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY image
    ADD CONSTRAINT image_pkey PRIMARY KEY (img_name);


--
-- Name: interwiki_iw_prefix_key; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY interwiki
    ADD CONSTRAINT interwiki_iw_prefix_key UNIQUE (iw_prefix);


--
-- Name: ipblocks_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY ipblocks
    ADD CONSTRAINT ipblocks_pkey PRIMARY KEY (ipb_id);


--
-- Name: job_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY job
    ADD CONSTRAINT job_pkey PRIMARY KEY (job_id);


--
-- Name: log_search_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY log_search
    ADD CONSTRAINT log_search_pkey PRIMARY KEY (ls_field, ls_value, ls_log_id);


--
-- Name: logging_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY logging
    ADD CONSTRAINT logging_pkey PRIMARY KEY (log_id);


--
-- Name: mwuser_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY mwuser
    ADD CONSTRAINT mwuser_pkey PRIMARY KEY (user_id);


--
-- Name: mwuser_user_name_key; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY mwuser
    ADD CONSTRAINT mwuser_user_name_key UNIQUE (user_name);


--
-- Name: objectcache_keyname_key; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY objectcache
    ADD CONSTRAINT objectcache_keyname_key UNIQUE (keyname);


--
-- Name: page_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY page
    ADD CONSTRAINT page_pkey PRIMARY KEY (page_id);


--
-- Name: page_props_pk; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY page_props
    ADD CONSTRAINT page_props_pk PRIMARY KEY (pp_page, pp_propname);


--
-- Name: page_restrictions_pk; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY page_restrictions
    ADD CONSTRAINT page_restrictions_pk PRIMARY KEY (pr_page, pr_type);


--
-- Name: page_restrictions_pr_id_key; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY page_restrictions
    ADD CONSTRAINT page_restrictions_pr_id_key UNIQUE (pr_id);


--
-- Name: pagecontent_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY pagecontent
    ADD CONSTRAINT pagecontent_pkey PRIMARY KEY (old_id);


--
-- Name: querycache_info_qci_type_key; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY querycache_info
    ADD CONSTRAINT querycache_info_qci_type_key UNIQUE (qci_type);


--
-- Name: recentchanges_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY recentchanges
    ADD CONSTRAINT recentchanges_pkey PRIMARY KEY (rc_id);


--
-- Name: revision_rev_id_key; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY revision
    ADD CONSTRAINT revision_rev_id_key UNIQUE (rev_id);


--
-- Name: site_stats_ss_row_id_key; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY site_stats
    ADD CONSTRAINT site_stats_ss_row_id_key UNIQUE (ss_row_id);


--
-- Name: sites_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY sites
    ADD CONSTRAINT sites_pkey PRIMARY KEY (site_id);


--
-- Name: transcache_tc_url_key; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY transcache
    ADD CONSTRAINT transcache_tc_url_key UNIQUE (tc_url);


--
-- Name: updatelog_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY updatelog
    ADD CONSTRAINT updatelog_pkey PRIMARY KEY (ul_key);


--
-- Name: uploadstash_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY uploadstash
    ADD CONSTRAINT uploadstash_pkey PRIMARY KEY (us_id);


--
-- Name: valid_tag_pkey; Type: CONSTRAINT; Schema: mediawiki; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY valid_tag
    ADD CONSTRAINT valid_tag_pkey PRIMARY KEY (vt_tag);


--
-- Name: archive_name_title_timestamp; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX archive_name_title_timestamp ON archive USING btree (ar_namespace, ar_title, ar_timestamp);


--
-- Name: archive_user_text; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX archive_user_text ON archive USING btree (ar_user_text);


--
-- Name: category_pages; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX category_pages ON category USING btree (cat_pages);


--
-- Name: category_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX category_title ON category USING btree (cat_title);


--
-- Name: change_tag_log_tag; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX change_tag_log_tag ON change_tag USING btree (ct_log_id, ct_tag);


--
-- Name: change_tag_rc_tag; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX change_tag_rc_tag ON change_tag USING btree (ct_rc_id, ct_tag);


--
-- Name: change_tag_rev_tag; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX change_tag_rev_tag ON change_tag USING btree (ct_rev_id, ct_tag);


--
-- Name: change_tag_tag_id; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX change_tag_tag_id ON change_tag USING btree (ct_tag, ct_rc_id, ct_rev_id, ct_log_id);


--
-- Name: cl_from; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX cl_from ON categorylinks USING btree (cl_from, cl_to);


--
-- Name: cl_sortkey; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX cl_sortkey ON categorylinks USING btree (cl_to, cl_sortkey, cl_from);


--
-- Name: externallinks_from_to; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX externallinks_from_to ON externallinks USING btree (el_from, el_to);


--
-- Name: externallinks_index; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX externallinks_index ON externallinks USING btree (el_index);


--
-- Name: fa_dupe; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX fa_dupe ON filearchive USING btree (fa_storage_group, fa_storage_key);


--
-- Name: fa_name_time; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX fa_name_time ON filearchive USING btree (fa_name, fa_timestamp);


--
-- Name: fa_notime; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX fa_notime ON filearchive USING btree (fa_deleted_timestamp);


--
-- Name: fa_nouser; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX fa_nouser ON filearchive USING btree (fa_deleted_user);


--
-- Name: fa_sha1; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX fa_sha1 ON filearchive USING btree (fa_sha1);


--
-- Name: il_from; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX il_from ON imagelinks USING btree (il_to, il_from);


--
-- Name: img_sha1; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX img_sha1 ON image USING btree (img_sha1);


--
-- Name: img_size_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX img_size_idx ON image USING btree (img_size);


--
-- Name: img_timestamp_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX img_timestamp_idx ON image USING btree (img_timestamp);


--
-- Name: ipb_address_unique; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX ipb_address_unique ON ipblocks USING btree (ipb_address, ipb_user, ipb_auto, ipb_anon_only);


--
-- Name: ipb_parent_block_id; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX ipb_parent_block_id ON ipblocks USING btree (ipb_parent_block_id);


--
-- Name: ipb_range; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX ipb_range ON ipblocks USING btree (ipb_range_start, ipb_range_end);


--
-- Name: ipb_user; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX ipb_user ON ipblocks USING btree (ipb_user);


--
-- Name: iwl_from; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX iwl_from ON iwlinks USING btree (iwl_from, iwl_prefix, iwl_title);


--
-- Name: iwl_prefix_from_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX iwl_prefix_from_title ON iwlinks USING btree (iwl_prefix, iwl_from, iwl_title);


--
-- Name: iwl_prefix_title_from; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX iwl_prefix_title_from ON iwlinks USING btree (iwl_prefix, iwl_title, iwl_from);


--
-- Name: job_cmd_namespace_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX job_cmd_namespace_title ON job USING btree (job_cmd, job_namespace, job_title);


--
-- Name: job_cmd_token; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX job_cmd_token ON job USING btree (job_cmd, job_token, job_random);


--
-- Name: job_cmd_token_id; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX job_cmd_token_id ON job USING btree (job_cmd, job_token, job_id);


--
-- Name: job_sha1; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX job_sha1 ON job USING btree (job_sha1);


--
-- Name: job_timestamp_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX job_timestamp_idx ON job USING btree (job_timestamp);


--
-- Name: l10n_cache_lc_lang_key; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX l10n_cache_lc_lang_key ON l10n_cache USING btree (lc_lang, lc_key);


--
-- Name: langlinks_lang_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX langlinks_lang_title ON langlinks USING btree (ll_lang, ll_title);


--
-- Name: langlinks_unique; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX langlinks_unique ON langlinks USING btree (ll_from, ll_lang);


--
-- Name: logging_page_id_time; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX logging_page_id_time ON logging USING btree (log_page, log_timestamp);


--
-- Name: logging_page_time; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX logging_page_time ON logging USING btree (log_namespace, log_title, log_timestamp);


--
-- Name: logging_times; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX logging_times ON logging USING btree (log_timestamp);


--
-- Name: logging_type_name; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX logging_type_name ON logging USING btree (log_type, log_timestamp);


--
-- Name: logging_user_text_time; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX logging_user_text_time ON logging USING btree (log_user_text, log_timestamp);


--
-- Name: logging_user_text_type_time; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX logging_user_text_type_time ON logging USING btree (log_user_text, log_type, log_timestamp);


--
-- Name: logging_user_time; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX logging_user_time ON logging USING btree (log_timestamp, log_user);


--
-- Name: logging_user_type_time; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX logging_user_type_time ON logging USING btree (log_user, log_type, log_timestamp);


--
-- Name: ls_log_id; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX ls_log_id ON log_search USING btree (ls_log_id);


--
-- Name: md_module_skin; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX md_module_skin ON module_deps USING btree (md_module, md_skin);


--
-- Name: mr_resource_lang; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX mr_resource_lang ON msg_resource USING btree (mr_resource, mr_lang);


--
-- Name: mrl_message_resource; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX mrl_message_resource ON msg_resource_links USING btree (mrl_message, mrl_resource);


--
-- Name: new_name_timestamp; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX new_name_timestamp ON recentchanges USING btree (rc_new, rc_namespace, rc_timestamp);


--
-- Name: objectcacache_exptime; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX objectcacache_exptime ON objectcache USING btree (exptime);


--
-- Name: oi_name_archive_name; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX oi_name_archive_name ON oldimage USING btree (oi_name, oi_archive_name);


--
-- Name: oi_name_timestamp; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX oi_name_timestamp ON oldimage USING btree (oi_name, oi_timestamp);


--
-- Name: oi_sha1; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX oi_sha1 ON oldimage USING btree (oi_sha1);


--
-- Name: page_len_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX page_len_idx ON page USING btree (page_len);


--
-- Name: page_main_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX page_main_title ON page USING btree (page_title text_pattern_ops) WHERE (page_namespace = 0);


--
-- Name: page_mediawiki_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX page_mediawiki_title ON page USING btree (page_title text_pattern_ops) WHERE (page_namespace = 8);


--
-- Name: page_project_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX page_project_title ON page USING btree (page_title text_pattern_ops) WHERE (page_namespace = 4);


--
-- Name: page_props_propname; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX page_props_propname ON page_props USING btree (pp_propname);


--
-- Name: page_random_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX page_random_idx ON page USING btree (page_random);


--
-- Name: page_talk_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX page_talk_title ON page USING btree (page_title text_pattern_ops) WHERE (page_namespace = 1);


--
-- Name: page_unique_name; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX page_unique_name ON page USING btree (page_namespace, page_title);


--
-- Name: page_user_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX page_user_title ON page USING btree (page_title text_pattern_ops) WHERE (page_namespace = 2);


--
-- Name: page_utalk_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX page_utalk_title ON page USING btree (page_title text_pattern_ops) WHERE (page_namespace = 3);


--
-- Name: pagelink_unique; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX pagelink_unique ON pagelinks USING btree (pl_from, pl_namespace, pl_title);


--
-- Name: pagelinks_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX pagelinks_title ON pagelinks USING btree (pl_title);


--
-- Name: pf_name_server; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX pf_name_server ON profiling USING btree (pf_name, pf_server);


--
-- Name: pp_propname_page; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX pp_propname_page ON page_props USING btree (pp_propname, pp_page);


--
-- Name: pp_propname_sortkey_page; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX pp_propname_sortkey_page ON page_props USING btree (pp_propname, pp_sortkey, pp_page) WHERE (pp_sortkey IS NOT NULL);


--
-- Name: protected_titles_unique; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX protected_titles_unique ON protected_titles USING btree (pt_namespace, pt_title);


--
-- Name: querycache_type_value; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX querycache_type_value ON querycache USING btree (qc_type, qc_value);


--
-- Name: querycachetwo_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX querycachetwo_title ON querycachetwo USING btree (qcc_type, qcc_namespace, qcc_title);


--
-- Name: querycachetwo_titletwo; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX querycachetwo_titletwo ON querycachetwo USING btree (qcc_type, qcc_namespacetwo, qcc_titletwo);


--
-- Name: querycachetwo_type_value; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX querycachetwo_type_value ON querycachetwo USING btree (qcc_type, qcc_value);


--
-- Name: rc_cur_id; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX rc_cur_id ON recentchanges USING btree (rc_cur_id);


--
-- Name: rc_ip; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX rc_ip ON recentchanges USING btree (rc_ip);


--
-- Name: rc_namespace_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX rc_namespace_title ON recentchanges USING btree (rc_namespace, rc_title);


--
-- Name: rc_timestamp; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX rc_timestamp ON recentchanges USING btree (rc_timestamp);


--
-- Name: rc_timestamp_bot; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX rc_timestamp_bot ON recentchanges USING btree (rc_timestamp) WHERE (rc_bot = 0);


--
-- Name: redirect_ns_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX redirect_ns_title ON redirect USING btree (rd_namespace, rd_title, rd_from);


--
-- Name: rev_text_id_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX rev_text_id_idx ON revision USING btree (rev_text_id);


--
-- Name: rev_timestamp_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX rev_timestamp_idx ON revision USING btree (rev_timestamp);


--
-- Name: rev_user_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX rev_user_idx ON revision USING btree (rev_user);


--
-- Name: rev_user_text_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX rev_user_text_idx ON revision USING btree (rev_user_text);


--
-- Name: revision_unique; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX revision_unique ON revision USING btree (rev_page, rev_id);


--
-- Name: si_key; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX si_key ON site_identifiers USING btree (si_key);


--
-- Name: si_site; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX si_site ON site_identifiers USING btree (si_site);


--
-- Name: si_type_key; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX si_type_key ON site_identifiers USING btree (si_type, si_key);


--
-- Name: site_domain; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX site_domain ON sites USING btree (site_domain);


--
-- Name: site_forward; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX site_forward ON sites USING btree (site_forward);


--
-- Name: site_global_key; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX site_global_key ON sites USING btree (site_global_key);


--
-- Name: site_group; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX site_group ON sites USING btree (site_group);


--
-- Name: site_language; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX site_language ON sites USING btree (site_language);


--
-- Name: site_protocol; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX site_protocol ON sites USING btree (site_protocol);


--
-- Name: site_source; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX site_source ON sites USING btree (site_source);


--
-- Name: site_type; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX site_type ON sites USING btree (site_type);


--
-- Name: tag_summary_log_id; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX tag_summary_log_id ON tag_summary USING btree (ts_log_id);


--
-- Name: tag_summary_rc_id; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX tag_summary_rc_id ON tag_summary USING btree (ts_rc_id);


--
-- Name: tag_summary_rev_id; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX tag_summary_rev_id ON tag_summary USING btree (ts_rev_id);


--
-- Name: templatelinks_from; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX templatelinks_from ON templatelinks USING btree (tl_from);


--
-- Name: templatelinks_unique; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX templatelinks_unique ON templatelinks USING btree (tl_namespace, tl_title, tl_from);


--
-- Name: ts2_page_text; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX ts2_page_text ON pagecontent USING gin (textvector);


--
-- Name: ts2_page_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX ts2_page_title ON page USING gin (titlevector);


--
-- Name: ufg_user_group; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX ufg_user_group ON user_former_groups USING btree (ufg_user, ufg_group);


--
-- Name: us_key_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX us_key_idx ON uploadstash USING btree (us_key);


--
-- Name: us_timestamp_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX us_timestamp_idx ON uploadstash USING btree (us_timestamp);


--
-- Name: us_user_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX us_user_idx ON uploadstash USING btree (us_user);


--
-- Name: user_email_token_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX user_email_token_idx ON mwuser USING btree (user_email_token);


--
-- Name: user_groups_unique; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX user_groups_unique ON user_groups USING btree (ug_user, ug_group);


--
-- Name: user_newtalk_id_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX user_newtalk_id_idx ON user_newtalk USING btree (user_id);


--
-- Name: user_newtalk_ip_idx; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX user_newtalk_ip_idx ON user_newtalk USING btree (user_ip);


--
-- Name: user_properties_property; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX user_properties_property ON user_properties USING btree (up_property);


--
-- Name: user_properties_user_property; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX user_properties_user_property ON user_properties USING btree (up_user, up_property);


--
-- Name: wl_user; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX wl_user ON watchlist USING btree (wl_user);


--
-- Name: wl_user_namespace_title; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE UNIQUE INDEX wl_user_namespace_title ON watchlist USING btree (wl_namespace, wl_title, wl_user);


--
-- Name: wl_user_notificationtimestamp; Type: INDEX; Schema: mediawiki; Owner: postgres; Tablespace: 
--

CREATE INDEX wl_user_notificationtimestamp ON watchlist USING btree (wl_user, wl_notificationtimestamp);


--
-- Name: page_deleted; Type: TRIGGER; Schema: mediawiki; Owner: postgres
--

CREATE TRIGGER page_deleted AFTER DELETE ON page FOR EACH ROW EXECUTE PROCEDURE page_deleted();


--
-- Name: ts2_page_text; Type: TRIGGER; Schema: mediawiki; Owner: postgres
--

CREATE TRIGGER ts2_page_text BEFORE INSERT OR UPDATE ON pagecontent FOR EACH ROW EXECUTE PROCEDURE ts2_page_text();


--
-- Name: ts2_page_title; Type: TRIGGER; Schema: mediawiki; Owner: postgres
--

CREATE TRIGGER ts2_page_title BEFORE INSERT OR UPDATE ON page FOR EACH ROW EXECUTE PROCEDURE ts2_page_title();


--
-- Name: archive_ar_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY archive
    ADD CONSTRAINT archive_ar_user_fkey FOREIGN KEY (ar_user) REFERENCES mwuser(user_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;


--
-- Name: categorylinks_cl_from_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY categorylinks
    ADD CONSTRAINT categorylinks_cl_from_fkey FOREIGN KEY (cl_from) REFERENCES page(page_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: externallinks_el_from_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY externallinks
    ADD CONSTRAINT externallinks_el_from_fkey FOREIGN KEY (el_from) REFERENCES page(page_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: filearchive_fa_deleted_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY filearchive
    ADD CONSTRAINT filearchive_fa_deleted_user_fkey FOREIGN KEY (fa_deleted_user) REFERENCES mwuser(user_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;


--
-- Name: filearchive_fa_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY filearchive
    ADD CONSTRAINT filearchive_fa_user_fkey FOREIGN KEY (fa_user) REFERENCES mwuser(user_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;


--
-- Name: image_img_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY image
    ADD CONSTRAINT image_img_user_fkey FOREIGN KEY (img_user) REFERENCES mwuser(user_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;


--
-- Name: imagelinks_il_from_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY imagelinks
    ADD CONSTRAINT imagelinks_il_from_fkey FOREIGN KEY (il_from) REFERENCES page(page_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: ipblocks_ipb_by_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY ipblocks
    ADD CONSTRAINT ipblocks_ipb_by_fkey FOREIGN KEY (ipb_by) REFERENCES mwuser(user_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: ipblocks_ipb_parent_block_id_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY ipblocks
    ADD CONSTRAINT ipblocks_ipb_parent_block_id_fkey FOREIGN KEY (ipb_parent_block_id) REFERENCES ipblocks(ipb_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;


--
-- Name: ipblocks_ipb_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY ipblocks
    ADD CONSTRAINT ipblocks_ipb_user_fkey FOREIGN KEY (ipb_user) REFERENCES mwuser(user_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;


--
-- Name: langlinks_ll_from_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY langlinks
    ADD CONSTRAINT langlinks_ll_from_fkey FOREIGN KEY (ll_from) REFERENCES page(page_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: logging_log_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY logging
    ADD CONSTRAINT logging_log_user_fkey FOREIGN KEY (log_user) REFERENCES mwuser(user_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;


--
-- Name: oldimage_oi_name_fkey_cascaded; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY oldimage
    ADD CONSTRAINT oldimage_oi_name_fkey_cascaded FOREIGN KEY (oi_name) REFERENCES image(img_name) ON UPDATE CASCADE ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: oldimage_oi_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY oldimage
    ADD CONSTRAINT oldimage_oi_user_fkey FOREIGN KEY (oi_user) REFERENCES mwuser(user_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;


--
-- Name: page_props_pp_page_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY page_props
    ADD CONSTRAINT page_props_pp_page_fkey FOREIGN KEY (pp_page) REFERENCES page(page_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: page_restrictions_pr_page_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY page_restrictions
    ADD CONSTRAINT page_restrictions_pr_page_fkey FOREIGN KEY (pr_page) REFERENCES page(page_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: pagelinks_pl_from_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY pagelinks
    ADD CONSTRAINT pagelinks_pl_from_fkey FOREIGN KEY (pl_from) REFERENCES page(page_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: protected_titles_pt_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY protected_titles
    ADD CONSTRAINT protected_titles_pt_user_fkey FOREIGN KEY (pt_user) REFERENCES mwuser(user_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;


--
-- Name: recentchanges_rc_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY recentchanges
    ADD CONSTRAINT recentchanges_rc_user_fkey FOREIGN KEY (rc_user) REFERENCES mwuser(user_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED;


--
-- Name: redirect_rd_from_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY redirect
    ADD CONSTRAINT redirect_rd_from_fkey FOREIGN KEY (rd_from) REFERENCES page(page_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: revision_rev_page_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY revision
    ADD CONSTRAINT revision_rev_page_fkey FOREIGN KEY (rev_page) REFERENCES page(page_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: revision_rev_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY revision
    ADD CONSTRAINT revision_rev_user_fkey FOREIGN KEY (rev_user) REFERENCES mwuser(user_id) ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED;


--
-- Name: templatelinks_tl_from_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY templatelinks
    ADD CONSTRAINT templatelinks_tl_from_fkey FOREIGN KEY (tl_from) REFERENCES page(page_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: user_former_groups_ufg_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY user_former_groups
    ADD CONSTRAINT user_former_groups_ufg_user_fkey FOREIGN KEY (ufg_user) REFERENCES mwuser(user_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: user_groups_ug_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY user_groups
    ADD CONSTRAINT user_groups_ug_user_fkey FOREIGN KEY (ug_user) REFERENCES mwuser(user_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: user_newtalk_user_id_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY user_newtalk
    ADD CONSTRAINT user_newtalk_user_id_fkey FOREIGN KEY (user_id) REFERENCES mwuser(user_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: user_properties_up_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY user_properties
    ADD CONSTRAINT user_properties_up_user_fkey FOREIGN KEY (up_user) REFERENCES mwuser(user_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: watchlist_wl_user_fkey; Type: FK CONSTRAINT; Schema: mediawiki; Owner: postgres
--

ALTER TABLE ONLY watchlist
    ADD CONSTRAINT watchlist_wl_user_fkey FOREIGN KEY (wl_user) REFERENCES mwuser(user_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED;


--
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

\connect postgres

SET default_transaction_read_only = off;

--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: postgres; Type: COMMENT; Schema: -; Owner: postgres
--

COMMENT ON DATABASE postgres IS 'default administrative connection database';


--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


--
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

\connect qfp

SET default_transaction_read_only = off;

--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


--
-- Name: plpython3u; Type: PROCEDURAL LANGUAGE; Schema: -; Owner: sawaya
--

CREATE OR REPLACE PROCEDURAL LANGUAGE plpython3u;


ALTER PROCEDURAL LANGUAGE plpython3u OWNER TO sawaya;

SET search_path = public, pg_catalog;

--
-- Name: breakdowntest(text, integer); Type: FUNCTION; Schema: public; Owner: sawaya
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


ALTER FUNCTION public.breakdowntest(name text, run integer) OWNER TO sawaya;

--
-- Name: cleanupresults(integer); Type: FUNCTION; Schema: public; Owner: sawaya
--

CREATE FUNCTION cleanupresults(run integer DEFAULT (-1)) RETURNS integer[]
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


ALTER FUNCTION public.cleanupresults(run integer) OWNER TO sawaya;

--
-- Name: dofullflitimport(text, text); Type: FUNCTION; Schema: public; Owner: sawaya
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
query = ("SELECT importqfpresults2('" + path + "', " +
         str(run) + ")")
res = plpy.execute(query)
query = ("SELECT importopcoderesults('" + path + "/pins'," +
         str(run) + ")")
res2 = plpy.execute(query)

return [res[0]['importqfpresults2'][0],res[0]['importqfpresults2'][1],
        res2[0]['importopcoderesults'][0],res2[0]['importopcoderesults'][1]]

$$;


ALTER FUNCTION public.dofullflitimport(path text, notes text) OWNER TO sawaya;

--
-- Name: importopcoderesults(text, integer); Type: FUNCTION; Schema: public; Owner: sawaya
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


ALTER FUNCTION public.importopcoderesults(path text, run integer) OWNER TO sawaya;

--
-- Name: importqfpresults(text); Type: FUNCTION; Schema: public; Owner: sawaya
--

CREATE FUNCTION importqfpresults(path text) RETURNS integer
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


ALTER FUNCTION public.importqfpresults(path text) OWNER TO sawaya;

--
-- Name: importqfpresults2(text, integer); Type: FUNCTION; Schema: public; Owner: sawaya
--

CREATE FUNCTION importqfpresults2(path text, run integer) RETURNS integer[]
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


ALTER FUNCTION public.importqfpresults2(path text, run integer) OWNER TO sawaya;

--
-- Name: importswitches(text); Type: FUNCTION; Schema: public; Owner: sawaya
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


ALTER FUNCTION public.importswitches(path text) OWNER TO sawaya;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: clusters; Type: TABLE; Schema: public; Owner: sawaya; Tablespace: 
--

CREATE TABLE clusters (
    testid integer NOT NULL,
    number integer
);


ALTER TABLE clusters OWNER TO sawaya;

--
-- Name: op_counts; Type: TABLE; Schema: public; Owner: sawaya; Tablespace: 
--

CREATE TABLE op_counts (
    test_id integer NOT NULL,
    opcode integer NOT NULL,
    count integer,
    pred_count integer,
    dynamic boolean NOT NULL
);


ALTER TABLE op_counts OWNER TO sawaya;

--
-- Name: opcodes; Type: TABLE; Schema: public; Owner: sawaya; Tablespace: 
--

CREATE TABLE opcodes (
    index integer NOT NULL,
    name text
);


ALTER TABLE opcodes OWNER TO sawaya;

--
-- Name: runs; Type: TABLE; Schema: public; Owner: sawaya; Tablespace: 
--

CREATE TABLE runs (
    index integer NOT NULL,
    rdate timestamp without time zone,
    notes text
);


ALTER TABLE runs OWNER TO sawaya;

--
-- Name: run_index_seq; Type: SEQUENCE; Schema: public; Owner: sawaya
--

CREATE SEQUENCE run_index_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE run_index_seq OWNER TO sawaya;

--
-- Name: run_index_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sawaya
--

ALTER SEQUENCE run_index_seq OWNED BY runs.index;


--
-- Name: skipped_pin; Type: TABLE; Schema: public; Owner: sawaya; Tablespace: 
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


ALTER TABLE skipped_pin OWNER TO sawaya;

--
-- Name: switch_conv; Type: TABLE; Schema: public; Owner: sawaya; Tablespace: 
--

CREATE TABLE switch_conv (
    abbrev text NOT NULL,
    switches text
);


ALTER TABLE switch_conv OWNER TO sawaya;

--
-- Name: tests; Type: TABLE; Schema: public; Owner: sawaya; Tablespace: 
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


ALTER TABLE tests OWNER TO sawaya;

--
-- Name: tests_colname_seq; Type: SEQUENCE; Schema: public; Owner: sawaya
--

CREATE SEQUENCE tests_colname_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE tests_colname_seq OWNER TO sawaya;

--
-- Name: tests_colname_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sawaya
--

ALTER SEQUENCE tests_colname_seq OWNED BY tests.index;


--
-- Name: index; Type: DEFAULT; Schema: public; Owner: sawaya
--

ALTER TABLE ONLY runs ALTER COLUMN index SET DEFAULT nextval('run_index_seq'::regclass);


--
-- Name: index; Type: DEFAULT; Schema: public; Owner: sawaya
--

ALTER TABLE ONLY tests ALTER COLUMN index SET DEFAULT nextval('tests_colname_seq'::regclass);


--
-- Name: clusters_pkey; Type: CONSTRAINT; Schema: public; Owner: sawaya; Tablespace: 
--

ALTER TABLE ONLY clusters
    ADD CONSTRAINT clusters_pkey PRIMARY KEY (testid);


--
-- Name: op_counts_pkey; Type: CONSTRAINT; Schema: public; Owner: sawaya; Tablespace: 
--

ALTER TABLE ONLY op_counts
    ADD CONSTRAINT op_counts_pkey PRIMARY KEY (test_id, opcode, dynamic);


--
-- Name: opcodes_pkey; Type: CONSTRAINT; Schema: public; Owner: sawaya; Tablespace: 
--

ALTER TABLE ONLY opcodes
    ADD CONSTRAINT opcodes_pkey PRIMARY KEY (index);


--
-- Name: runs_pkey; Type: CONSTRAINT; Schema: public; Owner: sawaya; Tablespace: 
--

ALTER TABLE ONLY runs
    ADD CONSTRAINT runs_pkey PRIMARY KEY (index);


--
-- Name: switch_conv_pkey; Type: CONSTRAINT; Schema: public; Owner: sawaya; Tablespace: 
--

ALTER TABLE ONLY switch_conv
    ADD CONSTRAINT switch_conv_pkey PRIMARY KEY (abbrev);


--
-- Name: tests_pkey; Type: CONSTRAINT; Schema: public; Owner: sawaya; Tablespace: 
--

ALTER TABLE ONLY tests
    ADD CONSTRAINT tests_pkey PRIMARY KEY (index);


--
-- Name: op_counts_opcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sawaya
--

ALTER TABLE ONLY op_counts
    ADD CONSTRAINT op_counts_opcode_fkey FOREIGN KEY (opcode) REFERENCES opcodes(index);


--
-- Name: op_counts_test_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sawaya
--

ALTER TABLE ONLY op_counts
    ADD CONSTRAINT op_counts_test_id_fkey FOREIGN KEY (test_id) REFERENCES tests(index);


--
-- Name: tests_run_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sawaya
--

ALTER TABLE ONLY tests
    ADD CONSTRAINT tests_run_fkey FOREIGN KEY (run) REFERENCES runs(index);


--
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- Name: breakdowntest(text, integer); Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON FUNCTION breakdowntest(name text, run integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION breakdowntest(name text, run integer) FROM sawaya;
GRANT ALL ON FUNCTION breakdowntest(name text, run integer) TO sawaya;
GRANT ALL ON FUNCTION breakdowntest(name text, run integer) TO PUBLIC;
GRANT ALL ON FUNCTION breakdowntest(name text, run integer) TO qfp;
GRANT ALL ON FUNCTION breakdowntest(name text, run integer) TO mbentley;


--
-- Name: cleanupresults(integer); Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON FUNCTION cleanupresults(run integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION cleanupresults(run integer) FROM sawaya;
GRANT ALL ON FUNCTION cleanupresults(run integer) TO sawaya;
GRANT ALL ON FUNCTION cleanupresults(run integer) TO PUBLIC;
GRANT ALL ON FUNCTION cleanupresults(run integer) TO mbentley;
GRANT ALL ON FUNCTION cleanupresults(run integer) TO qfp;


--
-- Name: dofullflitimport(text, text); Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON FUNCTION dofullflitimport(path text, notes text) FROM PUBLIC;
REVOKE ALL ON FUNCTION dofullflitimport(path text, notes text) FROM sawaya;
GRANT ALL ON FUNCTION dofullflitimport(path text, notes text) TO sawaya;
GRANT ALL ON FUNCTION dofullflitimport(path text, notes text) TO PUBLIC;
GRANT ALL ON FUNCTION dofullflitimport(path text, notes text) TO qfp;
GRANT ALL ON FUNCTION dofullflitimport(path text, notes text) TO mbentley;


--
-- Name: importopcoderesults(text, integer); Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON FUNCTION importopcoderesults(path text, run integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION importopcoderesults(path text, run integer) FROM sawaya;
GRANT ALL ON FUNCTION importopcoderesults(path text, run integer) TO sawaya;
GRANT ALL ON FUNCTION importopcoderesults(path text, run integer) TO PUBLIC;
GRANT ALL ON FUNCTION importopcoderesults(path text, run integer) TO mbentley;
GRANT ALL ON FUNCTION importopcoderesults(path text, run integer) TO qfp;


--
-- Name: importqfpresults(text); Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON FUNCTION importqfpresults(path text) FROM PUBLIC;
REVOKE ALL ON FUNCTION importqfpresults(path text) FROM sawaya;
GRANT ALL ON FUNCTION importqfpresults(path text) TO sawaya;
GRANT ALL ON FUNCTION importqfpresults(path text) TO PUBLIC;
GRANT ALL ON FUNCTION importqfpresults(path text) TO mbentley;
GRANT ALL ON FUNCTION importqfpresults(path text) TO qfp;


--
-- Name: importqfpresults2(text, integer); Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON FUNCTION importqfpresults2(path text, run integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION importqfpresults2(path text, run integer) FROM sawaya;
GRANT ALL ON FUNCTION importqfpresults2(path text, run integer) TO sawaya;
GRANT ALL ON FUNCTION importqfpresults2(path text, run integer) TO PUBLIC;
GRANT ALL ON FUNCTION importqfpresults2(path text, run integer) TO qfp;
GRANT ALL ON FUNCTION importqfpresults2(path text, run integer) TO mbentley;


--
-- Name: importswitches(text); Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON FUNCTION importswitches(path text) FROM PUBLIC;
REVOKE ALL ON FUNCTION importswitches(path text) FROM sawaya;
GRANT ALL ON FUNCTION importswitches(path text) TO sawaya;
GRANT ALL ON FUNCTION importswitches(path text) TO PUBLIC;
GRANT ALL ON FUNCTION importswitches(path text) TO mbentley;
GRANT ALL ON FUNCTION importswitches(path text) TO qfp;


--
-- Name: clusters; Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON TABLE clusters FROM PUBLIC;
REVOKE ALL ON TABLE clusters FROM sawaya;
GRANT ALL ON TABLE clusters TO sawaya;
GRANT ALL ON TABLE clusters TO qfp;
GRANT ALL ON TABLE clusters TO mbentley;


--
-- Name: op_counts; Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON TABLE op_counts FROM PUBLIC;
REVOKE ALL ON TABLE op_counts FROM sawaya;
GRANT ALL ON TABLE op_counts TO sawaya;
GRANT ALL ON TABLE op_counts TO mbentley;
GRANT ALL ON TABLE op_counts TO qfp;


--
-- Name: opcodes; Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON TABLE opcodes FROM PUBLIC;
REVOKE ALL ON TABLE opcodes FROM sawaya;
GRANT ALL ON TABLE opcodes TO sawaya;
GRANT ALL ON TABLE opcodes TO qfp;
GRANT ALL ON TABLE opcodes TO mbentley;


--
-- Name: runs; Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON TABLE runs FROM PUBLIC;
REVOKE ALL ON TABLE runs FROM sawaya;
GRANT ALL ON TABLE runs TO sawaya;
GRANT ALL ON TABLE runs TO qfp;
GRANT ALL ON TABLE runs TO mbentley;


--
-- Name: run_index_seq; Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON SEQUENCE run_index_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE run_index_seq FROM sawaya;
GRANT ALL ON SEQUENCE run_index_seq TO sawaya;
GRANT ALL ON SEQUENCE run_index_seq TO qfp;
GRANT ALL ON SEQUENCE run_index_seq TO mbentley;


--
-- Name: skipped_pin; Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON TABLE skipped_pin FROM PUBLIC;
REVOKE ALL ON TABLE skipped_pin FROM sawaya;
GRANT ALL ON TABLE skipped_pin TO sawaya;
GRANT ALL ON TABLE skipped_pin TO qfp;
GRANT ALL ON TABLE skipped_pin TO mbentley;


--
-- Name: switch_conv; Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON TABLE switch_conv FROM PUBLIC;
REVOKE ALL ON TABLE switch_conv FROM sawaya;
GRANT ALL ON TABLE switch_conv TO sawaya;
GRANT ALL ON TABLE switch_conv TO qfp;
GRANT ALL ON TABLE switch_conv TO mbentley;


--
-- Name: tests; Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON TABLE tests FROM PUBLIC;
REVOKE ALL ON TABLE tests FROM sawaya;
GRANT ALL ON TABLE tests TO sawaya;
GRANT ALL ON TABLE tests TO qfp;
GRANT ALL ON TABLE tests TO mbentley;


--
-- Name: tests_colname_seq; Type: ACL; Schema: public; Owner: sawaya
--

REVOKE ALL ON SEQUENCE tests_colname_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE tests_colname_seq FROM sawaya;
GRANT ALL ON SEQUENCE tests_colname_seq TO sawaya;
GRANT ALL ON SEQUENCE tests_colname_seq TO qfp;
GRANT ALL ON SEQUENCE tests_colname_seq TO mbentley;


--
-- Name: DEFAULT PRIVILEGES FOR SEQUENCES; Type: DEFAULT ACL; Schema: -; Owner: sawaya
--

ALTER DEFAULT PRIVILEGES FOR ROLE sawaya REVOKE ALL ON SEQUENCES  FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE sawaya REVOKE ALL ON SEQUENCES  FROM sawaya;
ALTER DEFAULT PRIVILEGES FOR ROLE sawaya GRANT ALL ON SEQUENCES  TO sawaya;
ALTER DEFAULT PRIVILEGES FOR ROLE sawaya GRANT ALL ON SEQUENCES  TO qfp;
ALTER DEFAULT PRIVILEGES FOR ROLE sawaya GRANT ALL ON SEQUENCES  TO mbentley;


--
-- Name: DEFAULT PRIVILEGES FOR FUNCTIONS; Type: DEFAULT ACL; Schema: -; Owner: sawaya
--

ALTER DEFAULT PRIVILEGES FOR ROLE sawaya REVOKE ALL ON FUNCTIONS  FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE sawaya REVOKE ALL ON FUNCTIONS  FROM sawaya;
ALTER DEFAULT PRIVILEGES FOR ROLE sawaya GRANT ALL ON FUNCTIONS  TO sawaya;
ALTER DEFAULT PRIVILEGES FOR ROLE sawaya GRANT ALL ON FUNCTIONS  TO PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE sawaya GRANT ALL ON FUNCTIONS  TO qfp;
ALTER DEFAULT PRIVILEGES FOR ROLE sawaya GRANT ALL ON FUNCTIONS  TO mbentley;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: -; Owner: sawaya
--

ALTER DEFAULT PRIVILEGES FOR ROLE sawaya REVOKE ALL ON TABLES  FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE sawaya REVOKE ALL ON TABLES  FROM sawaya;
ALTER DEFAULT PRIVILEGES FOR ROLE sawaya GRANT ALL ON TABLES  TO sawaya;
ALTER DEFAULT PRIVILEGES FOR ROLE sawaya GRANT ALL ON TABLES  TO qfp;
ALTER DEFAULT PRIVILEGES FOR ROLE sawaya GRANT ALL ON TABLES  TO mbentley;


--
-- PostgreSQL database dump complete
--

\connect qfp_pretrunc

SET default_transaction_read_only = off;

--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: compilers; Type: TABLE; Schema: public; Owner: sawaya; Tablespace: 
--

CREATE TABLE compilers (
    vendor character varying(255),
    version character varying(255),
    index integer NOT NULL
);


ALTER TABLE compilers OWNER TO sawaya;

--
-- Name: compilers_index_seq; Type: SEQUENCE; Schema: public; Owner: sawaya
--

CREATE SEQUENCE compilers_index_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE compilers_index_seq OWNER TO sawaya;

--
-- Name: compilers_index_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sawaya
--

ALTER SEQUENCE compilers_index_seq OWNED BY compilers.index;


--
-- Name: hosts; Type: TABLE; Schema: public; Owner: sawaya; Tablespace: 
--

CREATE TABLE hosts (
    index integer NOT NULL,
    name character varying(50),
    cpuinfo text,
    fqdn character varying(255)
);


ALTER TABLE hosts OWNER TO sawaya;

--
-- Name: hosts_index_seq; Type: SEQUENCE; Schema: public; Owner: sawaya
--

CREATE SEQUENCE hosts_index_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE hosts_index_seq OWNER TO sawaya;

--
-- Name: hosts_index_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sawaya
--

ALTER SEQUENCE hosts_index_seq OWNED BY hosts.index;


--
-- Name: tests2; Type: TABLE; Schema: public; Owner: sawaya; Tablespace: 
--

CREATE TABLE tests2 (
    switches character varying(512),
    "precision" character varying(1),
    sort character varying(2),
    score character varying(32),
    scored numeric(200,195),
    host character varying(50),
    compiler character varying(50),
    name character varying(255),
    index integer NOT NULL
);


ALTER TABLE tests2 OWNER TO sawaya;

--
-- Name: tests_pretrunc; Type: TABLE; Schema: public; Owner: sawaya; Tablespace: 
--

CREATE TABLE tests_pretrunc (
    switches character varying(512),
    "precision" character varying(1),
    sort character varying(2),
    score character varying(32),
    scored numeric(200,195),
    host character varying(50),
    compiler character varying(50),
    name character varying(255),
    index integer NOT NULL
);


ALTER TABLE tests_pretrunc OWNER TO sawaya;

--
-- Name: tests_index_seq; Type: SEQUENCE; Schema: public; Owner: sawaya
--

CREATE SEQUENCE tests_index_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE tests_index_seq OWNER TO sawaya;

--
-- Name: tests_index_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sawaya
--

ALTER SEQUENCE tests_index_seq OWNED BY tests_pretrunc.index;


--
-- Name: index; Type: DEFAULT; Schema: public; Owner: sawaya
--

ALTER TABLE ONLY compilers ALTER COLUMN index SET DEFAULT nextval('compilers_index_seq'::regclass);


--
-- Name: index; Type: DEFAULT; Schema: public; Owner: sawaya
--

ALTER TABLE ONLY hosts ALTER COLUMN index SET DEFAULT nextval('hosts_index_seq'::regclass);


--
-- Name: index; Type: DEFAULT; Schema: public; Owner: sawaya
--

ALTER TABLE ONLY tests_pretrunc ALTER COLUMN index SET DEFAULT nextval('tests_index_seq'::regclass);


--
-- Name: compilers_pkey; Type: CONSTRAINT; Schema: public; Owner: sawaya; Tablespace: 
--

ALTER TABLE ONLY compilers
    ADD CONSTRAINT compilers_pkey PRIMARY KEY (index);


--
-- Name: hosts_pkey; Type: CONSTRAINT; Schema: public; Owner: sawaya; Tablespace: 
--

ALTER TABLE ONLY hosts
    ADD CONSTRAINT hosts_pkey PRIMARY KEY (index);


--
-- Name: tests2_pkey; Type: CONSTRAINT; Schema: public; Owner: sawaya; Tablespace: 
--

ALTER TABLE ONLY tests2
    ADD CONSTRAINT tests2_pkey PRIMARY KEY (index);


--
-- Name: tests_pkey; Type: CONSTRAINT; Schema: public; Owner: sawaya; Tablespace: 
--

ALTER TABLE ONLY tests_pretrunc
    ADD CONSTRAINT tests_pkey PRIMARY KEY (index);


--
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

\connect template1

SET default_transaction_read_only = off;

--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: template1; Type: COMMENT; Schema: -; Owner: postgres
--

COMMENT ON DATABASE template1 IS 'default template for new databases';


--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


--
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

--
-- PostgreSQL database cluster dump complete
--

