/* -- LICENSE BEGIN --
 *
 * Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
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
#include "fsutil.h"
#include "subprocess.h"

#include <algorithm>
#include <memory>

namespace fsutil {
  // import all fsutil names into this new namespace
  using namespace ::flit::fsutil;
} // end of namespace fsutil

namespace {

bool string_startswith(const std::string main, const std::string needle) {
  if (main.size() < needle.size()) { return false; }
  if (main.size() == needle.size()) { return main == needle; }
  return main.substr(0, needle.size()) == needle;
}

bool string_endswith(const std::string main, const std::string needle) {
  if (main.size() < needle.size()) { return false; }
  if (main.size() == needle.size()) { return main == needle; }
  return main.substr(main.size() - needle.size(), needle.size()) == needle;
}

// trim from start
std::string ltrim(std::string s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
          [](int ch) { return !std::isspace(ch); }));
  return s;
}

// trim from end
std::string rtrim(std::string s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](int ch) { return !std::isspace(ch); }).base(),
          s.end());
  return s;
}

// trim from both ends
std::string trim(std::string s) {
  return ltrim(rtrim(s));
}

std::vector<std::string> splitlines(const std::string &tosplit) {
  std::vector<std::string> lines;
  std::string line;
  std::istringstream input(tosplit);
  while (std::getline(input, line)) {
    lines.push_back(line);
  }
  return lines;
}

void verify_listdir(const std::string &dir,
                    std::vector<std::string> expected)
{
  auto listing = fsutil::listdir(dir);
  std::sort(listing.begin(), listing.end());
  std::sort(expected.begin(), expected.end());
  TH_EQUAL(expected, listing);
}

void tst_string_startswith() {
  TH_VERIFY(!string_startswith("michae", "michael"));
  TH_VERIFY( string_startswith("michael", "michael"));
  TH_VERIFY( string_startswith("", ""));
  TH_VERIFY( string_startswith("a", ""));
  TH_VERIFY(!string_startswith("", "b"));
  TH_VERIFY( string_startswith("michael bentley", "michael"));
  TH_VERIFY(!string_startswith("michael bentley", "bentley"));
}
TH_REGISTER(tst_string_startswith);

void tst_string_endswith() {
  TH_VERIFY(!string_endswith("michae", "michael"));
  TH_VERIFY( string_endswith("michael", "michael"));
  TH_VERIFY( string_endswith("", ""));
  TH_VERIFY( string_endswith("a", ""));
  TH_VERIFY(!string_endswith("", "b"));
  TH_VERIFY( string_endswith("michael bentley", "bentley"));
  TH_VERIFY(!string_endswith("michael bentley", "michael"));
}
TH_REGISTER(tst_string_endswith);

void tst_splitlines() {
  using v = std::vector<std::string>;

  v expected = v{};
  TH_EQUAL(expected, splitlines(""));

  expected = v{"x"};
  TH_EQUAL(expected, splitlines("x"));

  expected = v{"x", "y", "", ""};
  TH_EQUAL(expected, splitlines("x\ny\n\n\n"));

  expected = v{"", "", "", "x"};
  TH_EQUAL(expected, splitlines("\n\n\nx"));
}
TH_REGISTER(tst_splitlines);

void tst_verify_listdir_exists() {
  fsutil::TempDir tempdir;

  // check that this fails
  TH_THROWS(verify_listdir(tempdir.name(), {"hello"}), th::AssertionError);

  verify_listdir(tempdir.name(), {});

  fsutil::touch(fsutil::join(tempdir.name(), "hello"));
  verify_listdir(tempdir.name(), {"hello"});

  fsutil::touch(fsutil::join(tempdir.name(), "apple of my eye"));
  verify_listdir(tempdir.name(), {"hello", "apple of my eye"});
}
TH_REGISTER(tst_verify_listdir_exists);

void tst_verify_listdir_doesnt_exist() {
  TH_THROWS(verify_listdir("/does/not/exist", {}), std::ios_base::failure);
}
TH_REGISTER(tst_verify_listdir_doesnt_exist);

void tst_ltrim() {
  TH_EQUAL("",        ltrim(""));
  TH_EQUAL("",        ltrim("  \t\n"));
  TH_EQUAL("abc  \n", ltrim("abc  \n"));
  TH_EQUAL("abc",     ltrim("\t\n   \nabc"));
  TH_EQUAL("abc  \n", ltrim("\t\n   \nabc  \n"));
  TH_EQUAL("a b c\n", ltrim("\na b c\n"));
}
TH_REGISTER(tst_ltrim);

void tst_rtrim() {
  TH_EQUAL("",             rtrim(""));
  TH_EQUAL("",             rtrim("  \t\n"));
  TH_EQUAL("abc",          rtrim("abc  \n"));
  TH_EQUAL("\t\n   \nabc", rtrim("\t\n   \nabc"));
  TH_EQUAL("\t\n   \nabc", rtrim("\t\n   \nabc  \n"));
  TH_EQUAL("\na b c",      rtrim("\na b c\n"));
}
TH_REGISTER(tst_rtrim);

void tst_trim() {
  TH_EQUAL("",      trim(""));
  TH_EQUAL("",      trim("  \t\n"));
  TH_EQUAL("abc",   trim("abc  \n"));
  TH_EQUAL("abc",   trim("\t\n   \nabc"));
  TH_EQUAL("abc",   trim("\t\n   \nabc  \n"));
  TH_EQUAL("a b c", trim("\na b c\n"));
}
TH_REGISTER(tst_trim);

} // end of unnamed namespace

namespace tst_functions {

void tst_join() {
  TH_EQUAL("path/to/my/object/file with spaces.txt",
           fsutil::join("path/to/my/object/file with spaces.txt"));
  TH_EQUAL("path/to/my/object/file with spaces.txt",
           fsutil::join("path/to/my/object", "file with spaces.txt"));
  TH_EQUAL("path/to/my/object/file with spaces.txt",
           fsutil::join("path/to", "my/object", "file with spaces.txt"));
  TH_EQUAL("path/to/my/object/file with spaces.txt",
           fsutil::join("path/to", "my", "object", "file with spaces.txt"));
  TH_EQUAL("path/to/my/object/file with spaces.txt",
           fsutil::join("path", "to", "my", "object", "file with spaces.txt"));
}
TH_REGISTER(tst_join);

void tst_readfile_filename() {
  fsutil::TempFile f;
  std::string content = "My string value";
  f.out << content;
  f.out.flush();
  TH_EQUAL(content, fsutil::readfile(f.name));
}
TH_REGISTER(tst_readfile_filename);

void tst_readfile_filename_doesnt_exist() {
  TH_THROWS(fsutil::readfile("/does/not/exist.txt"), std::ios_base::failure);
}
TH_REGISTER(tst_readfile_filename_doesnt_exist);

void tst_readfile_filepointer() {
  fsutil::FileCloser temporary_file(tmpfile());
  std::string content = "My string value";
  fprintf(temporary_file.file, "%s", content.c_str());
  fflush(temporary_file.file);
  TH_EQUAL(content, fsutil::readfile(temporary_file.file));
}
TH_REGISTER(tst_readfile_filepointer);

void tst_listdir() {
  fsutil::TempDir dir;
  auto listing = fsutil::listdir(dir.name());
  TH_EQUAL(std::vector<std::string> {}, listing);

  std::vector<std::string> expected_listing {
    "file1.txt",
    "file2.txt",
    "file3.txt",
    "file4.txt",
  };
  for (auto &fname : expected_listing) {
    fsutil::touch(fsutil::join(dir.name(), fname));
  }

  listing = fsutil::listdir(dir.name());
  std::sort(listing.begin(), listing.end());

  TH_EQUAL(expected_listing, listing);
}
TH_REGISTER(tst_listdir);

void tst_printdir_exists() {
  // a lambda function to verify the string output of printdir()
  auto verify_printdir = [] (
      const std::string &directory,
      const std::vector<std::string> &expected_contents)
  {
    // capture standard out
    std::ostringstream captured;
    fsutil::StreamBufReplace replacer(std::cout, captured);

    // call function under test
    fsutil::printdir(directory);

    // grab the output lines
    auto lines = splitlines(captured.str());
    std::sort(lines.begin() + 1, lines.end());

    // extend the expected contents to include the first line and sort
    // we also need to prepend contents with an indent of 2
    std::vector<std::string> copy;
    copy.push_back("Directory contents for " + directory);
    copy.insert(copy.end(), expected_contents.begin(),
                expected_contents.end());
    std::transform(copy.begin() + 1, copy.end(), copy.begin() + 1,
                   [] (const std::string &x) { return "  " + x; });
    std::sort(copy.begin() + 1, copy.end());

    TH_EQUAL(copy, lines);
  };

  // create a directory to list
  fsutil::TempDir tempdir;
  verify_printdir(tempdir.name(), { "./", "../" });

  // add some files to the directory
  fsutil::touch(fsutil::join(tempdir.name(), "file1.txt"));
  fsutil::touch(fsutil::join(tempdir.name(), "file2.txt"));
  fsutil::touch(fsutil::join(tempdir.name(), "file3.txt"));
  fsutil::touch(fsutil::join(tempdir.name(), "file4.txt"));
  fsutil::mkdir(fsutil::join(tempdir.name(), "dir1"), 0700);
  fsutil::mkdir(fsutil::join(tempdir.name(), "dir2"), 0700);
  fsutil::touch(fsutil::join(tempdir.name(), "dir1", "myfile1.tex"));
  fsutil::touch(fsutil::join(tempdir.name(), "dir1", "myfile2.tex"));

  verify_printdir(tempdir.name(),
                  {"./", "../", "file1.txt", "file2.txt", "file3.txt",
                   "file4.txt", "dir1/", "dir2/"});
  verify_printdir(fsutil::join(tempdir.name(), "dir1"),
                  {"./", "../", "myfile1.tex", "myfile2.tex"});
  verify_printdir(fsutil::join(tempdir.name(), "dir2"), {"./", "../"});
}
TH_REGISTER(tst_printdir_exists);

void tst_printdir_doesnt_exist() {
  TH_THROWS(fsutil::printdir("/dir/does/not/exist"), std::ios_base::failure);
}
TH_REGISTER(tst_printdir_doesnt_exist);

void tst_rec_rmdir_exists() {
  using fsutil::TempDir;
  using fsutil::mkdir;
  using fsutil::touch;
  using fsutil::join;
  using fsutil::rec_rmdir;

  // make a directory structure
  TempDir tempdir;
  mkdir(join(tempdir.name(), "dir1"));
  mkdir(join(tempdir.name(), "dir1", "subdir1"));
  mkdir(join(tempdir.name(), "dir1", "subdir2"));
  mkdir(join(tempdir.name(), "dir1", "subdir3"));
  mkdir(join(tempdir.name(), "dir2"));
  touch(join(tempdir.name(), "superfile1.txt"));
  touch(join(tempdir.name(), "superfile2.txt"));
  touch(join(tempdir.name(), "dir1", "file1.txt"));
  touch(join(tempdir.name(), "dir1", "file2.txt"));
  touch(join(tempdir.name(), "dir1", "file3.txt"));
  touch(join(tempdir.name(), "dir1", "file4.txt"));
  touch(join(tempdir.name(), "dir1", "subdir1", "subfile1-1.txt"));
  touch(join(tempdir.name(), "dir1", "subdir1", "subfile1-2.txt"));
  touch(join(tempdir.name(), "dir1", "subdir2", "subfile2-1.txt"));

  verify_listdir(tempdir.name(),
                 {"dir1", "dir2", "superfile1.txt", "superfile2.txt"});
  verify_listdir(join(tempdir.name(), "dir1"),
                 {"subdir1", "subdir2", "subdir3", "file1.txt", "file2.txt",
                  "file3.txt", "file4.txt"});
  verify_listdir(join(tempdir.name(), "dir1", "subdir1"),
                 {"subfile1-1.txt", "subfile1-2.txt"});
  verify_listdir(join(tempdir.name(), "dir1", "subdir2"), {"subfile2-1.txt"});
  verify_listdir(join(tempdir.name(), "dir2"), {});

  // remove an empty directory
  rec_rmdir(join(tempdir.name(), "dir2"));
  verify_listdir(tempdir.name(), {"dir1", "superfile1.txt", "superfile2.txt"});

  // remove one that has some things
  rec_rmdir(join(tempdir.name(), "dir1", "subdir2"));
  verify_listdir(join(tempdir.name(), "dir1"),
                 {"subdir1", "subdir3", "file1.txt", "file2.txt", "file3.txt",
                  "file4.txt"});

  // remove a directory that also has non-empty subdirectories
  rec_rmdir(join(tempdir.name(), "dir1"));
  verify_listdir(tempdir.name(), {"superfile1.txt", "superfile2.txt"});
}
TH_REGISTER(tst_rec_rmdir_exists);

void tst_rec_rmdir_doesnt_exist() {
  TH_THROWS(fsutil::rec_rmdir("/dir/does/not/exist"), std::ios_base::failure);
}
TH_REGISTER(tst_rec_rmdir_doesnt_exist);

void tst_rec_rmdir_on_file() {
  fsutil::TempFile tmpfile;
  TH_THROWS(fsutil::rec_rmdir(tmpfile.name), std::ios_base::failure);
}
TH_REGISTER(tst_rec_rmdir_on_file);

void tst_mkdir() {
  fsutil::TempDir tempdir;
  verify_listdir(tempdir.name(), {});
  fsutil::mkdir(fsutil::join(tempdir.name(), "dir1"));
  verify_listdir(tempdir.name(), {"dir1"});
  verify_listdir(fsutil::join(tempdir.name(), "dir1"), {});
}
TH_REGISTER(tst_mkdir);

void tst_rmdir_empty() {
  fsutil::TempDir tempdir;
  fsutil::mkdir(fsutil::join(tempdir.name(), "dir1"));
  verify_listdir(tempdir.name(), {"dir1"});
  fsutil::rmdir(fsutil::join(tempdir.name(), "dir1"));
  verify_listdir(tempdir.name(), {});
}
TH_REGISTER(tst_rmdir_empty);

void tst_rmdir_full() {
  fsutil::TempDir tempdir;
  fsutil::mkdir(fsutil::join(tempdir.name(), "dir1"));
  fsutil::touch(fsutil::join(tempdir.name(), "dir1", "file.txt"));
  verify_listdir(tempdir.name(), {"dir1"});
  verify_listdir(fsutil::join(tempdir.name(), "dir1"), {"file.txt"});
  TH_THROWS(fsutil::rmdir(fsutil::join(tempdir.name(), "dir1")),
            std::ios_base::failure);
  verify_listdir(tempdir.name(), {"dir1"});
  verify_listdir(fsutil::join(tempdir.name(), "dir1"), {"file.txt"});
}
TH_REGISTER(tst_rmdir_full);

void tst_rmdir_doesnt_exist() {
  TH_THROWS(fsutil::rmdir("/does/not/exist"), std::ios_base::failure);
}
TH_REGISTER(tst_rmdir_doesnt_exist);

void tst_curdir() {
  fsutil::TempDir tempdir1;
  fsutil::TempDir tempdir2;
  auto originaldir = fsutil::curdir();
  {
    fsutil::PushDir pd1(tempdir1.name());
    TH_EQUAL(tempdir1.name(), fsutil::curdir());
    {
      fsutil::PushDir pd2(tempdir2.name());
      TH_EQUAL(tempdir2.name(), fsutil::curdir());
    }
    TH_EQUAL(tempdir1.name(), fsutil::curdir());
  }
  TH_EQUAL(originaldir, fsutil::curdir());
}
TH_REGISTER(tst_curdir);

void tst_chdir_exists() {
  fsutil::TempDir tempdir;
  auto originaldir = fsutil::curdir();
  try {
    fsutil::chdir(tempdir.name());
    TH_EQUAL(tempdir.name(), fsutil::curdir());
  } catch (...) {
    fsutil::chdir(originaldir);
    throw;
  }
  fsutil::chdir(originaldir);
  TH_EQUAL(originaldir, fsutil::curdir());
}
TH_REGISTER(tst_chdir_exists);

void tst_chdir_doesnt_exist() {
  TH_THROWS(fsutil::chdir("/does/not/exist"), std::ios_base::failure);
}
TH_REGISTER(tst_chdir_doesnt_exist);

void tst_chdir_on_file() {
  fsutil::TempFile tempfile;
  TH_THROWS(fsutil::chdir(tempfile.name), std::ios_base::failure);
}
TH_REGISTER(tst_chdir_on_file);

void tst_which_defaultpath_empty() {
  TH_THROWS(fsutil::which(""), std::ios_base::failure);
}
TH_REGISTER(tst_which_defaultpath_empty);

void tst_which_defaultpath_absolute() {
  fsutil::TempDir tempdir;
  TH_THROWS(fsutil::which(fsutil::join(tempdir.name(), "does/not/exist")),
            std::ios_base::failure);

  fsutil::TempFile tempfile;
  TH_EQUAL(tempfile.name, fsutil::which(tempfile.name));

  // directory fails even if it exists
  TH_THROWS(fsutil::which(tempdir.name()), std::ios_base::failure);
}
TH_REGISTER(tst_which_defaultpath_absolute);

void tst_which_defaultpath_relative() {
  fsutil::TempDir tempdir;
  fsutil::PushDir pusher(tempdir.name());

  TH_THROWS(fsutil::which("does/not/exist"), std::ios_base::failure);

  fsutil::mkdir("does");
  fsutil::touch("does/exist");
  TH_EQUAL(fsutil::join(tempdir.name(), "does/exist"),
           fsutil::which("does/exist"));

  // in current directory
  fsutil::touch("exists");
  TH_EQUAL(fsutil::join(tempdir.name(), "exists"), fsutil::which("./exists"));
  TH_THROWS(fsutil::which("exists"), std::ios_base::failure);

  // directory fails even if it exists
  TH_THROWS(fsutil::which("./does"), std::ios_base::failure);
}
TH_REGISTER(tst_which_defaultpath_relative);

void tst_which_defaultpath_tofind() {
  fsutil::TempDir tempdir;
  fsutil::PushDir pusher(tempdir.name());

  TH_THROWS(fsutil::which("does-not-exist"), std::ios_base::failure);

  auto bash_path = trim(flit::call_with_output("which bash").out);
  TH_EQUAL(bash_path, fsutil::which("bash"));
}
TH_REGISTER(tst_which_defaultpath_tofind);

void tst_which_givenpath_empty() {
  fsutil::TempDir path_piece1, path_piece2;
  auto path = path_piece1.name() + ":" + path_piece2.name();

  TH_THROWS(fsutil::which("", path), std::ios_base::failure);
}
TH_REGISTER(tst_which_givenpath_empty);

void tst_which_givenpath_absolute() {
  fsutil::TempDir path_piece1, path_piece2;
  auto path = path_piece1.name() + ":" + path_piece2.name();

  fsutil::TempDir tempdir;
  TH_THROWS(fsutil::which(fsutil::join(tempdir.name(), "does/not/exist"), path),
            std::ios_base::failure);

  fsutil::TempFile tempfile;
  TH_EQUAL(tempfile.name, fsutil::which(tempfile.name, path));

  // directory fails even if it exists
  TH_THROWS(fsutil::which(tempdir.name(), path), std::ios_base::failure);
}
TH_REGISTER(tst_which_givenpath_absolute);

void tst_which_givenpath_relative() {
  fsutil::TempDir path_piece1, path_piece2;
  auto path = path_piece1.name() + ":" + path_piece2.name();

  fsutil::TempDir tempdir;
  fsutil::PushDir pusher(tempdir.name());

  TH_THROWS(fsutil::which("does/not/exist", path), std::ios_base::failure);

  fsutil::mkdir("does");
  fsutil::touch("does/exist");
  TH_EQUAL(fsutil::join(tempdir.name(), "does/exist"),
           fsutil::which("does/exist", path));

  // in current directory
  fsutil::touch("exists");
  TH_EQUAL(fsutil::join(tempdir.name(), "exists"), fsutil::which("./exists", path));
  TH_THROWS(fsutil::which("exists", path), std::ios_base::failure);

  // directory fails even if it exists
  TH_THROWS(fsutil::which("./does", path), std::ios_base::failure);
}
TH_REGISTER(tst_which_givenpath_relative);

void tst_which_givenpath_tofind() {
  fsutil::TempDir path_piece1, path_piece2;
  auto path = path_piece1.name() + ":" + path_piece2.name();

  fsutil::TempDir tempdir;
  fsutil::PushDir pusher(tempdir.name());

  TH_THROWS(fsutil::which("does-not-exist", path), std::ios_base::failure);

  std::string file1_name("first-file");
  std::string file2_name("second-file");
  auto file1 = fsutil::join(path_piece1.name(), file1_name);
  auto file2 = fsutil::join(path_piece2.name(), file2_name);
  fsutil::touch(file1);
  fsutil::touch(file2);

  TH_EQUAL(file1, fsutil::which(file1_name, path));
  TH_EQUAL(file2, fsutil::which(file2_name, path));

  // check that the first one is returned if there are duplicates
  path = ".:" + path;
  fsutil::touch(file1_name);
  std::cout << "expected: '" << fsutil::join(tempdir.name(), file1_name)
            << "'\n"
            << "actual:   '" << fsutil::which(file1_name, path) << "'\n";
  std::cout.flush();
  TH_EQUAL(fsutil::join(tempdir.name(), file1_name),
           fsutil::which(file1_name, path));

  // check that directories do not match, even if they are duplicates
  fsutil::mkdir(fsutil::join(path_piece1.name(), "mydirectory"));
  TH_THROWS(fsutil::which("mydirectory", path), std::ios_base::failure);
  fsutil::mkdir(fsutil::join(path_piece1.name(), file2_name));
  TH_EQUAL(file2, fsutil::which(file2_name, path));
}
TH_REGISTER(tst_which_givenpath_tofind);

} // end of namespace tst_functions

namespace tst_PushDir {

void tst_PushDir_constructor_existing_dir() {
  fsutil::TempDir temp;
  std::string curdir = fsutil::curdir();
  fsutil::PushDir pd(temp.name());
  TH_EQUAL(curdir, pd.previous_dir());
  TH_EQUAL(temp.name(), fsutil::curdir());
}
TH_REGISTER(tst_PushDir_constructor_existing_dir);

void tst_PushDir_constructor_missing_dir() {
  TH_THROWS(fsutil::PushDir pd("/does/not/exist"), std::ios_base::failure);
}
TH_REGISTER(tst_PushDir_constructor_missing_dir);

void tst_PushDir_destructor_existing_dir() {
  fsutil::TempDir temp1;
  fsutil::TempDir temp2;
  auto curdir = fsutil::curdir();
  {
    fsutil::PushDir pd1(temp1.name());
    TH_EQUAL(curdir, pd1.previous_dir());
    {
      fsutil::PushDir pd2(temp2.name());
      TH_EQUAL(temp1.name(), pd2.previous_dir());
      TH_EQUAL(temp2.name(), fsutil::curdir());
    }
    TH_EQUAL(temp1.name(), fsutil::curdir());
  }
  TH_EQUAL(curdir, fsutil::curdir());
}
TH_REGISTER(tst_PushDir_destructor_existing_dir);

void tst_PushDir_destructor_missing_dir() {
  auto curdir = fsutil::curdir();
  std::ostringstream captured;
  // capture stderr and assert on its output
  fsutil::StreamBufReplace buffer_replacer(std::cerr, captured);
  {
    fsutil::TempDir temp2;
    std::unique_ptr<fsutil::PushDir> pd1;
    {
      std::unique_ptr<fsutil::PushDir> pd2;
      {
        fsutil::TempDir temp1;

        pd1.reset(new fsutil::PushDir(temp1.name()));
        TH_EQUAL(curdir, pd1->previous_dir());
        TH_EQUAL(temp1.name(), fsutil::curdir());

        pd2.reset(new fsutil::PushDir(temp2.name()));
        TH_EQUAL(temp1.name(), pd2->previous_dir());
        TH_EQUAL(temp2.name(), fsutil::curdir());
      } // deletes temp1
      TH_EQUAL(temp2.name(), fsutil::curdir());
      TH_EQUAL("", captured.str());
    } // deletes pd2, with temp1.name() not existing
    // should be unable to go to temp1.name() since it doesn't exist
    // but no exception since it is from a destructor
    TH_VERIFY(string_startswith(captured.str(),
                                "ios_base error: Could not change directory"));
    TH_EQUAL(temp2.name(), fsutil::curdir());
  } // deletes pd1 and temp2
  TH_EQUAL(curdir, fsutil::curdir());
  // make sure there is at most one newline and it is at the end
  TH_VERIFY(captured.str().find('\n') == captured.str().size() - 1);
}
TH_REGISTER(tst_PushDir_destructor_missing_dir);

} // end of namespace tst_PushDir

namespace tst_TempFile {

void tst_TempFile() {
  std::string filename;
  std::string contents = "hello, my name is Michael\n...\nBentley, that is.";
  {
    fsutil::TempFile tempfile;
    filename = tempfile.name;
    tempfile.out << contents;
    tempfile.out.flush();
    TH_EQUAL(contents, fsutil::readfile(filename));
  }
  // make sure the file has been deleted.
  TH_THROWS(fsutil::readfile(filename), std::ios_base::failure);
}
TH_REGISTER(tst_TempFile);

} // end of namespace tst_TempFile

namespace tst_TempDir {

void tst_TempDir() {
  std::string tmpdir_name;
  std::string filename = "myfile.txt";
  std::string filepath;

  {
    fsutil::TempDir tmpdir;
    tmpdir_name = tmpdir.name();
    filepath = fsutil::join(tmpdir_name, filename);
    fsutil::touch(filepath);
    auto listing = fsutil::listdir(tmpdir_name);
    std::vector<std::string> expected_listing { filename };
    TH_EQUAL(expected_listing, listing);
    TH_EQUAL("", fsutil::readfile(filepath));
  }

  // check that the directory no longer exists
  TH_THROWS(fsutil::listdir(tmpdir_name), std::ios_base::failure);

  // check that the file no longer exists
  TH_THROWS(fsutil::readfile(filepath), std::ios_base::failure);
}
TH_REGISTER(tst_TempDir);

} // end of namespace tst_TempDir

namespace tst_TinyDir {

void tst_TinyDir_constructor_doesnt_exist() {
  TH_THROWS(fsutil::TinyDir("/dir/does/not/exist"), std::ios_base::failure);
}
TH_REGISTER(tst_TinyDir_constructor_doesnt_exist);

void tst_TinyDir_iterate() {
  fsutil::TempDir tempdir;
  fsutil::touch(fsutil::join(tempdir.name(), "file1.txt"));
  fsutil::touch(fsutil::join(tempdir.name(), "file2.txt"));
  fsutil::touch(fsutil::join(tempdir.name(), "file3.txt"));
  std::vector<std::string> listing;
  std::vector<std::string> expected_listing {
    ".",
    "..",
    "file1.txt",
    "file2.txt",
    "file3.txt",
  };
  fsutil::TinyDir tinydir(tempdir.name());
  while (tinydir.hasnext()) {
    auto file = tinydir.readfile();
    tinydir.nextfile();
    listing.push_back(file.name);
  }
  std::sort(listing.begin(), listing.end());
  std::sort(expected_listing.begin(), expected_listing.end());
  TH_EQUAL(expected_listing, listing);
}
TH_REGISTER(tst_TinyDir_iterate);

void tst_TinyDir_iterator() {
  fsutil::TempDir tempdir;
  fsutil::touch(fsutil::join(tempdir.name(), "file1.txt"));
  fsutil::touch(fsutil::join(tempdir.name(), "file2.txt"));
  fsutil::touch(fsutil::join(tempdir.name(), "file3.txt"));
  std::vector<std::string> listing;
  std::vector<std::string> expected_listing {
    ".",
    "..",
    "file1.txt",
    "file2.txt",
    "file3.txt",
  };
  fsutil::TinyDir tinydir(tempdir.name());
  for (auto file : tinydir) {
    listing.push_back(file.name);
  }
  std::sort(listing.begin(), listing.end());
  std::sort(expected_listing.begin(), expected_listing.end());
  TH_EQUAL(expected_listing, listing);
}
TH_REGISTER(tst_TinyDir_iterator);

} // end of namespace tst_TinyDir

namespace tst_FileCloser {

void tst_FileCloser() {
  fsutil::TempDir temp_dir;
  std::string content = "hello world!";
  std::string fname = fsutil::join(temp_dir.name(), "tmp.txt");
  FILE* temp_file = fopen(fname.c_str(), "w");
  {
    fsutil::FileCloser closer(temp_file);
    fprintf(temp_file, "%s", content.c_str());
  }
  TH_EQUAL(content, fsutil::readfile(fname));
  // Unfortunately, there is no way to check that the FILE* pointer was closed
  // without stubbing.  This is because it is undefined behavior if you use the
  // FILE* pointer in any way after fclose().
}
TH_REGISTER(tst_FileCloser);

} // end of namespace tst_FileCloser

namespace tst_FdReplace {

void tst_FdReplace() {
  fsutil::FileCloser t1(tmpfile());
  fsutil::FileCloser t2(tmpfile());
  fprintf(t1.file, "hi there");
  {
    fsutil::FdReplace replacer(t1.file, t2.file);
    fprintf(t1.file, "hello ");
    fflush(t1.file);
    fprintf(t2.file, "world");
    fflush(t2.file);
    TH_EQUAL("hello world", fsutil::readfile(t1.file));
    TH_EQUAL("hello world", fsutil::readfile(t2.file));
  }
  TH_EQUAL("hi there", fsutil::readfile(t1.file));
  TH_EQUAL("hello world", fsutil::readfile(t2.file));
}
TH_REGISTER(tst_FdReplace);

void tst_FdReplace_nullptr() {
  fsutil::FileCloser t1(tmpfile());
  fsutil::FileCloser t2(tmpfile());
  TH_THROWS(fsutil::FdReplace(nullptr, nullptr), std::ios_base::failure);
  TH_THROWS(fsutil::FdReplace(t1.file, nullptr), std::ios_base::failure);
  TH_THROWS(fsutil::FdReplace(nullptr, t1.file), std::ios_base::failure);
}

} // end of namespace tst_FdReplace

namespace tst_StreamBufReplace {

void tst_StreamBufReplace() {
  std::stringstream s1;
  std::stringstream s2;
  s1 << "hi there";
  TH_EQUAL(s1.str(), "hi there");
  {
    fsutil::StreamBufReplace stream_replacer(s1, s2);
    s1 << "hello ";
    TH_EQUAL(s2.str(), "hello ");

    s2 << "world";
    TH_EQUAL(s2.str(), "hello world");
  }
  TH_EQUAL(s1.str(), "hi there");
  TH_EQUAL(s2.str(), "hello world");
}
TH_REGISTER(tst_StreamBufReplace);

} // end of namespace tst_StreamBufReplace
