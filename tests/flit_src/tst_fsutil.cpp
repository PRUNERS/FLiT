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

#include <flit/fsutil.h>
#include <flit/subprocess.h>
#include <flit/fsutil.cpp>

#include <algorithm>
#include <memory>

namespace {

struct PopDir {
  std::string originaldir;
  PopDir() : originaldir(flit::curdir()) {}
  ~PopDir() { flit::chdir(originaldir); }
} popper;

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
  auto listing = flit::listdir(dir);
  std::sort(listing.begin(), listing.end());
  std::sort(expected.begin(), expected.end());
  TH_EQUAL(expected, listing);
}

TH_TEST(tst_string_startswith) {
  TH_VERIFY(!string_startswith("michae", "michael"));
  TH_VERIFY( string_startswith("michael", "michael"));
  TH_VERIFY( string_startswith("", ""));
  TH_VERIFY( string_startswith("a", ""));
  TH_VERIFY(!string_startswith("", "b"));
  TH_VERIFY( string_startswith("michael bentley", "michael"));
  TH_VERIFY(!string_startswith("michael bentley", "bentley"));
}

TH_TEST(tst_string_endswith) {
  TH_VERIFY(!string_endswith("michae", "michael"));
  TH_VERIFY( string_endswith("michael", "michael"));
  TH_VERIFY( string_endswith("", ""));
  TH_VERIFY( string_endswith("a", ""));
  TH_VERIFY(!string_endswith("", "b"));
  TH_VERIFY( string_endswith("michael bentley", "bentley"));
  TH_VERIFY(!string_endswith("michael bentley", "michael"));
}

TH_TEST(tst_splitlines) {
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

TH_TEST(tst_verify_listdir_exists) {
  flit::TempDir tempdir;

  // check that this fails
  TH_THROWS(verify_listdir(tempdir.name(), {"hello"}), th::AssertionError);

  verify_listdir(tempdir.name(), {});

  flit::touch(flit::join(tempdir.name(), "hello"));
  verify_listdir(tempdir.name(), {"hello"});

  flit::touch(flit::join(tempdir.name(), "apple of my eye"));
  verify_listdir(tempdir.name(), {"hello", "apple of my eye"});
}

TH_TEST(tst_verify_listdir_doesnt_exist) {
  TH_THROWS(verify_listdir("/does/not/exist", {}), std::ios_base::failure);
}

TH_TEST(tst_ltrim) {
  TH_EQUAL("",        ltrim(""));
  TH_EQUAL("",        ltrim("  \t\n"));
  TH_EQUAL("abc  \n", ltrim("abc  \n"));
  TH_EQUAL("abc",     ltrim("\t\n   \nabc"));
  TH_EQUAL("abc  \n", ltrim("\t\n   \nabc  \n"));
  TH_EQUAL("a b c\n", ltrim("\na b c\n"));
}

TH_TEST(tst_rtrim) {
  TH_EQUAL("",             rtrim(""));
  TH_EQUAL("",             rtrim("  \t\n"));
  TH_EQUAL("abc",          rtrim("abc  \n"));
  TH_EQUAL("\t\n   \nabc", rtrim("\t\n   \nabc"));
  TH_EQUAL("\t\n   \nabc", rtrim("\t\n   \nabc  \n"));
  TH_EQUAL("\na b c",      rtrim("\na b c\n"));
}

TH_TEST(tst_trim) {
  TH_EQUAL("",      trim(""));
  TH_EQUAL("",      trim("  \t\n"));
  TH_EQUAL("abc",   trim("abc  \n"));
  TH_EQUAL("abc",   trim("\t\n   \nabc"));
  TH_EQUAL("abc",   trim("\t\n   \nabc  \n"));
  TH_EQUAL("a b c", trim("\na b c\n"));
}

} // end of unnamed namespace

namespace tst_functions {

TH_TEST(tst_join) {
  TH_EQUAL("path/to/my/object/file with spaces.txt",
           flit::join("path/to/my/object/file with spaces.txt"));
  TH_EQUAL("path/to/my/object/file with spaces.txt",
           flit::join("path/to/my/object", "file with spaces.txt"));
  TH_EQUAL("path/to/my/object/file with spaces.txt",
           flit::join("path/to", "my/object", "file with spaces.txt"));
  TH_EQUAL("path/to/my/object/file with spaces.txt",
           flit::join("path/to", "my", "object", "file with spaces.txt"));
  TH_EQUAL("path/to/my/object/file with spaces.txt",
           flit::join("path", "to", "my", "object", "file with spaces.txt"));
}

TH_TEST(tst_readfile_filename) {
  flit::TempFile f;
  std::string content = "My string value";
  f.out << content;
  f.out.flush();
  TH_EQUAL(content, flit::readfile(f.name));
}

TH_TEST(tst_readfile_filename_doesnt_exist) {
  TH_THROWS(flit::readfile("/does/not/exist.txt"), std::ios_base::failure);
}

TH_TEST(tst_readfile_filepointer) {
  flit::FileCloser temporary_file(tmpfile());
  std::string content = "My string value";
  fprintf(temporary_file.file, "%s", content.c_str());
  fflush(temporary_file.file);
  TH_EQUAL(content, flit::readfile(temporary_file.file));
}

TH_TEST(tst_listdir) {
  flit::TempDir dir;
  auto listing = flit::listdir(dir.name());
  TH_EQUAL(std::vector<std::string> {}, listing);

  std::vector<std::string> expected_listing {
    "file1.txt",
    "file2.txt",
    "file3.txt",
    "file4.txt",
  };
  for (auto &fname : expected_listing) {
    flit::touch(flit::join(dir.name(), fname));
  }

  listing = flit::listdir(dir.name());
  std::sort(listing.begin(), listing.end());

  TH_EQUAL(expected_listing, listing);
}

TH_TEST(tst_printdir_exists) {
  // a lambda function to verify the string output of printdir()
  auto verify_printdir = [] (
      const std::string &directory,
      const std::vector<std::string> &expected_contents)
  {
    // capture standard out
    std::ostringstream captured;
    flit::StreamBufReplace replacer(std::cout, captured);

    // call function under test
    flit::printdir(directory);

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
  flit::TempDir tempdir;
  verify_printdir(tempdir.name(), { "./", "../" });

  // add some files to the directory
  flit::touch(flit::join(tempdir.name(), "file1.txt"));
  flit::touch(flit::join(tempdir.name(), "file2.txt"));
  flit::touch(flit::join(tempdir.name(), "file3.txt"));
  flit::touch(flit::join(tempdir.name(), "file4.txt"));
  flit::mkdir(flit::join(tempdir.name(), "dir1"), 0700);
  flit::mkdir(flit::join(tempdir.name(), "dir2"), 0700);
  flit::touch(flit::join(tempdir.name(), "dir1", "myfile1.tex"));
  flit::touch(flit::join(tempdir.name(), "dir1", "myfile2.tex"));

  verify_printdir(tempdir.name(),
                  {"./", "../", "file1.txt", "file2.txt", "file3.txt",
                   "file4.txt", "dir1/", "dir2/"});
  verify_printdir(flit::join(tempdir.name(), "dir1"),
                  {"./", "../", "myfile1.tex", "myfile2.tex"});
  verify_printdir(flit::join(tempdir.name(), "dir2"), {"./", "../"});
}

TH_TEST(tst_printdir_doesnt_exist) {
  TH_THROWS(flit::printdir("/dir/does/not/exist"), std::ios_base::failure);
}

TH_TEST(tst_rec_rmdir_exists) {
  using flit::TempDir;
  using flit::mkdir;
  using flit::touch;
  using flit::join;
  using flit::rec_rmdir;

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

TH_TEST(tst_rec_rmdir_doesnt_exist) {
  TH_THROWS(flit::rec_rmdir("/dir/does/not/exist"), std::ios_base::failure);
}

TH_TEST(tst_rec_rmdir_on_file) {
  flit::TempFile tmpfile;
  TH_THROWS(flit::rec_rmdir(tmpfile.name), std::ios_base::failure);
}

TH_TEST(tst_mkdir) {
  flit::TempDir tempdir;
  verify_listdir(tempdir.name(), {});
  flit::mkdir(flit::join(tempdir.name(), "dir1"));
  verify_listdir(tempdir.name(), {"dir1"});
  verify_listdir(flit::join(tempdir.name(), "dir1"), {});
}

TH_TEST(tst_mkdir_parent_does_not_exist) {
  TH_THROWS(flit::mkdir(flit::join("doesnt-exist", "newdir")),
            std::ios_base::failure);
}

TH_TEST(tst_rmdir_empty) {
  flit::TempDir tempdir;
  flit::mkdir(flit::join(tempdir.name(), "dir1"));
  verify_listdir(tempdir.name(), {"dir1"});
  flit::rmdir(flit::join(tempdir.name(), "dir1"));
  verify_listdir(tempdir.name(), {});
}

TH_TEST(tst_rmdir_full) {
  flit::TempDir tempdir;
  flit::mkdir(flit::join(tempdir.name(), "dir1"));
  flit::touch(flit::join(tempdir.name(), "dir1", "file.txt"));
  verify_listdir(tempdir.name(), {"dir1"});
  verify_listdir(flit::join(tempdir.name(), "dir1"), {"file.txt"});
  TH_THROWS(flit::rmdir(flit::join(tempdir.name(), "dir1")),
            std::ios_base::failure);
  verify_listdir(tempdir.name(), {"dir1"});
  verify_listdir(flit::join(tempdir.name(), "dir1"), {"file.txt"});
}

TH_TEST(tst_rmdir_doesnt_exist) {
  TH_THROWS(flit::rmdir("/does/not/exist"), std::ios_base::failure);
}

TH_TEST(tst_curdir) {
  flit::TempDir tempdir1;
  flit::TempDir tempdir2;
  auto originaldir = flit::curdir();
  {
    flit::PushDir pd1(tempdir1.name());
    TH_EQUAL(tempdir1.name(), flit::curdir());
    {
      flit::PushDir pd2(tempdir2.name());
      TH_EQUAL(tempdir2.name(), flit::curdir());
    }
    TH_EQUAL(tempdir1.name(), flit::curdir());
  }
  TH_EQUAL(originaldir, flit::curdir());
}

TH_TEST(tst_curdir_doesnt_exist) {
  PopDir popper; // go back to current directory at the end
  {
    flit::TempDir tempdir;
    flit::chdir(tempdir.name());
  }
  TH_THROWS(flit::curdir(), std::ios_base::failure);
}

TH_TEST(tst_chdir_exists) {
  flit::TempDir tempdir;
  auto originaldir = flit::curdir();
  {
    PopDir popper; // go back to current directory at the end of scope
    flit::chdir(tempdir.name());
    TH_EQUAL(tempdir.name(), flit::curdir());
  }
  TH_EQUAL(originaldir, flit::curdir());
}

TH_TEST(tst_chdir_doesnt_exist) {
  TH_THROWS(flit::chdir("/does/not/exist"), std::ios_base::failure);
}

TH_TEST(tst_chdir_on_file) {
  flit::TempFile tempfile;
  TH_THROWS(flit::chdir(tempfile.name), std::ios_base::failure);
}

TH_TEST(tst_which_defaultpath_empty) {
  TH_THROWS(flit::which(""), std::ios_base::failure);
}

TH_TEST(tst_which_defaultpath_absolute) {
  flit::TempDir tempdir;
  TH_THROWS(flit::which(flit::join(tempdir.name(), "does/not/exist")),
            std::ios_base::failure);

  flit::TempFile tempfile;
  TH_EQUAL(tempfile.name, flit::which(tempfile.name));

  // directory fails even if it exists
  TH_THROWS(flit::which(tempdir.name()), std::ios_base::failure);
}

TH_TEST(tst_which_defaultpath_relative) {
  flit::TempDir tempdir;
  flit::PushDir pusher(tempdir.name());

  TH_THROWS(flit::which("does/not/exist"), std::ios_base::failure);

  flit::mkdir("does");
  flit::touch("does/exist");
  TH_EQUAL(flit::join(tempdir.name(), "does/exist"),
           flit::which("does/exist"));

  // in current directory
  flit::touch("exists");
  TH_EQUAL(flit::join(tempdir.name(), "exists"), flit::which("./exists"));
  TH_THROWS(flit::which("exists"), std::ios_base::failure);

  // directory fails even if it exists
  TH_THROWS(flit::which("./does"), std::ios_base::failure);
}

TH_TEST(tst_which_defaultpath_tofind) {
  flit::TempDir tempdir;
  flit::PushDir pusher(tempdir.name());

  TH_THROWS(flit::which("does-not-exist"), std::ios_base::failure);

  auto bash_path = trim(flit::call_with_output("which bash").out);
  TH_EQUAL(bash_path, flit::which("bash"));
}

TH_TEST(tst_which_givenpath_empty) {
  flit::TempDir path_piece1, path_piece2;
  auto path = path_piece1.name() + ":" + path_piece2.name();

  TH_THROWS(flit::which("", path), std::ios_base::failure);
}

TH_TEST(tst_which_givenpath_absolute) {
  flit::TempDir path_piece1, path_piece2;
  auto path = path_piece1.name() + ":" + path_piece2.name();

  flit::TempDir tempdir;
  TH_THROWS(flit::which(flit::join(tempdir.name(), "does/not/exist"), path),
            std::ios_base::failure);

  flit::TempFile tempfile;
  TH_EQUAL(tempfile.name, flit::which(tempfile.name, path));

  // directory fails even if it exists
  TH_THROWS(flit::which(tempdir.name(), path), std::ios_base::failure);
}

TH_TEST(tst_which_givenpath_relative) {
  flit::TempDir path_piece1, path_piece2;
  auto path = path_piece1.name() + ":" + path_piece2.name();

  flit::TempDir tempdir;
  flit::PushDir pusher(tempdir.name());

  TH_THROWS(flit::which("does/not/exist", path), std::ios_base::failure);

  flit::mkdir("does");
  flit::touch("does/exist");
  TH_EQUAL(flit::join(tempdir.name(), "does/exist"),
           flit::which("does/exist", path));

  // in current directory
  flit::touch("exists");
  TH_EQUAL(flit::join(tempdir.name(), "exists"), flit::which("./exists", path));
  TH_THROWS(flit::which("exists", path), std::ios_base::failure);

  // directory fails even if it exists
  TH_THROWS(flit::which("./does", path), std::ios_base::failure);
}

TH_TEST(tst_which_givenpath_tofind) {
  flit::TempDir path_piece1, path_piece2;
  auto path = path_piece1.name() + ":" + path_piece2.name();

  flit::TempDir tempdir;
  flit::PushDir pusher(tempdir.name());

  TH_THROWS(flit::which("does-not-exist", path), std::ios_base::failure);

  std::string file1_name("first-file");
  std::string file2_name("second-file");
  auto file1 = flit::join(path_piece1.name(), file1_name);
  auto file2 = flit::join(path_piece2.name(), file2_name);
  flit::touch(file1);
  flit::touch(file2);

  TH_EQUAL(file1, flit::which(file1_name, path));
  TH_EQUAL(file2, flit::which(file2_name, path));

  // check that the first one is returned if there are duplicates
  path = ".:" + path;
  flit::touch(file1_name);
  TH_EQUAL(flit::join(tempdir.name(), file1_name),
           flit::which(file1_name, path));

  // check that directories do not match, even if they are duplicates
  flit::mkdir(flit::join(path_piece1.name(), "mydirectory"));
  TH_THROWS(flit::which("mydirectory", path), std::ios_base::failure);
  flit::mkdir(flit::join(path_piece1.name(), file2_name));
  TH_EQUAL(file2, flit::which(file2_name, path));
}

TH_TEST(tst_realpath_already_absolute) {
  flit::TempDir temp;
  TH_EQUAL(flit::realpath(temp.name()), temp.name());
}

TH_TEST(tst_realpath_does_not_exist) {
  std::string nonexistent;
  {
    flit::TempDir temp;
    nonexistent = flit::join(temp.name(), "does-not-exist.txt");
  }
  TH_THROWS(flit::realpath(nonexistent), std::ios_base::failure);
}

TH_TEST(tst_realpath_vs_curdir) {
  TH_EQUAL(flit::realpath("."), flit::curdir());
}

TH_TEST(tst_realpath_from_relative_path) {
  flit::TempDir parent;
  std::string child = "child-dir";
  std::string child_abs = flit::join(parent.name(), child);
  flit::mkdir(child_abs);
  {
    flit::PushDir pusher(parent.name());
    TH_EQUAL(flit::realpath("."), parent.name());
    TH_EQUAL(flit::realpath(child), child_abs);
  }
}

TH_TEST(tst_realpath_absolute_with_relative_movements) {
  flit::TempDir parent;
  std::string child_1 = "child-dir-1";
  std::string child_2 = "child-dir-2";
  auto child_1_abs = flit::join(parent.name(), child_1);
  auto child_2_abs = flit::join(parent.name(), child_2);
  flit::mkdir(child_1_abs);
  flit::mkdir(child_2_abs);
  TH_EQUAL(flit::realpath(flit::join(child_1_abs, "..", child_1, ".")),
           child_1_abs);
  TH_EQUAL(flit::realpath(flit::join(child_1_abs, "..", child_1, "..", ".", child_2)),
           child_2_abs);
}

TH_TEST(tst_basename) {
  TH_EQUAL(flit::basename("//usr/lib"), "lib");
  TH_EQUAL(flit::basename("/usr//lib"), "lib");
  TH_EQUAL(flit::basename("/usr/lib"), "lib");
  TH_EQUAL(flit::basename("/usr///"), "usr");
  TH_EQUAL(flit::basename("/usr//"), "usr");
  TH_EQUAL(flit::basename("/usr/"), "usr");
  TH_EQUAL(flit::basename("usr"), "usr");
  TH_EQUAL(flit::basename("///"), "/");
  TH_EQUAL(flit::basename("/"), "/");
  TH_EQUAL(flit::basename("."), ".");
  TH_EQUAL(flit::basename(".."), "..");
  TH_EQUAL(flit::basename(""), "");
}

TH_TEST(tst_dirname) {
  TH_EQUAL(flit::dirname("//usr/lib"), "/usr");
  TH_EQUAL(flit::dirname("/usr//lib"), "/usr");
  TH_EQUAL(flit::dirname("/usr/lib"), "/usr");
  TH_EQUAL(flit::dirname("/usr///"), "/");
  TH_EQUAL(flit::dirname("/usr//"), "/");
  TH_EQUAL(flit::dirname("/usr/"), "/");
  TH_EQUAL(flit::dirname("usr"), ".");
  TH_EQUAL(flit::dirname("///"), "/");
  TH_EQUAL(flit::dirname("/"), "/");
  TH_EQUAL(flit::dirname("."), ".");
  TH_EQUAL(flit::dirname(".."), ".");
  TH_EQUAL(flit::dirname(""), ".");
}

} // end of namespace tst_functions

namespace tst_PushDir {

TH_TEST(tst_PushDir_constructor_existing_dir) {
  flit::TempDir temp;
  std::string curdir = flit::curdir();
  flit::PushDir pd(temp.name());
  TH_EQUAL(curdir, pd.previous_dir());
  TH_EQUAL(temp.name(), flit::curdir());
}

TH_TEST(tst_PushDir_constructor_missing_dir) {
  TH_THROWS(flit::PushDir pd("/does/not/exist"), std::ios_base::failure);
}

TH_TEST(tst_PushDir_destructor_existing_dir) {
  flit::TempDir temp1;
  flit::TempDir temp2;
  auto curdir = flit::curdir();
  {
    flit::PushDir pd1(temp1.name());
    TH_EQUAL(curdir, pd1.previous_dir());
    {
      flit::PushDir pd2(temp2.name());
      TH_EQUAL(temp1.name(), pd2.previous_dir());
      TH_EQUAL(temp2.name(), flit::curdir());
    }
    TH_EQUAL(temp1.name(), flit::curdir());
  }
  TH_EQUAL(curdir, flit::curdir());
}

TH_TEST(tst_PushDir_destructor_missing_dir) {
  auto curdir = flit::curdir();
  std::ostringstream captured;
  // capture stderr and assert on its output
  flit::StreamBufReplace buffer_replacer(std::cerr, captured);
  {
    flit::TempDir temp2;
    std::unique_ptr<flit::PushDir> pd1;
    {
      std::unique_ptr<flit::PushDir> pd2;
      {
        flit::TempDir temp1;

        pd1.reset(new flit::PushDir(temp1.name()));
        TH_EQUAL(curdir, pd1->previous_dir());
        TH_EQUAL(temp1.name(), flit::curdir());

        pd2.reset(new flit::PushDir(temp2.name()));
        TH_EQUAL(temp1.name(), pd2->previous_dir());
        TH_EQUAL(temp2.name(), flit::curdir());
      } // deletes temp1
      TH_EQUAL(temp2.name(), flit::curdir());
      TH_EQUAL("", captured.str());
    } // deletes pd2, with temp1.name() not existing
    // should be unable to go to temp1.name() since it doesn't exist
    // but no exception since it is from a destructor
    TH_VERIFY(string_startswith(captured.str(),
                                "ios_base error: Could not change directory"));
    TH_EQUAL(temp2.name(), flit::curdir());
  } // deletes pd1 and temp2
  TH_EQUAL(curdir, flit::curdir());
  // make sure there is at most one newline and it is at the end
  TH_VERIFY(captured.str().find('\n') == captured.str().size() - 1);
}

} // end of namespace tst_PushDir

namespace tst_TempFile {

TH_TEST(tst_TempFile) {
  std::string filename;
  std::string contents = "hello, my name is Michael\n...\nBentley, that is.";
  {
    flit::TempFile tempfile;
    filename = tempfile.name;
    tempfile.out << contents;
    tempfile.out.flush();
    TH_EQUAL(contents, flit::readfile(filename));
  }
  // make sure the file has been deleted.
  TH_THROWS(flit::readfile(filename), std::ios_base::failure);
}

TH_TEST(tst_TempFile_failure) {
  TH_THROWS(flit::TempFile("/parent/does/not/exist"), std::ios_base::failure);
}

} // end of namespace tst_TempFile

namespace tst_TempDir {

TH_TEST(tst_TempDir) {
  std::string tmpdir_name;
  std::string filename = "myfile.txt";
  std::string filepath;

  {
    flit::TempDir tmpdir;
    tmpdir_name = tmpdir.name();
    filepath = flit::join(tmpdir_name, filename);
    flit::touch(filepath);
    auto listing = flit::listdir(tmpdir_name);
    std::vector<std::string> expected_listing { filename };
    TH_EQUAL(expected_listing, listing);
    TH_EQUAL("", flit::readfile(filepath));
  }

  // check that the directory no longer exists
  TH_THROWS(flit::listdir(tmpdir_name), std::ios_base::failure);

  // check that the file no longer exists
  TH_THROWS(flit::readfile(filepath), std::ios_base::failure);
}

TH_TEST(tst_TempDir_parent_does_not_exist) {
  TH_THROWS(flit::TempDir("/parent/does/not/exist"), std::ios_base::failure);
}

TH_TEST(tst_TempDir_directory_already_deleted) {
  flit::TempDir temp;
  flit::rec_rmdir(temp.name());
  // testing that the TempDir destructor does not throw or crash
}

} // end of namespace tst_TempDir

namespace tst_TinyDir {

TH_TEST(tst_TinyDir_constructor_doesnt_exist) {
  TH_THROWS(flit::TinyDir("/dir/does/not/exist"), std::ios_base::failure);
}

TH_TEST(tst_TinyDir_iterate) {
  flit::TempDir tempdir;
  flit::touch(flit::join(tempdir.name(), "file1.txt"));
  flit::touch(flit::join(tempdir.name(), "file2.txt"));
  flit::touch(flit::join(tempdir.name(), "file3.txt"));
  std::vector<std::string> listing;
  std::vector<std::string> expected_listing {
    ".",
    "..",
    "file1.txt",
    "file2.txt",
    "file3.txt",
  };
  flit::TinyDir tinydir(tempdir.name());
  while (tinydir.hasnext()) {
    auto file = tinydir.readfile();
    tinydir.nextfile();
    listing.push_back(file.name);
  }
  std::sort(listing.begin(), listing.end());
  std::sort(expected_listing.begin(), expected_listing.end());
  TH_EQUAL(expected_listing, listing);
}

TH_TEST(tst_TinyDir_iterator) {
  flit::TempDir tempdir;
  flit::touch(flit::join(tempdir.name(), "file1.txt"));
  flit::touch(flit::join(tempdir.name(), "file2.txt"));
  flit::touch(flit::join(tempdir.name(), "file3.txt"));
  std::vector<std::string> listing;
  std::vector<std::string> expected_listing {
    ".",
    "..",
    "file1.txt",
    "file2.txt",
    "file3.txt",
  };
  flit::TinyDir tinydir(tempdir.name());
  for (auto file : tinydir) {
    listing.push_back(file.name);
  }
  std::sort(listing.begin(), listing.end());
  std::sort(expected_listing.begin(), expected_listing.end());
  TH_EQUAL(expected_listing, listing);
}

} // end of namespace tst_TinyDir

namespace tst_FileCloser {

TH_TEST(tst_FileCloser) {
  flit::TempDir temp_dir;
  std::string content = "hello world!";
  std::string fname = flit::join(temp_dir.name(), "tmp.txt");
  FILE* temp_file = fopen(fname.c_str(), "w");
  {
    flit::FileCloser closer(temp_file);
    fprintf(temp_file, "%s", content.c_str());
  }
  TH_EQUAL(content, flit::readfile(fname));
  // Unfortunately, there is no way to check that the FILE* pointer was closed
  // without stubbing.  This is because it is undefined behavior if you use the
  // FILE* pointer in any way after fclose().
}

TH_TEST(tst_FileCloser_null_file) {
  TH_THROWS(flit::FileCloser(nullptr), std::ios_base::failure);
}

// can cause segfaults
//TH_TEST(tst_FileCloser_bad_file) {
//  TH_THROWS(flit::FileCloser(reinterpret_cast<FILE*>(12345)), std::ios_base::failure);
//}

} // end of namespace tst_FileCloser

namespace tst_FdReplace {

TH_TEST(tst_FdReplace) {
  flit::FileCloser t1(tmpfile());
  flit::FileCloser t2(tmpfile());
  fprintf(t1.file, "hi there");
  {
    flit::FdReplace replacer(t1.file, t2.file);
    fprintf(t1.file, "hello ");
    fflush(t1.file);
    fprintf(t2.file, "world");
    fflush(t2.file);
    TH_EQUAL("hello world", flit::readfile(t1.file));
    TH_EQUAL("hello world", flit::readfile(t2.file));
  }
  TH_EQUAL("hi there", flit::readfile(t1.file));
  TH_EQUAL("hello world", flit::readfile(t2.file));
}

TH_TEST(tst_FdReplace_nullptr) {
  flit::FileCloser t1(tmpfile());
  flit::FileCloser t2(tmpfile());
  TH_THROWS(flit::FdReplace(nullptr, nullptr), std::ios_base::failure);
  TH_THROWS(flit::FdReplace(t1.file, nullptr), std::ios_base::failure);
  TH_THROWS(flit::FdReplace(nullptr, t1.file), std::ios_base::failure);
}

} // end of namespace tst_FdReplace

namespace tst_StreamBufReplace {

TH_TEST(tst_StreamBufReplace) {
  std::stringstream s1;
  std::stringstream s2;
  s1 << "hi there";
  TH_EQUAL(s1.str(), "hi there");
  {
    flit::StreamBufReplace stream_replacer(s1, s2);
    s1 << "hello ";
    TH_EQUAL(s2.str(), "hello ");

    s2 << "world";
    TH_EQUAL(s2.str(), "hello world");
  }
  TH_EQUAL(s1.str(), "hi there");
  TH_EQUAL(s2.str(), "hello world");
}

} // end of namespace tst_StreamBufReplace
