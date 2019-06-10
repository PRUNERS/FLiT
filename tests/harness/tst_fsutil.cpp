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

#include <algorithm>
#include <memory>

namespace {

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

void tst_printdir() {
  TH_SKIP("unimplemented");
}
TH_REGISTER(tst_printdir);

void tst_rec_rmdir() {
  TH_SKIP("unimplemented");
}
TH_REGISTER(tst_rec_rmdir);

void tst_mkdir() {
  TH_SKIP("unimplemented");
}
TH_REGISTER(tst_mkdir);

void tst_rmdir() {
  TH_SKIP("unimplemented");
}
TH_REGISTER(tst_rmdir);

void tst_curdir() {
  TH_SKIP("unimplemented");
}
TH_REGISTER(tst_curdir);

void tst_chdir() {
  TH_SKIP("unimplemented");
}
TH_REGISTER(tst_chdir);

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
  TH_THROWS(fsutil::PushDir pd("/does/not/exist"), std::runtime_error);
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
  // TODO: capture stderr and assert on its output
  auto curdir = fsutil::curdir();
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
    } // deletes pd2, with temp1.name() not existing
    // should be unable to go to temp1.name() since it doesn't exist
    // but no exception since it is from a destructor
    TH_EQUAL(temp2.name(), fsutil::curdir());
  } // deletes pd1 and temp2
  TH_EQUAL(curdir, fsutil::curdir());
}
TH_REGISTER(tst_PushDir_destructor_missing_dir);

} // end of namespace tst_PushDir

namespace tst_TempFile {

void tst_TempFile_constructor() {
  TH_SKIP("unimplemented");
}
TH_REGISTER(tst_TempFile_constructor);

void tst_TempFile_destructor() {
  TH_SKIP("unimplemented");
}
TH_REGISTER(tst_TempFile_destructor);

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

void tst_TinyDir_constructor() {
  TH_SKIP("unimplemented");
}
TH_REGISTER(tst_TinyDir_constructor);

void tst_TinyDir_destructor() {
  TH_SKIP("unimplemented");
}
TH_REGISTER(tst_TinyDir_destructor);

void tst_TinyDir_readfile() {
  TH_SKIP("unimplemented");
}
TH_REGISTER(tst_TinyDir_readfile);

void tst_TinyDir_hasnext() {
  TH_SKIP("unimplemented");
}
TH_REGISTER(tst_TinyDir_hasnext);

void tst_TinyDir_iterate() {
  TH_SKIP("unimplemented");
}
TH_REGISTER(tst_TinyDir_iterate);

void tst_TinyDir_iterator() {
  TH_SKIP("unimplemented");
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
