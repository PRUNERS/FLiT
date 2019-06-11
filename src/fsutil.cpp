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

#include "fsutil.h"

#if !defined(__GNUC__) || defined(__clang__) || \
    (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 8))
#define FSUTIL_IS_NOT_GCC_4_8
#endif

#if defined(_WIN32)
#include <dirent.h>
#include <direct.h>
#else // !defined(_WIN32)
#include <sys/types.h>  // required for stat.h
#include <sys/stat.h>   // for mkdir()
#include <unistd.h>     // for rmdir()
#endif // defined _MSC_VER

#include <fstream>
#include <ios>          // for std::ios_base::failure
#include <iostream>
#include <stdexcept>    // for std::runtime_error
#include <sstream>      // for std::istringstream
#include <string>
#include <vector>

namespace flit {
namespace fsutil {

std::string readfile(FILE* file) {
  fseek(file, 0, SEEK_END);
  auto size = ftell(file);
  rewind(file);
  std::vector<char> buffer(size + 1, '\0');
  long read_size = fread(buffer.data(), 1, size, file);
  if (read_size != size) {
    throw std::ios_base::failure(
        "Did not read in one go (" + std::to_string(size) + ", "
        + std::to_string(read_size) + ")");
  }
  return std::string(buffer.data());
}

void rec_rmdir(const std::string &directory) {
  // remove contents first
  auto contents = listdir(directory);
  for (auto &name : contents) {
    std::string path = join(directory, name);
    tinydir_file file;
    tinydir_file_open(&file, path.c_str());
    if (file.is_dir) {
      rec_rmdir(path);
    } else {
      std::remove(path.c_str());
    }
  }
  // now remove the directory
  ::flit::fsutil::rmdir(directory);
}

void mkdir(const std::string &directory, int mode) {
  int err = 0;
#if defined(_WIN32)
  err = ::_mkdir(directory.c_str()); // windows-specific
#else
  err = ::mkdir(directory.c_str(), mode); // drwx------
#endif
  if (err != 0) {
    std::string msg = "Could not create directory: ";
    msg += strerror(err);
    throw std::ios_base::failure(msg);
  }
}

void rmdir(const std::string &directory) {
  int err = 0;
#if defined(_WIN32)
  err = ::_rmdir(directory.c_str());
#else
  err = ::rmdir(directory.c_str());
#endif
  if (err != 0) {
    std::string msg = "Could not remove directory: '" + directory
                      + "': ";
    msg += strerror(err);
    throw std::ios_base::failure(msg);
  }
}

std::string curdir() {
  const int bufsize = 10000; // you never know...
  char buffer[bufsize]; // where to store the current directory
  char *ret = nullptr;
#if defined(_WIN32)
  ret = ::_getcwd(buffer, bufsize);
#else
  ret = ::getcwd(buffer, bufsize);
#endif
  if (ret == nullptr) {
    throw std::ios_base::failure("Could not get current directory");
  }
  return std::string(buffer);
}

void chdir(const std::string &directory) {
  int err = 0;
#if defined(_WIN32)
  err = ::_chdir(directory.c_str());
#else
  err = ::chdir(directory.c_str());
#endif
  if (err != 0) {
    std::string msg = "Could not change directory '" + directory
                      + "': ";
    msg += strerror(err);
    throw std::ios_base::failure(msg);
  }
}

TempFile::TempFile() {
  char fname_buf[L_tmpnam];
  char *s = std::tmpnam(fname_buf);    // gives a warning, but I'm not worried
  if (s != fname_buf) {
    throw std::ios_base::failure("Could not create temporary file");
  }

  name = fname_buf;
  name += "-flit-testfile.in";    // this makes the danger much less likely
  out.exceptions(std::ios::failbit);
  out.open(name);
}

TempDir::TempDir() {
  char fname_buf[L_tmpnam];
  char *s = std::tmpnam(fname_buf);    // gives a warning, but I'm not worried
  if (s != fname_buf) {
    throw std::ios_base::failure("Could not find temporary directory name");
  }

  _name = fname_buf;
  _name += "-flit-testdir";    // this makes the danger much less likely
  ::flit::fsutil::mkdir(_name, 0700);
}

TempDir::~TempDir() {
  try {
    // recursively remove all directories and files found
    rec_rmdir(_name.c_str());
  }
  catch (std::ios_base::failure &ex) {
    std::cerr << "Error"
#ifdef FSUTIL_IS_NOT_GCC_4_8
// This was an unimplemented part of C++11 from GCC 4.8
                 " (" << ex.code() << ")"
#endif
                 ": " << ex.what() << std::endl;
  } catch (std::runtime_error &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
  }
}

void TinyDir::checkerr(int err, std::string msg) const {
  // TODO: handle broken symlinks
  if (err != 0) {
    throw std::ios_base::failure(
        msg + strerror(errno)
#ifdef FSUTIL_IS_NOT_GCC_4_8
// This was an unimplemented part of C++11 from GCC 4.8
        , std::error_code(errno, std::generic_category())
#endif
        );
  }
}

FdReplace::FdReplace(FILE* _old_file, FILE* _replace_file)
  : old_file(_old_file)
  , replace_file(_replace_file)
{
  if (old_file == nullptr || replace_file == nullptr) {
    throw std::ios_base::failure("Null FILE pointer passed to FdReplace");
  }
  old_fd = fileno(old_file);
  replace_fd = fileno(replace_file);
  if (old_fd < 0) {
    throw std::ios_base::failure("Could not get fileno of old_fd");
  }
  if (replace_fd < 0) {
    throw std::ios_base::failure("Could not get fileno of replace_fd");
  }
  fflush(old_file);
  old_fd_copy = dup(old_fd);
  if (old_fd_copy < 0) {
    throw std::ios_base::failure("Could not dup old_fd");
  }
  if (dup2(replace_fd, old_fd) < 0) {
    throw std::ios_base::failure("Could not replace old_fd");
  }
}

FdReplace::~FdReplace() {
  fflush(old_file);
  dup2(old_fd_copy, old_fd);
}

} // end of namespace fsutil
} // end of namespace flit
