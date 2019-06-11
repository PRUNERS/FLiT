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

/**
 * Some simple functionality for cross-platform file operations, such as
 * deleting an entire directory, creating a temporary file or temporary folder
 * with a name, and listing the contents of a directory.
 */

#ifndef FSUTIL_H
#define FSUTIL_H

#include "tinydir.h"    // for tinydir_file and tinydir_dir

#include <string>       // for std::string
#include <vector>       // for std::vector
#include <fstream>      // for std::ofstream and std::ifstream
#include <iostream>     // for std::ios, std::ostream, and std::cout
#include <streambuf>    // for std::streambuf
#include <sstream>      // for std::ostringstream

#include <cstdio>       // for FILE*

namespace flit {
namespace fsutil {

//= CONSTANTS =//

static const std::string separator = "/";


//= DECLARATIONS =//

// inline
template <typename ... Args> inline std::string join(Args ... args);
inline std::string readfile(FILE* filepointer);
inline std::vector<std::string> listdir(const std::string &directory);
inline void printdir(const std::string &directory);
inline void touch(const std::string &path);

// non-inlined
std::string readfile(const std::string &path);
void rec_rmdir(const std::string &directory);
void mkdir(const std::string &directory, int mode = 0755);
void rmdir(const std::string &directory);
std::string curdir();
void chdir(const std::string &directory);

/** constructor changes current dir, destructor undoes it */
class PushDir {
public:
  PushDir(const std::string &directory);
  ~PushDir();
  const std::string& previous_dir() { return _old_curdir; };
private:
  std::string _old_curdir;
};

/** Constructs a temporary named file with an associated stream.
 *
 * The file will be deleted in the destructor
 */
struct TempFile {
  std::string name;
  std::ofstream out;
  TempFile();
  ~TempFile();
};

/** Constructs a temporary named directory with 0700 permissions.
 *
 * The directory, and all contents, will be deleted in the destructor.
 */
class TempDir {
public:
  TempDir();
  ~TempDir();
  const std::string& name() { return _name; }
private:
  std::string _name;
};

/** Allows listing of a directory.
 *
 * Class wrapper around tinydir.h C library.
 */
class TinyDir {
public:
  TinyDir(const std::string &directory);
  ~TinyDir();
  tinydir_file readfile() const;
  bool hasnext();
  void nextfile();

  /** Iterator class for iterating contents of a directory */
  class Iterator {
  public:
    Iterator(TinyDir *td) : _td(td) {}
    ~Iterator() = default;
    Iterator& operator++();
    bool operator!=(const Iterator &other) const { return _td != other._td; }
    tinydir_file operator*() const { return _td->readfile(); }
  private:
    TinyDir *_td;
  };

  Iterator begin() { return Iterator(this); }
  Iterator end() const { return Iterator(nullptr); }

private:
  void checkerr(int err, std::string msg) const;

private:
  tinydir_dir _dir;
};

/** Closes the FILE* in the destructor */
struct FileCloser {
  FILE* file;
  int fd;
  FileCloser(FILE* _file);
  ~FileCloser() { fclose(file); }
};

/** Replaces a file descriptor with another to redirect output.
 *
 * Destructor restores the file descriptor.
 */
struct FdReplace {
  int old_fd;
  int replace_fd;
  int old_fd_copy;
  FILE* old_file;
  FILE* replace_file;

  FdReplace(FILE* _old_file, FILE* _replace_file);
  ~FdReplace();
};

/** Replaces a stream buffer with another to redirect output.
 *
 * Destructor restores the buffer.
 */
struct StreamBufReplace {
  std::ios &old_stream;
  std::ios &replace_stream;
  std::streambuf *old_buffer;
  StreamBufReplace(std::ios &_old_stream, std::ios &_replace_stream);
  ~StreamBufReplace();
};


//= DEFINITIONS =//

inline void _join_impl(std::ostream &out, const std::string &piece) {
  out << piece;
}

template <typename ... Args>
inline void _join_impl(std::ostream &out, const std::string &piece,
                              Args ... args)
{
  out << piece << separator;
  _join_impl(out, args ...);
}

template <typename ... Args>
inline std::string join(Args ... args) {
  std::ostringstream path_builder;
  _join_impl(path_builder, args ...);
  return path_builder.str();
}

inline std::string readfile(const std::string &path) {
  std::ifstream input;

  // setting the failbit will cause an exception if path does not exist
  input.exceptions(std::ios::failbit);
  input.open(path);
  input.exceptions(std::ios::goodbit);

  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

inline std::vector<std::string> listdir(const std::string &directory) {
  std::vector<std::string> ls;
  TinyDir dir(directory);
  for (auto file : dir) {
    if (file.name != std::string(".") && file.name != std::string("..")) {
      ls.emplace_back(file.name);
    }
  }
  return ls;
}

inline void printdir(const std::string &directory) {
  TinyDir dir(directory);
  std::cout << "Directory contents for " << directory << std::endl;
  for (auto file : dir) {
    std::cout << "  " << file.name;
    if (file.is_dir) {
      std::cout << separator;
    }
    std::cout << std::endl;
  }
}

inline void touch(const std::string &path) {
  // Just creating an output stream and closing it will touch the file.
  std::ofstream out(path);
}

inline PushDir::PushDir(const std::string &directory)
  : _old_curdir(::flit::fsutil::curdir())
{
  ::flit::fsutil::chdir(directory);
}

inline PushDir::~PushDir() {
  try {
    ::flit::fsutil::chdir(_old_curdir);
  } catch (std::ios_base::failure &ex) {
    std::cerr << "ios_base error: " << ex.what() << std::endl;
  }
}

inline TempFile::~TempFile() {
  out.close();
  std::remove(name.c_str());
}

inline TinyDir::TinyDir(const std::string &directory) {
  int err = tinydir_open(&_dir, directory.c_str());
  checkerr(err, "Error opening directory: ");
}

inline TinyDir::~TinyDir() { tinydir_close(&_dir); }

inline tinydir_file TinyDir::readfile() const {
  tinydir_file file;
  int err = tinydir_readfile(&_dir, &file);
  std::string msg = "Error reading file '";
  msg += file.name;
  msg += "' from directory: ";
  checkerr(err, msg);
  return file;
}

inline bool TinyDir::hasnext() {
  return static_cast<bool>(_dir.has_next);
}

inline void TinyDir::nextfile() {
  int err = tinydir_next(&_dir);
  checkerr(err, "Error getting next file from directory: ");
}

inline TinyDir::Iterator& TinyDir::Iterator::operator++() {
  if (_td != nullptr) {
    if (_td->hasnext()) {
      _td->nextfile();
    }
    if (!_td->hasnext()) {
      _td = nullptr;
    }
  }
  return *this;
}

inline FileCloser::FileCloser(FILE* _file) : file(_file) {
  if (file == nullptr) {
    throw std::ios_base::failure("Null FILE pointer passed to FileCloser");
  }
  fd = fileno(file);
  if (fd < 0) {
    throw std::ios_base::failure("Could not get file descriptor");
  }
}

inline StreamBufReplace::StreamBufReplace(std::ios &_old_stream,
                                   std::ios &_replace_stream)
  : old_stream(_old_stream)
  , replace_stream(_replace_stream)
  , old_buffer(_old_stream.rdbuf())
{
  old_stream.rdbuf(replace_stream.rdbuf());
}

inline StreamBufReplace::~StreamBufReplace() {
  old_stream.rdbuf(old_buffer);
}

} // end of namespace fsutil
} // end of namespace flit

#endif // FSUTIL_H
