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

#include "tinydir.h"    // to list directory contents

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
#include <sstream>      // for std::istringstream
#include <stdexcept>    // for std::runtime_error
#include <string>
#include <vector>

#if !defined(__GNUC__) || defined(__clang__) || \
    (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 8))
#define FSUTIL_IS_NOT_GCC_4_8
#endif

namespace fsutil {

//= CONSTANTS =//

const std::string separator = "/";


//= DECLARATIONS =//

template <typename ... Args> inline std::string join(Args ... args);
inline std::string readfile(const std::string &path);
inline std::vector<std::string> listdir(const std::string &directory);
inline void printdir(const std::string &directory);
inline void rec_rmdir(const std::string &directory);
inline void mkdir(const std::string &directory, int mode = 0777);
inline void rmdir(const std::string &directory);
inline std::string curdir();
inline void chdir(const std::string &directory);
inline void touch(const std::string &path);

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

/** Closes the file in the destructor */
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
  std::ifstream input(path);
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

inline void rec_rmdir(const std::string &directory) {
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
  ::fsutil::rmdir(directory);
}

inline void mkdir(const std::string &directory, int mode) {
  int err = 0;
#if defined(_WIN32)
  err = ::_mkdir(directory.c_str()); // windows-specific
#else
  err = ::mkdir(directory.c_str(), mode); // drwx------
#endif
  if (err != 0) {
    std::string msg = "Could not create temporary directory: ";
    msg += strerror(err);
    throw std::runtime_error(msg);
  }
}

inline void rmdir(const std::string &directory) {
  int err = 0;
#if defined(_WIN32)
  err = ::_rmdir(directory.c_str());
#else
  err = ::rmdir(directory.c_str());
#endif
  if (err != 0) {
    std::string msg = "Could not create temporary directory '" + directory
                      + "': ";
    msg += strerror(err);
    throw std::runtime_error(msg);
  }
}

inline std::string curdir() {
  const int bufsize = 10000; // you never know...
  char buffer[bufsize]; // where to store the current directory
  char *ret = nullptr;
#if defined(_WIN32)
  ret = ::_getcwd(buffer, bufsize);
#else
  ret = ::getcwd(buffer, bufsize);
#endif
  if (ret == nullptr) {
    throw std::runtime_error("Could not get current directory");
  }
  return std::string(buffer);
}

inline void chdir(const std::string &directory) {
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
    throw std::runtime_error(msg);
  }
}

inline void touch(const std::string &path) {
  // Just creating an output stream and closing it will touch the file.
  std::ofstream out(path);
}

PushDir::PushDir(const std::string &directory)
  : _old_curdir(::fsutil::curdir())
{
  ::fsutil::chdir(directory);
}

PushDir::~PushDir() {
  try {
    ::fsutil::chdir(_old_curdir);
  } catch (std::runtime_error &ex) {
    std::cerr << "Runtime error: " << ex.what() << std::endl;
  }
}

inline TempFile::TempFile() {
  char fname_buf[L_tmpnam];
  char *s = std::tmpnam(fname_buf);    // gives a warning, but I'm not worried
  if (s != fname_buf) {
    throw std::runtime_error("Could not create temporary file");
  }

  name = fname_buf;
  name += "-flit-testfile.in";    // this makes the danger much less likely
  out.exceptions(std::ios::failbit);
  out.open(name);
}

inline TempFile::~TempFile() {
  out.close();
  std::remove(name.c_str());
}

inline TempDir::TempDir() {
  char fname_buf[L_tmpnam];
  char *s = std::tmpnam(fname_buf);    // gives a warning, but I'm not worried
  if (s != fname_buf) {
    throw std::runtime_error("Could not find temporary directory name");
  }

  _name = fname_buf;
  _name += "-flit-testdir";    // this makes the danger much less likely
  ::fsutil::mkdir(_name, 0700);
}

inline TempDir::~TempDir() {
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
  }
  catch (std::runtime_error &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
  }
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

inline void TinyDir::checkerr(int err, std::string msg) const {
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

FileCloser::FileCloser(FILE* _file) : file(_file) {
  if (file == nullptr) {
    throw std::ios_base::failure("Null FILE pointer passed to FileCloser");
  }
  fd = fileno(file);
  if (fd < 0) {
    throw std::ios_base::failure("Could not get file descriptor");
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

StreamBufReplace::StreamBufReplace(std::ios &_old_stream,
                                   std::ios &_replace_stream)
  : old_stream(_old_stream)
  , replace_stream(_replace_stream)
  , old_buffer(_old_stream.rdbuf())
{
  old_stream.rdbuf(replace_stream.rdbuf());
}

StreamBufReplace::~StreamBufReplace() {
  old_stream.rdbuf(old_buffer);
}

} // end of namespace fsutil

