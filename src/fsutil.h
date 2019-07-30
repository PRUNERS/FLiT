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
#include <cstring>      // for strerror()

namespace flit {

//= CONSTANTS =//

static const std::string separator = "/";


//= DECLARATIONS =//

// inline
inline void ifopen(std::ifstream& in, const std::string &filename);
inline void ofopen(std::ofstream& out, const std::string &filename);
template <typename ... Args> inline std::string join(Args ... args);
inline std::string readfile(const std::string &path);
inline std::vector<std::string> listdir(const std::string &directory);
inline void printdir(const std::string &directory);
inline void touch(const std::string &path);
// throws a std::ios_base::failure() if not found
inline tinydir_file file_status(const std::string &filepath);

// non-inlined
std::string readfile(FILE* filepointer);
void rec_rmdir(const std::string &directory);
void mkdir(const std::string &directory, int mode = 0755);
void rmdir(const std::string &directory);
std::string curdir();
void chdir(const std::string &directory);
// throws a std::ios_base::failure() if not found
std::string which(const std::string &command);
std::string which(const std::string &command, const std::string &path);
std::string realpath(const std::string &relative);

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
  TempFile(const std::string &parent = "/tmp");
  ~TempFile();
};

/** Constructs a temporary named directory with 0700 permissions.
 *
 * The directory, and all contents, will be deleted in the destructor.
 */
class TempDir {
public:
  TempDir(const std::string &parent = "/tmp");
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

/** Opens a file, but on failure, throws std::ios::failure
 *
 * T must be one of {fstream, ifstream, ofstream}
 *
 * The passed in filestream should be an empty-constructed stream
 */
template <typename T>
void _openfile_check(T& filestream, const std::string &filename) {
  // turn on exceptions on failure
  filestream.exceptions(std::ios::failbit);

  // opening will throw if the file does not exist or is not readable
  filestream.open(filename);

  // turn off all exceptions (back to the default behavior)
  filestream.exceptions(std::ios::goodbit);
}

/** Opens a file for reading, but on failure, throws std::ios::failure
 *
 * The passed in filestream should be an empty-constructed stream
 *
 * Note: This was changed to pass in the stream rather than return it because
 * GCC 4.8 failed to compile - it failed to use the move assignment operator
 * and move constructor, and instead tried to use the copy constructor.
 */
inline void ifopen(std::ifstream& in, const std::string &filename) {
  _openfile_check<std::ifstream>(in, filename);
}

/** Opens a file for writing, but on failure, throws std::ios::failure
 *
 * The passed in filestream should be an empty-constructed stream
 *
 * Note: this was changed to pass in the stream rather than return it because
 * GCC 4.8 failed to compile - it failed to use the move assignment operator
 * and move constructor, and instead tried to use the copy constructor.
 */
inline void ofopen(std::ofstream& out, const std::string &filename) {
  _openfile_check<std::ofstream>(out, filename);
  out.precision(1000);  // lots of digits of precision
}

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

inline tinydir_file file_status(const std::string &filepath) {
  tinydir_file filestat;
  int status = tinydir_file_open(&filestat, filepath.c_str());
  if (status != 0) {
    throw std::ios_base::failure("Error reading file " + filepath + " "
                                 + strerror(errno));
  }
  return filestat;
}

inline PushDir::PushDir(const std::string &directory)
  : _old_curdir(::flit::curdir())
{
  ::flit::chdir(directory);
}

inline PushDir::~PushDir() {
  try {
    ::flit::chdir(_old_curdir);
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

} // end of namespace flit

#endif // FSUTIL_H
