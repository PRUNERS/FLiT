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

#include <sys/types.h>  // required for stat.h
#include <sys/stat.h>   // for mkdir()
#include <unistd.h>     // for rmdir()

#include <fstream>
#include <iostream>
#include <string>
#include <vector>


namespace fsutil {

// declarations
inline std::vector<std::string> listdir(const std::string &directory);
inline void printdir(const std::string &directory);
inline void rec_rmdir(const std::string &directory);

struct TempFile {
  std::string name;
  std::ofstream out;
  TempFile();
  ~TempFile();
};

class TempDir {
public:
  TempDir();
  ~TempDir();
  const std::string& name() { return _name; }
private:
  std::string _name;
};

class TinyDir {
public:
  TinyDir(const std::string &directory);
  ~TinyDir();
  tinydir_file readfile() const;
  bool hasnext();
  void nextfile();

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

// definitions

TempFile::TempFile() {
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

TempFile::~TempFile() {
  out.close();
  std::remove(name.c_str());
}

TempDir::TempDir() {
  char fname_buf[L_tmpnam];
  char *s = std::tmpnam(fname_buf);    // gives a warning, but I'm not worried
  if (s != fname_buf) {
    throw std::runtime_error("Could not find temporary directory name");
  }

  _name = fname_buf;
  _name += "-flit-testdir";    // this makes the danger much less likely
  int err = 0;
#if defined(_WIN32)
  err = _mkdir(_name.c_str()); // windows-specific
#else
  err = mkdir(_name.c_str(), 0700); // drwx------
#endif
  if (err != 0) {
    std::string msg = "Could not create temporary directory: ";
    msg += strerror(err);
    throw std::runtime_error(msg);
  }
}

TempDir::~TempDir() {
  try {
    // recursively remove all directories and files found
    rec_rmdir(_name.c_str());
  }
  catch (std::ios_base::failure &ex) {
    std::cerr << "Error (" << ex.code() << "): " << ex.what() << std::endl;
  }
}

TinyDir::TinyDir(const std::string &directory) {
  int err = tinydir_open(&_dir, directory.c_str());
  checkerr(err, "Error opening directory: ");
}

TinyDir::~TinyDir() { tinydir_close(&_dir); }

tinydir_file TinyDir::readfile() const {
  tinydir_file file;
  int err = tinydir_readfile(&_dir, &file);
  std::string msg = "Error reading file '";
  msg += file.name;
  msg += "' from directory: ";
  checkerr(err, msg);
  return file;
}

bool TinyDir::hasnext() {
  return static_cast<bool>(_dir.has_next);
}

void TinyDir::nextfile() {
  int err = tinydir_next(&_dir);
  checkerr(err, "Error getting next file from directory: ");
}

TinyDir::Iterator& TinyDir::Iterator::operator++() {
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

void TinyDir::checkerr(int err, std::string msg) const {
  // TODO: handle broken symlinks
  if (err != 0) {
    throw std::ios_base::failure(
        msg + strerror(errno),
        std::error_code(errno, std::generic_category()));
  }
}

inline std::vector<std::string> listdir(const std::string &directory) {
  std::vector<std::string> ls;
  TinyDir dir(directory);
  for (auto file : dir) {
    ls.emplace_back(file.name);
  }
  return ls;
}

inline void printdir(const std::string &directory) {
  TinyDir dir(directory);
  std::cout << "Directory contents for " << directory << std::endl;
  for (auto file : dir) {
    std::cout << "  " << file.name;
    if (file.is_dir) {
      std::cout << "/";
    }
    std::cout << std::endl;
  }
}

inline void rec_rmdir(const std::string &directory) {
  // remove contents first
  auto contents = listdir(directory);
  for (auto &path : contents) {
    tinydir_file file;
    tinydir_file_open(&file, path.c_str());
    if (file.is_dir) {
      rec_rmdir(file.name);
    } else {
      std::remove(path.c_str());
    }
  }
  // now remove the directory
  int err = 0;
#if defined(_WIN32)
  err = _rmdir(directory.c_str());
#else
  err = rmdir(directory.c_str());
#endif
  if (err != 0) {
    throw std::runtime_error("Could not delete directory");
  }
}

} // end of namespace fsutil

