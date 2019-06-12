#include "subprocess.h"
#include "fsutil.h"

#include <string>
#include <sstream>

#include <cstdio>

namespace flit {

ProcResult call_with_output(const std::string &command) {
  fsutil::TempFile tempfile;

  // run the command
  std::string cmd = command + " 2>" + tempfile.name;
  FILE* output = popen(cmd.c_str(), "r");

  // grab stdout
  std::ostringstream outbuilder;
  const int bufsize = 256;
  char buf[bufsize];
  if (output) {
    while (!feof(output)) {
      if (fgets(buf, bufsize, output) != nullptr) {
        outbuilder << buf;
      }
    }
  }

  // wait and grab the return code
  int ret = pclose(output);
  auto err = fsutil::readfile(tempfile.name);

  if (WIFEXITED(ret)) {
    ret = WEXITSTATUS(ret);
  }

  return ProcResult { ret, outbuilder.str(), err };
}

std::ostream& operator<<(std::ostream& out, const ProcResult &res) {
  out << "ProcResult(\""
      << res.out << "\", \""
      << res.err << "\", "
      << res.ret << ")";
  return out;
}

} // end of namespace flit
