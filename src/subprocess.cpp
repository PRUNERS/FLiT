#include "subprocess.h"
#include "fsutil.h"
#include "flitHelpers.h"

#include <string>
#include <sstream>
#include <map>

#include <cstdio>

namespace {

std::map<std::string, flit::MainFunc*> main_name_map;
std::map<flit::MainFunc*, std::string> main_func_map;

} // end of unnamed namespace

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

void register_main_func(const std::string &main_name, MainFunc* main_func) {
  if (main_func == nullptr) {
    throw std::invalid_argument("Main func is null");
  }
  if (main_name_map.find(main_name) != main_name_map.end()) {
    throw std::logic_error("Main name already registered: " + main_name);
  }
  if (main_func_map.find(main_func) != main_func_map.end()) {
    throw std::logic_error("Main func already registered with " + main_name);
  }
  main_name_map[main_name] = main_func;
  main_func_map[main_func] = main_name;
}

MainFunc* find_main_func(const std::string &main_name) {
  return main_name_map.at(main_name);
}

std::string find_main_name(MainFunc *main_func) {
  if (main_func == nullptr) {
    throw std::invalid_argument("Main func is null");
  }
  return main_func_map.at(main_func);
}

ProcResult call_main(MainFunc *func, std::string progname,
                     std::string remaining_args)
{
  // calls myself
  auto funcname = find_main_name(func);
  if (progname == "") {
    progname = g_program_name;
  }
  std::ostringstream command_builder;
  command_builder << '"' << g_program_name << '"'
                  << " --progname \"" << progname << "\""
                  << " --call-main " << funcname
                  << " " << remaining_args;
  return call_with_output(command_builder.str());
}

} // end of namespace flit
