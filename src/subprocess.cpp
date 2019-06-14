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

flit::ProcResult call_main_impl(
    flit::MainFunc *func, std::string run_wrap, std::string progname,
    std::string remaining_args)
{
  // calls myself
  auto funcname = flit::find_main_name(func);
  if (progname == "") {
    progname = flit::g_program_path;
  }
  std::ostringstream command_builder;
  command_builder << run_wrap << " "
                  << '"' << flit::g_program_path << '"'
                  << " --progname \"" << progname << "\""
                  << " --call-main " << funcname
                  << " " << remaining_args;
  return flit::call_with_output(command_builder.str());
}

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
  out << "ProcResult("
      << "ret="   << res.ret << ", "
      << "out=\"" << res.out << "\", "
      << "err=\"" << res.err << "\")";
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
  std::string run_wrapper = "";
  return call_main_impl(func, run_wrapper, progname, remaining_args);
}

ProcResult call_mpi_main(MainFunc *func, std::string mpirun_command,
                         std::string progname, std::string remaining_args)
{
  return call_main_impl(func, mpirun_command, progname, remaining_args);
}

} // end of namespace flit
