#include "InfoStream.h"

#include <mutex>

namespace {
  class NullBuffer : public std::streambuf {
  public:
    int overflow(int c) override { return c; }
  };

  class InfoStreamBackend : public std::ostream {
  public:
    InfoStreamBackend() : std::ostream(), _null() { hide(); }
    ~InfoStreamBackend() { hide(); }
    void show() { init(std::cout.rdbuf()); }
    void hide() { init(&_null); }

  private:
    NullBuffer _null;
  };

  static InfoStreamBackend infoStreamBackendSingleton;
  std::mutex infoStreamMutex;
} // end of unnamed namespace

InfoStream::InfoStream()
  : std::ostream()
  , _threadbuf()
{
  init(_threadbuf.rdbuf());
}

InfoStream::~InfoStream() {
  flushout();
}

void InfoStream::show() {
  std::lock_guard<std::mutex> locker(infoStreamMutex);
  infoStreamBackendSingleton.show();
}

void InfoStream::hide() {
  std::lock_guard<std::mutex> locker(infoStreamMutex);
  infoStreamBackendSingleton.show();
}

void InfoStream::flushout() {
  {
    std::lock_guard<std::mutex> locker(infoStreamMutex);
    infoStreamBackendSingleton << _threadbuf.str();
  }
  _threadbuf.str("");
}

