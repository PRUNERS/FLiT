#ifndef INFO_STREAM_H
#define INFO_STREAM_H

#include <sstream>
#include <iostream>

class InfoStream : public std::ostream {
public:
  InfoStream();
  ~InfoStream();

  // enable move
  InfoStream(InfoStream &&other) = default;
  InfoStream& operator=(InfoStream &&other) = default;

  void show();
  void hide();
  void flushout();

private:
  std::ostringstream _threadbuf;
};

#endif // INFO_STREAM_H

