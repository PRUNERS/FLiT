#ifndef A_H
#define A_H

class A {
public:
  static int offset; // is a global strong symbol that is not a function
  A();
  virtual ~A();
  virtual int fileA_method1_PROBLEM();
};

int fileA_all();

#endif // A_H
