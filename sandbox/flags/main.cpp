#include <cstdio>
#include <cstdlib>

float reinterpret_int(int val) {
  return *reinterpret_cast<float*>(&val);
}

unsigned reinterpret_float(float val) {
  return *reinterpret_cast<unsigned*>(&val);
}

double rand_double() {
  int32_t vals[2];
  vals[0] = rand();
  vals[1] = rand();
  return *reinterpret_cast<double*>(vals);
}

unsigned long reinterpret_double(double val) {
  return *reinterpret_cast<unsigned long*>(&val);
}

void test_float(float a, float b, float c);
void test_double(double a, double b, double c);

int main(void) {
  float a = reinterpret_int(rand());
  float b = 1.09822958423098e23f;
  float c = 1.89503429849218e6f;

  //printf("a: %.40e\n", a);
  //printf("b: %.40e\n", b);
  //printf("c: %.40e\n", c);
  
  test_float(a, b, c);

  //float a = 3.3673678e26;
  //float b = 1.0982295e23;
  //float c = 1.8950343e6;
  //
  //test_float(a, b, c);

  return 0;
}

void test_float(float a, float b, float c) {

  float first  = (a + b) * c;
  float second = (a * c) + (b * c);
  float third  = first - second;

  printf("a = %g\t = 0x%08x\n", a, reinterpret_float(a));
  printf("b = %g\t = 0x%08x\n", b, reinterpret_float(b));
  printf("c = %g\t = 0x%08x\n", c, reinterpret_float(c));
  
  printf("\n");

  printf("sizeof(float)     = %lu\n", sizeof(float));
  printf("(a + b) * c       = %g\n", first);
  printf("(a * c) + (b * c) = %g\n", second);
  printf("difference        = %g\n", third);

  printf("\n");

  printf("(a + b) * c       = 0x%08x\n", reinterpret_float(first));
  printf("(a * c) + (b * c) = 0x%08x\n", reinterpret_float(second));
  printf("difference        = 0x%08x\n", reinterpret_float(third));
}

