#include <cmath>
#include <typeinfo>

#include <stdio.h>

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#ifdef __CUDA__
#include "CUHelpers.hpp"
#include "cudaTests.hpp"

template <typename T>
//__global__
void
DoMatrixMultSanityKernel(uint32_t dim, cudaResultElement* results){
  printf("entered kernel\n");
  auto b = CUHelpers::VectorCU<T>::getRandomVector(dim);
  printf("ok\n");
  auto c = CUHelpers::MatrixCU<T>::Identity(dim) * b;
  printf("ok\n");
  results[0].s1 = c.L1Distance(b);
  printf("ok\n");
  results[0].s2 = c.LInfDistance(b);
  printf("from kernel, %f and %f are results\n", c.L1Distance(b), c.LInfDistance(b));
}

template <typename T>
QFPTest::resultType DoMatrixMultSanity_CUDA(uint32_t dim,
                                            std::string id,
                                            uint32_t count = 1){
  printf("hi from cuda func\n");
  cudaResultElement* dresults;
  cudaResultElement* hresults = new cudaResultElement[1];
  // cudaMalloc(&dresults, sizeof(cudaResultElement) * 1);
  // DoMatrixMultSanityKernel<T><<<1,1>>>(dim, dresults);
  // cudaMemcpy(hresults, dresults, 1, cudaMemcpyDeviceToHost);
  DoMatrixMultSanityKernel<T>(dim, hresults);
  QFPTest::resultType rt;
  for(uint32_t x = 0; x < count; ++x){
    rt[{id + std::to_string(hresults[x].index), typeid(T).name()}] =
      {hresults[x].s1, hresults[x].s2};
  }
  printf("exiting cuda func, error %d\n", cudaGetLastError());
  
  return rt;
}
                                            
#endif

template <typename T>
class DoMatrixMultSanity: public QFPTest::TestBase {
public:
  DoMatrixMultSanity(std::string id) : QFPTest::TestBase(id){}

  QFPTest::resultType operator()(const QFPTest::testInput& ti) {
#ifdef __CUDA__
    return DoMatrixMultSanity_CUDA<T>(ti.highestDim, id);
#else
    auto dim = ti.highestDim;
    QFPHelpers::Vector<T> b = QFPHelpers::Vector<T>::getRandomVector(dim);
    auto c = QFPHelpers::Matrix<T>::Identity(dim) * b;
    QFPHelpers::info_stream << "Product is: " << c << std::endl;
    bool eq = c == b;
    QFPHelpers::info_stream << "A * b == b? " << eq << std::endl;
    return {{
      {id, typeid(T).name()}, {c.L1Distance(b), c.LInfDistance(b)}
    }};
#endif
  }
};

REGISTER_TYPE(DoMatrixMultSanity)
