#include "testBase.hpp"
#include "QFPHelpers.hpp"

//setup for Eigen library test suite
//there's a race on this container -- switching to 1 concurrency
std::map<std::string, QFPTest::resultType> eigenResults;
//QFPTest::resultType eigenResults;
std::mutex eigenResults_mutex;
std::mutex g_test_stack_mutex;


// namespace adjoint {
// #include "eigen/adjoint.cpp"
// }
// EIGEN_CLASS_DEF(EigenAdjoint, adjoint)
// REGISTER_TYPE(EigenAdjoint)

// namespace array {
// #include "eigen/array.cpp"
// }
// EIGEN_CLASS_DEF(EigenArray, array)
// REGISTER_TYPE(EigenArray)

// namespace array_for_matrix {
// #include "eigen/array_for_matrix.cpp"
// }
// EIGEN_CLASS_DEF(EigenArrayForMatrix, array_for_matrix)
// REGISTER_TYPE(EigenArrayForMatrix)

// namespace array_replicate {
// #include "eigen/array_replicate.cpp"
// }
// EIGEN_CLASS_DEF(EigenArrayReplicate, array_replicate)
// REGISTER_TYPE(EigenArrayReplicate)

// namespace array_reverse {
//   #include "eigen/array_reverse.cpp"
// }
// EIGEN_CLASS_DEF(EigenArrayReverse, array_reverse)
// REGISTER_TYPE(EigenArrayReverse)

// namespace bandmatrix {
//   #include "eigen/bandmatrix.cpp"
// }
// EIGEN_CLASS_DEF(EigenBandMatrix, bandmatrix)
// REGISTER_TYPE(EigenBandMatrix)

// namespace basicstuff {
//   #include "eigen/basicstuff.cpp"
// }
// EIGEN_CLASS_DEF(EigenBasicStuff, basicstuff)
// REGISTER_TYPE(EigenBasicStuff)

// // namespace bicgstab {
// //   #include "eigen/Eigen/src/Core/util/ForwardDeclarations.h"
// //   #include "eigen/bicgstab.cpp"
// // }
// // EIGEN_CLASS_DEF(EigenBicGStab, bicgstab)
// // REGISTER_TYPE(EigenBicGStab)

// namespace block {
//   #include "eigen/block.cpp"
// }
// EIGEN_CLASS_DEF(EigenBlock, block)
// REGISTER_TYPE(EigenBlock)

// namespace cholesky {
//   #include "eigen/cholesky.cpp"
// }
// EIGEN_CLASS_DEF(EigenCholesky, cholesky)
// REGISTER_TYPE(EigenCholesky)

// // namespace cholmod_support {
// //   #include "eigen/cholmod_support.cpp"
// // }
// // EIGEN_CLASS_DEF(EigenCholmodSupport, cholmod_support)
// // REGISTER_TYPE(EigenCholmodSupport)

// namespace commainitializer {
//   #include "eigen/commainitializer.cpp"
// }
// EIGEN_CLASS_DEF(EigenCommaInitializer, commainitializer)
// REGISTER_TYPE(EigenCommaInitializer)

// // namespace conjugate_gradient {
// //   #include "eigen/Eigen/src/IterativeLinearSolvers/ConjugateGradient.h"
// //   #include "eigen/conjugate_gradient.cpp"
// // }
// // EIGEN_CLASS_DEF(EigenConjugateGradient, conjugate_gradient)
// // REGISTER_TYPE(EigenConjugateGradient)

// namespace corners {
//   #include "eigen/corners.cpp"
// }
// EIGEN_CLASS_DEF(EigenCorners, corners)
// REGISTER_TYPE(EigenCorners)

// namespace cwiseop {
//   #include "eigen/cwiseop.cpp"
// }
// EIGEN_CLASS_DEF(EigenCWiseop, cwiseop)
// REGISTER_TYPE(EigenCWiseop)


//we're going to have to isolate InfoStream.hpppppp for multiple eigen tests
#include "eigenMain.hpp"
#include "eigen/unsupported/test/levenberg_marquardt.cpp"
EIGEN_CLASS_DEF(EigenLevenbergMarquardt, levenberg_marquardt)
REGISTER_TYPE(EigenLevenbergMarquardt)
