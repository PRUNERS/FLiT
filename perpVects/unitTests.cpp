#include <utility>
#include <string>
#include <map>

#include "QFPHelpers.h"

namespace UnitTests{

  using namespace QFPHelpers;
  
  typedef std::pair<std::string, bool> results;

  

  template<typename T>
  static results
  TestGenOrthoVector(){
    bool result = true;
    auto A = Vector<T>::getRandomVector(10, -10, 10);
    auto O = A.genOrthoVector();
    if(!(A.isOrtho(O))) result = false;
    return {__func__, result};
  }  

  template<typename T>
  static results
  TestL1Distance(){
    bool result = true;
    Vector<T> A = {12.25, 77.45, 99.9};
    Vector<T> B = {-17.29, 33.3, -1};
    T output = A.L1Distance(B);
    T expected = (12.25 - -17.29) + (77.45 - 33.3) + (99.9 - -1);
    if(!(output == expected)){
      result = false;
      info_stream << "in " << __func__ << ":" << std::endl;
      info_stream << "A:" << std::endl << A << std::endl;
      info_stream << "B:" << std::endl << B << std::endl;
      info_stream << "expected:" << std::endl << expected << std::endl;
      info_stream << "output:" << std::endl << output << std::endl;
    }
    return{__func__, result};
  }

  template<typename T>
  static results
  TestCrossProd(){
    bool result = true;
    auto A = Vector<T>::getRandomVector(3, -3, 3);
    auto B = Vector<T>::getRandomVector(3, -3, 3);
    // Vector<T> A = {1, 0, 0};
    // Vector<T> B = {0, 1, 1};
    auto C = A.cross(B);
    T r1 = A ^ C;
    T r2 = B ^ C;
    bool expected = 0;
    if(!(r1 == expected && r2 == expected)){
      result = false;
      info_stream << "in " << __func__ << "(AXB), check ortho AC, BC:" << std::endl;
      info_stream << "A:" << std::endl << A << std::endl;
      info_stream << "B:" << std::endl << B << std::endl;
      info_stream << "C:" << std::endl << C << std::endl;
      info_stream << "expected:" << std::endl << expected << std::endl;
      info_stream << "r1, r2: " << r1 << ", " << r2 << std::endl;
    }
    return{__func__, result};
  }
  
  template<typename T>
  static results
  TestInnerProd(){
    bool result = true;
    Vector<T> A = {12.25, 77.45, 99.9};
    Vector<T> B = {-17.29, 33.3, -1};
    T expected = (12.25*-17.29)+(77.45*33.3)+(99.9*-1);
    T output = A ^ B;
    if(!(output == expected)){
      result = false;
      info_stream << "in " << __func__ << ":" << std::endl;
      info_stream << "A:" << std::endl << A << std::endl;
      info_stream << "B:" << std::endl << B << std::endl;
      info_stream << "expected:" << std::endl << expected << std::endl;
      info_stream << "output:" << std::endl << output << std::endl;
    }
    return{__func__, result};
  }

  
  template<typename T>
  static results
  TestL2Distance(){
    bool result = true;
    Vector<T> A = {2, 3, 4};
    Vector<T> B = {4, 5, 6};
    if(!(A.L2Distance(B) != sqrt(12))) result = false;
    return {__func__, result};
  }
  
  template<typename T>
  static results
  UnitVector(){
    bool result = true;
    auto V = Vector<T>::getRandomVector(5, -20, 20);
    T expected = 1.0;
    auto output = V.getUnitVector().L2Norm();
    if(!(output == expected)){
      result = false;
      info_stream << "in " << __func__ << ":" << std::endl;
      info_stream << "V:" << std::endl << V << std::endl;
      info_stream << "expected:" << std::endl << expected << std::endl;
      info_stream << "output:" << std::endl << output << std::endl;
      info_stream << "expected bits: " << std::hex << FPWrap<T>(expected) << std::endl;
      info_stream << "output bits:" << std::hex << FPWrap<T>(output) << std::endl;
    }
    return{__func__, result};
  }

  template<typename T>
  static results
  MplusM(){
    bool result = true;
    Matrix<T> A = {{1, 2, 3},
		   {4, 5, 6},
		   {7, 8, 9}};
    Matrix<T> B = {{9, 8, 7},
		   {6, 5, 4},
		   {3, 2, 1}};
    Matrix<T> expected = {{10, 10, 10},
			  {10, 10, 10},
			  {10, 10, 10}};
    if(!(A + B == expected)) result = false;
    return {__func__, result};
  }

  template<typename T>
  static results
  MxV(){
    bool result = true;
    Matrix<T> A = {{77, 16.23, 99},
		   {17.7777, -23.3, 11},
		   {131, 134, 137}};
    Vector<T> b = {-18, 374, 12};
    Vector<T> expected = {5872.02, -8902.199, 49402};
    auto output = A * b;
    if(!(output == expected)){
      result = false;
      info_stream << "in " << __func__ << ":" << std::endl;
      info_stream << "A:" << std::endl << A << std::endl;
      info_stream << "b:" << std::endl << b << std::endl;
      info_stream << "expected:" << std::endl << expected << std::endl;
      info_stream << "output:" << std::endl << output << std::endl;
    }
    return {__func__, result};
  }
  
  template<typename T>
  static results
  MxM(){
    bool result = true;
    Matrix<T> A = {{1,2,3},
		   {.25, .5, .75},
		   {4,5,6}};
    Matrix<T> B = {{7,8,9},
		   {11,12,13},
		   {8,7,5}};
    Matrix<T> expected = {{53, 53, 50},
			  {13.25, 13.25, 12.5},
			  {131, 134, 131}};
    auto output = A * B;
    if(!(output == expected)){
      result = false;
      info_stream << "in " << __func__ << ":" << std::endl;
      info_stream << "A:" << std::endl << A << std::endl;
      info_stream << "B:" << std::endl << B << std::endl;
      info_stream << "expected:" << std::endl << expected << std::endl;
      info_stream << "output:" << std::endl << output << std::endl;
    }
    return {__func__, result};
  
  }
  template<typename T>
  static results
  MxI(){
    bool result = true;
    info_stream << "in " << __func__ << ":" << std::endl;
    auto ident = Matrix<T>::Identity(3);
    Matrix<T> M = {{1, 2, 3},
		   {4, 5, 6},
		   {7, 8, 9}};
    info_stream << "Multiplying matrices: " << std::endl;
    info_stream << M << std::endl;
    info_stream << "and..." << std::endl;
    info_stream << ident << std::endl;
    if(M * ident == M) info_stream << "success" << std::endl;
    else{
      info_stream << "failure with: " << std::endl;
      info_stream << (M * ident) << std::endl;
      result = false;
    }
    return {__func__, result};
  }

  //tests Matrix*scal operator
  template<typename T>
  static results
  MxS(){
    bool retVal = true;
    info_stream << "in " << __func__ << ":" << std::endl;
    Matrix<T> M = {{1,2,3},
		   {4,5,6},
		   {7,8,9}};
    Matrix<T> const target = {{10, 20, 30},
			      {40, 50, 60},
			      {70, 80, 90}};
    info_stream << M << std::endl;
    T scalar = 10;
    info_stream << "and scalar: " << scalar << std::endl;
    auto res = M * scalar;
    if(res == target) info_stream << "success" << std::endl;
    else{
      info_stream << "failed with: " << std::endl;
      info_stream << res << std::endl;
      retVal = false;
    }
    return {__func__, retVal};
  }
  void RunTests(std::map<std::string, bool> &results, bool detailed = false){
    typedef float prec;
    if(detailed) info_stream.show(); //reinit(cout.rdbuf());
    //cout << "starting unit tests" << std::endl;
    results.insert(TestGenOrthoVector<prec>());
    results.insert(TestGenOrthoVector<prec>());
    results.insert(TestL1Distance<prec>());
    results.insert(TestCrossProd<prec>());
    results.insert(TestInnerProd<prec>());
    results.insert(UnitVector<prec>());
    results.insert(MplusM<prec>());
    results.insert(MxV<prec>());
    results.insert(MxM<prec>());
    results.insert(MxI<prec>());
    results.insert(MxS<prec>());
  }

  int DoTests(){
    int retVal = 0;
    std::map<std::string, bool> results;
    RunTests(results);
    if(std::any_of(results.begin(), results.end(), [](UnitTests::results x){return x.second;})){
      for(auto i: results){
	std::cout << i.first << "\t" << i.second << std::endl;
      }
      retVal = 1;
      std::cout << "here are the details:" << std::endl;
      RunTests(results, true);
    }else std::cout << "all tests passed" << std::endl;
    return retVal;
  }
}
