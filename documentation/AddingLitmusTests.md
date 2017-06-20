#Adding Your Own Litmus Tests to FLiT

All litmus tests that are included in general execution are located in the `QFP/qfpc/tests` directory. For a test to be included in a run it only need be present in this directory. Tests must be in a templated C++ class which extends `flit::TestBase<T>`.

##getInputsPerRun
This function returns the number of arguments the code under tests uses.

##getDefaultInput
This function returns a `flit::TestInput`, which is a vector like object whose `.vals` field holds a list of inputs. Note that more than one set of inputs may be placed in this field.

##run_impl
This function is the actual code under test. The argument is a `flit::TestInput` with length of the `.vals` field equal to the number of inputs requested per run. The return is a vector of two values representing the score of the test. This does not need to represent 'good' or 'bad' results, as the score is used to classify different compilations.

##REGISTER_TEST
Just as it sounds, this function is used to register the test in the framework. It's argument is the class created for the test.

#CUDA Tests
The main setup for CUDA tests is the same, with the exception that the `run_impl` function is a CUDA kernel which is placed in the framework though `getKernel()` and it has a second argument which is treated as an inout variable with fields `.s1` and `.s2` to hold the scores.

#Integrating into the test framework
The test cpp file must be placed in `QFP/qfpc/tests` for it to be included in subsequent runs. No other changes are required.

#Starting code
This is a simple source file to start your own tests.

```
#include "test_base.hpp" 
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

// These functions are not needed, but it is nice to break out the core computation being done
template <typename T>
DEVICE
T 
my_cuda_test_core(const T input_1) // Number of arguments can be changed
{
    return input_1; // Cuda test code
}

template <typename T>
T 
my_cpp_test_core(const T input_1) // Number of arguments can be changed
{
    return input_1; // Cpp test code
}


template <typename T>
GLOBAL
void 
my_cuda_kern(const flit::CuTestInput<T>* tiList, flit::CudaResultElement* results){
#ifdef __CUDA__
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
#else
    auto idx = 0;
#endif
    T input_1 = tiList[idx].vals[0]; // Extract arguments from vals

    // Loops or other structures can be used here such as in TrianglePSylv.cpp  
    double score = my_cuda_test_core(input);

    results[idx].s1 = score;
    results[idx].s2 = 0.0;
}

template <typename T>
class MyTest: public flit::TestBase<T> {
public:
    MyTest(std::string id) : flit::TestBase<T>(std::move(id)) {}

    // This must be changed to match the number of arguments used in the test code
    virtual size_t getInputsPerRun() { return 1; } 

    virtual flit::TestInput<T> getDefaultInput() {
        flit::TestInput<T> ti;
        ti.vals = { 6.0 };
        return ti;
    }

protected:
    virtual
    flit::KernelFunction<T>* getKernel() {return my_cuda_kern; }
    virtual
    flit::ResultType::mapped_type run_impl(const flit::TestInput<T>& ti) {
        T input_1 = ti.vals[0]; // Extract arguments from vals
        
        // Loops or other structures can be used here such as in TrianglePSylv.cpp  
        double score = my_cpp_kern(input_1);

        return {score, 0.0};
    }

protected:
    using flit::TestBase<T>::id;
};

REGISTER_TYPE(MyTest)
```
