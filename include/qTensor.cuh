#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <complex>
#include <set>
#include <array>
#include <cuda_runtime.h>
#include <cuda.h>
#include "bitset.hpp"

#ifdef USE_FLOAT
using dtype = float;
#else
using dtype = double;
#endif

class QTensor
{
    void cuda_err_check_cpu (cudaError_t err, const char *file, int line)
    {
        if (err != cudaSuccess)
        {
            fprintf (stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString (err), file, line);
            exit (EXIT_FAILURE);
        }
    }
    public:
        // std::set<unsigned char> span;
        // std::vector<std::complex<dtype> > values;
        unsigned char * span;
        std::complex<dtype> * values;
        QTensor() {}
        // QTensor(std::set<unsigned char> span): span(span){this->rank = span.size();}
        QTensor(std::set<unsigned char> span){ 
            this->rank = span.size();
            cudaMallocHost(&this->span, span.size());
            // this->span = (unsigned char *)malloc(span.size());
            std::copy(span.begin(), span.end(), this->span);
        }
        template <typename T>
        QTensor(std::vector<T> values, std::vector<unsigned char> span_) : QTensor(std::set<unsigned char>(span_.begin(), span_.end()))
        {   
            cudaMallocHost(&this->values, std::pow(2, rank * 2) * sizeof(std::complex<dtype>));
            for (size_t i = 0 ; i < values.size(); i+=2)
                this->values[i/2] = {(dtype)values[i], (dtype)values[i+1]};
        }

        void setValues(std::vector<std::complex<dtype>> values) 
        {
            this->values = (std::complex<dtype> *)malloc(std::pow(2, rank * 2) * sizeof(std::complex<dtype>));
            if (values.size() != std::pow(2, rank * 2))
            {
                std::cerr << "Error: the number of values is not consistent with the rank of the tensor" << std::endl;
                exit(1);
            }else{
                for (size_t i = 0 ; i < values.size(); i++)
                {
                    this->values[i] = values[i];
                }
            } 
        }

        std::complex<dtype> getValue(size_t index) { return values[index]; }

        void printValues(std::ostream& os = std::cout) const
        {
            for (size_t i = 0 ; i < std::pow(2, 2 * rank); i++)
            {
                if (i != 0 && (i % (1 << rank)) == 0)
                {
                    os << std::endl;
                }
                os << values[i] << ", ";
            }
        }

        size_t getValuesSize() const { return std::pow(2, 2 * rank); }

        size_t getRank() const { return rank; }
    public:
        size_t rank;
};