#include "qTensor.cuh"
#include "qTensorCUDA.cuh"
#include "bitsetCU.cuh"
#include "Contraction.hpp"

#include <cuComplex.h>
#include <unordered_map>
#include <cublas_v2.h>

#define DEBUG false

// using namespace cuda_classes;
#ifdef USE_FLOAT
using dtype = float;
using cpx = cuFloatComplex;
#else
using dtype = double;
using cpx = cuDoubleComplex;
#endif

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

void cuda_err_check (cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf (stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString (err), file, line);
        exit (EXIT_FAILURE);
    }
}

auto findCommonValues = [](std::vector<unsigned char> set1, std::vector<unsigned char> set2) -> std::vector<unsigned char> {
    std::vector<unsigned char> commonValues;
    for (auto value : set1) {
        if (std::find(set2.begin(), set2.end(), value) != set2.end()) {
            commonValues.push_back((unsigned char)value);
        }
    }
    return commonValues;
};

unsigned char getIndexInSet(unsigned char* set, unsigned char element, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (set[i] == element) {
            return i;
        }
    }
    return 255; // Element not found in the set
}

__device__ void keepNtoMbits(cuda_classes::bitset& bits, size_t n, size_t m) 
{ 
    for (size_t i = 0; i < n; i++) 
    { 
        bits.set(i, 0);
    }  
    for (size_t i = m; i < 64; i++) 
    { 
        bits.set(i, 0);
    }  
}

__device__ void print_bitset(cuda_classes::bitset& bits) {
    for (size_t i = 0; i < 64; i++) {
        printf("%d", bits.get(i));
    }
    printf("\n");
}

__global__ void contractionKernel(cuda_classes::bitset* bit_addressesA, cuda_classes::bitset* bit_addressesB, cpx* d_valuesA, cpx* d_valuesB, cpx* d_resultValues, size_t rankA, size_t rankB, size_t rankResult, size_t connectionsSize, unsigned char* indexesA_connections, unsigned char* indexesB_connections)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= (1 << (rankResult*2))) return;

    #ifdef USE_FLOAT
    cpx value = cuCmulf(d_valuesA[bit_addressesA[i].to_ulong()], d_valuesB[bit_addressesB[i].to_ulong()]);
    d_resultValues[i] = cuCaddf(d_resultValues[i], value);
    #else
    cpx value = cuCmul(d_valuesA[bit_addressesA[i].to_ulong()], d_valuesB[bit_addressesB[i].to_ulong()]);
    d_resultValues[i] = cuCadd(d_resultValues[i], value);
    #endif
    
    size_t old_gray = 0;
    for (size_t m = 1; m < (1 << connectionsSize); m++)
    {
        size_t gray_code = m ^ (m >> 1);

        unsigned int position_vacant =  __ffsll(gray_code ^ old_gray) - 1;

        unsigned char indexA = indexesA_connections[position_vacant];
        unsigned char indexB = indexesB_connections[position_vacant];

        bit_addressesA[i].xor_op(1 << (rankA + indexA));
        bit_addressesB[i].xor_op(1 << (indexB));

        if (i == 0 && DEBUG) {
            // print indexA
            printf("IndexA: %d\n", indexA);
            // print indexB
            printf("IndexB: %d\n", indexB);
            printf("m: %d\n", m);
            printf("gray_code: %d\n", gray_code);
            printf("old_gray: %d\n", old_gray);
            printf("position_vacant: %d\n", position_vacant);
            printf("RankA: %d\n", rankA);
            printf("RankB: %d\n", rankB);
            printf("ConnectionsSize: %d\n", connectionsSize);
            printf("BitAdressA: ");
            print_bitset(bit_addressesA[i]);
            printf("BitAdressB: ");
            print_bitset(bit_addressesB[i]);
            printf("BitaAdressA: %d\n", bit_addressesA[i].to_ulong());
            printf("BitaAdressB: %d\n", bit_addressesB[i].to_ulong());
            printf("Values to multiply: %f + %fj and %f + %fj\n", d_valuesA[bit_addressesA[i].to_ulong()].x, d_valuesA[bit_addressesA[i].to_ulong()].y, d_valuesB[bit_addressesB[i].to_ulong()].x, d_valuesB[bit_addressesB[i].to_ulong()].y);
        }

        #ifdef USE_FLOAT
        cpx value = cuCmulf(d_valuesA[bit_addressesA[i].to_ulong()], d_valuesB[bit_addressesB[i].to_ulong()]);
        d_resultValues[i] = cuCaddf(d_resultValues[i], value);
        #else
        cpx value = cuCmul(d_valuesA[bit_addressesA[i].to_ulong()], d_valuesB[bit_addressesB[i].to_ulong()]);
        d_resultValues[i] = cuCadd(d_resultValues[i], value);
        #endif

        old_gray = gray_code;
    }
}

__global__ void compute_bit_address_map(cuda_classes::bitset* bit_addressesA, cuda_classes::bitset* bit_addressesB, size_t rankA, size_t rankB, size_t rankResult,  unsigned char* indexesA, unsigned char* indexesB){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= (1 << (rankResult*2))) return;

    cuda_classes::bitset bits(i);


    for (size_t k = 0 ; k < rankResult; k++)
    {
        if (indexesB[k] != 255) bit_addressesB[i].set(rankB + indexesB[k], bits.get(rankResult + k));
        else                    bit_addressesA[i].set(rankA + indexesA[k], bits.get(rankResult + k));

        if (indexesA[k] != 255) bit_addressesA[i].set(indexesA[k], bits.get(k));
        else                    bit_addressesB[i].set(indexesB[k], bits.get(k));
    
    }

    if (i == 0 && DEBUG) {
        printf("BitAdressA: ");
        print_bitset(bit_addressesA[i]);
        printf("BitAdressB: ");
        print_bitset(bit_addressesB[i]);
    }
}

size_t round_div_up (size_t a, size_t b){
    return (a + b - 1)/b;
}

struct gpuQtensor {
    cpx* values;
};

std::unordered_map<Contraction*, gpuQtensor> gpuQtensorMap;

cublasHandle_t handle;

gpuQtensor moveQtensorToGPU (Contraction* contraction, cudaStream_t stream) {
    cpx* d_values;

    cudaError_t err;

    err = cudaMallocAsync(&d_values, contraction->data.getValuesSize() * sizeof(cpx), stream); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpyAsync(d_values,  contraction->data.values, contraction->data.getValuesSize() * sizeof(cpx), cudaMemcpyHostToDevice, stream); cuda_err_check(err, __FILE__, __LINE__);

    gpuQtensor gpuA = {d_values};
    return gpuA;
}

cudaEvent_t leftEvent, rightEvent;
auto contractTreeGPU_r(Contraction* root) -> void {
    if (root == nullptr)
        return;
    if (root->kind == "C") {
        contractTreeGPU_r(root->left);
        contractTreeGPU_r(root->right);

        cudaError_t err;

        err = cudaEventRecord(leftEvent, root->left->stream); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaEventRecord(rightEvent, root->right->stream); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaStreamWaitEvent(root->stream, leftEvent, 0); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaStreamWaitEvent(root->stream, rightEvent, 0); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaStreamDestroy(root->left->stream); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaStreamDestroy(root->right->stream); cuda_err_check(err, __FILE__, __LINE__);

        if (root->left->kind == "G") 
            gpuQtensorMap[root->left] = moveQtensorToGPU(root->left, root->stream);

        if (root->right->kind == "G")
            gpuQtensorMap[root->right] = moveQtensorToGPU(root->right, root->stream);

        std::vector<unsigned char> connections = findCommonValues(root->left->span, root->right->span);

        /** ----------------------------- CUDA ----------------------------- **/

        // start transfering data to the GPU
        cpx *d_resultValues;

        // unsigned char indexesA[root->span.size()];
        // unsigned char indexesB[root->span.size()];

        // unsigned char indexes_connectionsA[connections.size()];
        // unsigned char indexes_connectionsB[connections.size()];

        // use cudaMallocHost to allocate pinned memory
        unsigned char* indexesA, *indexesB, *indexes_connectionsA, *indexes_connectionsB;
        err = cudaMallocHost(&indexesA, root->span.size() * sizeof(unsigned char), cudaHostAllocWriteCombined); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocHost(&indexesB, root->span.size() * sizeof(unsigned char), cudaHostAllocWriteCombined); cuda_err_check(err, __FILE__, __LINE__);
        
        err = cudaMallocHost(&indexes_connectionsA, connections.size() * sizeof(unsigned char), cudaHostAllocWriteCombined); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocHost(&indexes_connectionsB, connections.size() * sizeof(unsigned char), cudaHostAllocWriteCombined); cuda_err_check(err, __FILE__, __LINE__);

        // memcopies
        {

            err = cudaMallocAsync(&d_resultValues, (1 << (root->span.size()*2)) * sizeof(cpx), root->stream); cuda_err_check(err, __FILE__, __LINE__);

            err = cudaMemsetAsync(d_resultValues, 0, (1 << (root->span.size()*2)) * sizeof(cpx), root->stream); cuda_err_check(err, __FILE__, __LINE__);
        }

        gpuQtensorMap[root] = {d_resultValues};

        // kernel call
        {
            size_t nels = 1 << (root->span.size()*2);
            size_t blocksize = 256;
            size_t numBlocks = round_div_up(nels, blocksize);
            size_t sharedMemSize = root->span.size() * sizeof(unsigned char);

            // std::cout << "numBlocks: " << numBlocks << " blocksize: " << blocksize << std::endl;

            // if the span are the same use gemm
            if (root->left->span == root->right->span) {
                cublasSetStream(handle, root->stream);
                size_t nels = 1 << (root->span.size());
                cpx alpha = {1.0, 0.0};
                cpx beta = {0.0, 0.0};
                #ifdef USE_FLOAT
                cublasStatus_t status = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nels, nels, nels, &alpha, gpuQtensorMap[root->left].values, nels, gpuQtensorMap[root->right].values, nels, &beta, gpuQtensorMap[root].values, nels);
                #else
                cublasStatus_t status = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nels, nels, nels, &alpha, gpuQtensorMap[root->left].values, nels, gpuQtensorMap[root->right].values, nels, &beta, gpuQtensorMap[root].values, nels);
                #endif
                if (status != CUBLAS_STATUS_SUCCESS) {
                    fprintf(stderr, "cublasCgemm failed: %s\n", _cudaGetErrorEnum(status));
                    exit(EXIT_FAILURE);
                }
                // err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
                // err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);
            }
            else {
                cuda_classes::bitset* bit_addressesA, *bit_addressesB;

                err = cudaMallocAsync(&bit_addressesA, nels * sizeof(cuda_classes::bitset), root->stream); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaMallocAsync(&bit_addressesB, nels * sizeof(cuda_classes::bitset), root->stream); cuda_err_check(err, __FILE__, __LINE__);

                err = cudaMemsetAsync(bit_addressesA, 0, nels * sizeof(cuda_classes::bitset), root->stream); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaMemsetAsync(bit_addressesB, 0, nels * sizeof(cuda_classes::bitset), root->stream); cuda_err_check(err, __FILE__, __LINE__);

                #pragma omp parallel for
                for (size_t i = 0; i < root->span.size(); i++) {
                    indexesA[i] = getIndexInSet(root->left->span.data(), root->span[i], root->left->span.size());
                    indexesB[i] = getIndexInSet(root->right->span.data(),  root->span[i], root->right->span.size());
                }

                // // print indexA 
                // for (size_t i = 0; i < root->span.size(); i++) {
                //     std::cout << "IndexA[" << i << "]: " << (int)indexesA[i] << std::endl;
                // }

                // // print indexB
                // for (size_t i = 0; i < root->span.size(); i++) {
                //     std::cout << "IndexB[" << i << "]: " << (int)indexesB[i] << std::endl;
                // }
                
                #pragma omp parallel for
                for (size_t i = 0; i < connections.size(); i++) {
                    indexes_connectionsA[i] = getIndexInSet(root->left->span.data(), connections[i], root->left->span.size());
                    indexes_connectionsB[i] = getIndexInSet(root->right->span.data(),  connections[i], root->right->span.size());
                }

                unsigned char* d_indexesA, *d_indexesB;
                err = cudaMallocAsync(&d_indexesA, root->span.size() * sizeof(unsigned char), root->stream); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaMallocAsync(&d_indexesB, root->span.size() * sizeof(unsigned char), root->stream); cuda_err_check(err, __FILE__, __LINE__);

                err = cudaMemcpyAsync(d_indexesA, indexesA, root->span.size() * sizeof(unsigned char), cudaMemcpyHostToDevice, root->stream); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaMemcpyAsync(d_indexesB, indexesB, root->span.size() * sizeof(unsigned char), cudaMemcpyHostToDevice, root->stream); cuda_err_check(err, __FILE__, __LINE__);

                unsigned char* d_indexes_connectionsA, *d_indexes_connectionsB;
                err = cudaMallocAsync(&d_indexes_connectionsA, connections.size() * sizeof(unsigned char), root->stream); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaMallocAsync(&d_indexes_connectionsB, connections.size() * sizeof(unsigned char), root->stream); cuda_err_check(err, __FILE__, __LINE__);

                err = cudaMemcpyAsync(d_indexes_connectionsA, indexes_connectionsA, connections.size() * sizeof(unsigned char), cudaMemcpyHostToDevice, root->stream); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaMemcpyAsync(d_indexes_connectionsB, indexes_connectionsB, connections.size() * sizeof(unsigned char), cudaMemcpyHostToDevice, root->stream); cuda_err_check(err, __FILE__, __LINE__);

                double gb_used = (double)(sizeof(cuda_classes::bitset) * nels * 2) / (1024 * 1024 * 1024);

                if (gb_used > 1)
                    std::cout << "Memory allocation: " << (double)(sizeof(cuda_classes::bitset) * nels * 2) / (1024 * 1024 * 1024) << " GB" << std::endl;

                compute_bit_address_map<<<numBlocks, blocksize, 0, root->stream>>>(bit_addressesA, bit_addressesB, root->left->span.size(), root->right->span.size(), root->span.size(), d_indexesA, d_indexesB);

                // err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
                // err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

                contractionKernel<<<numBlocks, blocksize, 0, root->stream>>>(bit_addressesA, bit_addressesB, gpuQtensorMap[root->left].values, gpuQtensorMap[root->right].values, gpuQtensorMap[root].values, root->left->span.size(), root->right->span.size(), root->span.size(), connections.size(), d_indexes_connectionsA, d_indexes_connectionsB);

                // err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
                // err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

                err = cudaFreeAsync(d_indexesA, root->stream); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaFreeAsync(d_indexesB, root->stream); cuda_err_check(err, __FILE__, __LINE__);

                err = cudaFreeAsync(d_indexes_connectionsA, root->stream); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaFreeAsync(d_indexes_connectionsB, root->stream); cuda_err_check(err, __FILE__, __LINE__);

                err = cudaFreeAsync(bit_addressesA, root->stream); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaFreeAsync(bit_addressesB, root->stream); cuda_err_check(err, __FILE__, __LINE__);
            }
            
            err = cudaFreeAsync(gpuQtensorMap[root->left].values, root->stream); cuda_err_check(err, __FILE__, __LINE__);
            err = cudaFreeAsync(gpuQtensorMap[root->right].values, root->stream); cuda_err_check(err, __FILE__, __LINE__);
        }
    }
}

auto contractTreeGPU(Contraction* root) -> void {
    cublasStatus_t status;
    status = cublasCreate(&handle); 
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed: %s\n", _cudaGetErrorEnum(status));
        exit(EXIT_FAILURE);
    }
    
    cudaError_t err;

    err = cudaEventCreate(&leftEvent); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventCreate(&rightEvent); cuda_err_check(err, __FILE__, __LINE__);

    contractTreeGPU_r(root);

    err = cudaEventDestroy(leftEvent); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventDestroy(rightEvent); cuda_err_check(err, __FILE__, __LINE__);

    std::vector<std::complex<dtype>> resultValues(1 << (root->span.size()*2));
    err = cudaMemcpy(resultValues.data(), gpuQtensorMap[root].values, resultValues.size() * sizeof(cpx), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
    root->data = QTensor();
    root->data.rank = root->span.size();
    root->data.setValues(resultValues);

    err = cudaFree(gpuQtensorMap[root].values); cuda_err_check(err, __FILE__, __LINE__);
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasDestroy failed: %s\n", _cudaGetErrorEnum(status));
        exit(EXIT_FAILURE);
    }

    // std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;
}
