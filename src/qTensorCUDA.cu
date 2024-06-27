#include "qTensor.hpp"
#include "qTensorCUDA.cuh"
#include "bitsetCU.cuh"
#include "Contraction.hpp"

#include <cuComplex.h>
#include <unordered_map>
#include <cublas_v2.h>

// using namespace cuda_classes;
using cpx = cuComplex;

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

__device__ void keepNtoMbits(cuda_classes::bitset& bits, int n, int m) 
{ 
    for (int i = 0; i < n; i++) 
    { 
        bits.set(i, 0);
    }  
    for (int i = m; i < 64; i++) 
    { 
        bits.set(i, 0);
    }  
}

__device__ unsigned char getIndexInSet(unsigned char* set, unsigned char element, int size) {
    for (int i = 0; i < size; i++) {
        if (set[i] == element) {
            return i;
        }
    }
    return 255; // Element not found in the set
}

__device__ void print_bitset(cuda_classes::bitset& bits) {
    for (int i = 0; i < 64; i++) {
        printf("%d", bits.get(i));
    }
    printf("\n");
}

__global__ void contractionKernel(unsigned char* d_spanA, unsigned char* d_spanB, unsigned char* d_newSpan, unsigned char* connections, cpx* d_valuesA, cpx* d_valuesB, cpx* d_resultValues, int rankA, int rankB, int rankResult, int connectionsSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= (1 << (rankResult*2))) return;

    cuda_classes::bitset bits(i);

    cuda_classes::bitset a(0);
    cuda_classes::bitset b(0);

    auto lane = d_newSpan;
    for (int k = 0 ; k < rankResult; k++)
    {
        unsigned char indexA = getIndexInSet(d_spanA, *lane, rankA);
        unsigned char indexB = getIndexInSet(d_spanB, *lane, rankB);

        if (indexA != 255) a.set(2*rankA - indexA - 1, bits.get(rankResult*2 - 1 - k));
        else               b.set(2*rankB - indexB - 1, bits.get(rankResult*2 - 1 - k));

        if (indexB != 255) b.set(rankB - indexB - 1, bits.get(rankResult - 1 - k));
        else               a.set(rankA - indexA - 1, bits.get(rankResult - 1 - k));

        lane++;
    }

    for (int m = 0; m < (1 << connectionsSize); m++)
    {
        cuda_classes::bitset address_vacant(m);
        int cnt = 0;
        for (int c = 0; c < connectionsSize; c++)
        {
            unsigned char indexA = getIndexInSet(d_spanA, connections[c], rankA);
            unsigned char indexB = getIndexInSet(d_spanB, connections[c], rankB);
            a.set(rankA - indexA - 1, address_vacant.get(cnt));
            b.set(2*rankB - indexB - 1, address_vacant.get(cnt));
            cnt++;
        }

        cpx value = cuCmulf(d_valuesA[a.to_ulong()], d_valuesB[b.to_ulong()]);
        d_resultValues[i] = cuCaddf(d_resultValues[i], value);
    }
}

int round_div_up (int a, int b){
    return (a + b - 1)/b;
}

QTensor contractionGPU(QTensor A, QTensor B) 
{
    std::set<unsigned char> newSpan;
    newSpan.insert(A.span.begin(), A.span.end());
    newSpan.insert(B.span.begin(), B.span.end());

    // convert all sets to vectors
    std::vector<unsigned char> newSpanVec(newSpan.begin(), newSpan.end());
    std::vector<unsigned char> spanA(A.span.begin(), A.span.end());
    std::vector<unsigned char> spanB(B.span.begin(), B.span.end());

    QTensor result = QTensor(newSpan);
    std::vector<std::complex<float>> resultValues(1 << (result.rank*2), {0.0, 0.0});

    std::vector<unsigned char> connections = findCommonValues(spanA, spanB);

    /** ----------------------------- CUDA ----------------------------- **/
    cudaError_t err;

    // start transfering data to the GPU
    unsigned char* d_spanA, *d_spanB, *d_newSpan, *d_connections;
    cpx* d_valuesA, *d_valuesB, *d_resultValues;

    // memcopies
    {
        err = cudaMalloc(&d_spanA, A.span.size() * sizeof(unsigned char)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMalloc(&d_spanB, B.span.size() * sizeof(unsigned char)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMalloc(&d_newSpan, newSpan.size() * sizeof(unsigned char)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMalloc(&d_connections, connections.size() * sizeof(unsigned char)); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMalloc(&d_valuesA, A.values.size() * sizeof(cpx)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMalloc(&d_valuesB, B.values.size() * sizeof(cpx)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMalloc(&d_resultValues, resultValues.size() * sizeof(cpx)); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMemcpy(d_spanA, spanA.data(), A.span.size() * sizeof(unsigned char), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_spanB, spanB.data(), B.span.size() * sizeof(unsigned char), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_newSpan, newSpanVec.data(), newSpan.size() * sizeof(unsigned char), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_connections, connections.data(), connections.size() * sizeof(unsigned char), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMemcpy(d_valuesA, A.values.data(), A.values.size() * sizeof(cpx), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_valuesB, B.values.data(), B.values.size() * sizeof(cpx), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_resultValues, resultValues.data(), resultValues.size() * sizeof(cpx), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    }

    // kernel call
    {
        int nels = 1 << (result.rank*2);
        int blocksize = 256;
        int numBlocks = round_div_up(nels, blocksize);

        // std::cout << "numBlocks: " << numBlocks << " blocksize: " << blocksize << std::endl;

        contractionKernel<<<numBlocks, blocksize>>>(d_spanA, d_spanB, d_newSpan, d_connections, d_valuesA, d_valuesB, d_resultValues, A.rank, B.rank, result.rank, connections.size());
        err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMemcpy(resultValues.data(), d_resultValues, resultValues.size() * sizeof(cpx), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
    }

    // free memory
    {
        err = cudaFree(d_spanA); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree(d_spanB); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree(d_newSpan); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree(d_connections); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree(d_valuesA); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree(d_valuesB); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree(d_resultValues); cuda_err_check(err, __FILE__, __LINE__);
    }

    result.setValues(resultValues);
    return result;
}

struct gpuQtensor {
    unsigned char* span;
    cpx* values;
};

std::unordered_map<Contraction*, gpuQtensor> gpuQtensorMap;

cublasHandle_t handle;

gpuQtensor moveQtensorToGPU (Contraction* contraction) {
    unsigned char* d_span;
    cpx* d_values;

    cudaError_t err;

    err = cudaMalloc(&d_span, contraction->data.span.size() * sizeof(unsigned char)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_values, contraction->data.values.size() * sizeof(cpx)); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(d_span, contraction->span.data(), contraction->data.span.size() * sizeof(unsigned char), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_values, contraction->data.values.data(), contraction->data.values.size() * sizeof(cpx), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    gpuQtensor gpuA = {d_span, d_values};
    return gpuA;
}

auto contractTreeGPU_r(Contraction* root) -> void {
    if (root == nullptr)
        return;
    if (root->kind == "C") {
        contractTreeGPU_r(root->left);
        contractTreeGPU_r(root->right);

        if (root->left->kind == "G") 
            gpuQtensorMap[root->left] = moveQtensorToGPU(root->left);

        if (root->right->kind == "G")
            gpuQtensorMap[root->right] = moveQtensorToGPU(root->right);

        std::vector<unsigned char> connections = findCommonValues(root->left->span, root->right->span);

        /** ----------------------------- CUDA ----------------------------- **/
        cudaError_t err;

        // start transfering data to the GPU
        unsigned char *d_newSpan, *d_connections;
        cpx *d_resultValues;

        // memcopies
        {
            err = cudaMalloc(&d_newSpan, root->span.size() * sizeof(unsigned char)); cuda_err_check(err, __FILE__, __LINE__);
            err = cudaMalloc(&d_connections, connections.size() * sizeof(unsigned char)); cuda_err_check(err, __FILE__, __LINE__);

            err = cudaMalloc(&d_resultValues, (1 << (root->span.size()*2)) * sizeof(cpx)); cuda_err_check(err, __FILE__, __LINE__);

            err = cudaMemcpy(d_newSpan, root->span.data(), root->span.size() * sizeof(unsigned char), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
            err = cudaMemcpy(d_connections, connections.data(), connections.size() * sizeof(unsigned char), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

            err = cudaMemset(d_resultValues, 0, (1 << (root->span.size()*2)) * sizeof(cpx)); cuda_err_check(err, __FILE__, __LINE__);
        }

        gpuQtensorMap[root] = {d_newSpan, d_resultValues};

        // kernel call
        {
            int nels = 1 << (root->span.size()*2);
            int blocksize = 256;
            int numBlocks = round_div_up(nels, blocksize);

            // std::cout << "numBlocks: " << numBlocks << " blocksize: " << blocksize << std::endl;

            // if the span are the same use gemm
            if (root->left->span == root->right->span && false) {
                size_t nels = 1 << (root->span.size());
                cpx alpha = {1.0, 0.0};
                cpx beta = {0.0, 0.0};

                cublasStatus_t status = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nels, nels, nels, &alpha, gpuQtensorMap[root->left].values, nels, gpuQtensorMap[root->right].values, nels, &beta, gpuQtensorMap[root].values, nels);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    fprintf(stderr, "cublasCgemm failed: %s\n", _cudaGetErrorEnum(status));
                    exit(EXIT_FAILURE);
                }
                err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);
            }
            else {
                contractionKernel<<<numBlocks, blocksize>>>(gpuQtensorMap[root->right].span, gpuQtensorMap[root->left].span, gpuQtensorMap[root].span, d_connections, gpuQtensorMap[root->right].values, gpuQtensorMap[root->left].values, gpuQtensorMap[root].values, root->right->span.size(), root->left->span.size(), root->span.size(), connections.size());
                err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
                err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);
            }
            
            err = cudaFree(gpuQtensorMap[root->left].span); cuda_err_check(err, __FILE__, __LINE__);
            err = cudaFree(gpuQtensorMap[root->left].values); cuda_err_check(err, __FILE__, __LINE__);
            err = cudaFree(gpuQtensorMap[root->right].span); cuda_err_check(err, __FILE__, __LINE__);
            err = cudaFree(gpuQtensorMap[root->right].values); cuda_err_check(err, __FILE__, __LINE__);
            
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

    contractTreeGPU_r(root);

    cudaError_t err;
    std::vector<std::complex<float>> resultValues(1 << (root->span.size()*2));
    err = cudaMemcpy(resultValues.data(), gpuQtensorMap[root].values, resultValues.size() * sizeof(cpx), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
    root->data = QTensor();
    root->data.rank = root->span.size();
    root->data.setValues(resultValues);

    err = cudaFree(gpuQtensorMap[root].span); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(gpuQtensorMap[root].values); cuda_err_check(err, __FILE__, __LINE__);
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasDestroy failed: %s\n", _cudaGetErrorEnum(status));
        exit(EXIT_FAILURE);
    }
}
