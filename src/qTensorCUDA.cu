// #include "bitsetCU.cuh"
#include "qTensor.hpp"
#include "bitsetCU.cuh"

#include <cuComplex.h>

// using namespace cuda_classes;
using cpx = cuComplex;

void cuda_err_check (cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf (stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString (err), file, line);
        exit (EXIT_FAILURE);
    }
}

auto findCommonValues = [](std::set<int> set1, std::set<int> set2) -> std::vector<unsigned char> {
    std::vector<unsigned char> commonValues;
    for (auto value : set1) {
        if (set2.find(value) != set2.end()) {
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

#define thread_to_check 123456789

__global__ void contractionKernel(unsigned char* d_spanA, unsigned char* d_spanB, unsigned char* d_newSpan, unsigned char* connections, cpx* d_valuesA, cpx* d_valuesB, cpx* d_resultValues, int rankA, int rankB, int rankResult, int connectionsSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (1 << (rankResult*2))) return;

    cuda_classes::bitset bits(i);

    if (i == thread_to_check)
    print_bitset(bits);

    // cuda_classes::bitset row_res = bits;
    // keepNtoMbits(row_res, rankResult, 2*rankResult);
    // shiftBitsDx(row_res, rankResult);
    // swapFirstNBits(row_res, rankResult);

    // cuda_classes::bitset col_res = bits;
    // keepNtoMbits(col_res, 0, rankResult);
    // swapFirstNBits(col_res, rankResult);

    cuda_classes::bitset a(0);
    cuda_classes::bitset b(0);

    // cuda_classes::bitset row_a(0);
    // cuda_classes::bitset col_a(0);

    // cuda_classes::bitset row_b(0);
    // cuda_classes::bitset col_b(0);

    auto lane = d_newSpan;
    for (int k = 0 ; k < rankResult; k++)
    {
        int indexA = getIndexInSet(d_spanA, *lane, rankA);
        int indexB = getIndexInSet(d_spanB, *lane, rankB);

        if (i == thread_to_check){
        printf("indexA: %d, indexB: %d, k: %d\n", indexA, indexB, k);
        }

        if (indexA != 255) { a.set(2*rankA - indexA - 1, bits.get(rankResult*2 -1 - k));         if (i == thread_to_check){printf("a:"); print_bitset(a); }}
        else               b.set(2*rankB - indexB - 1, bits.get(rankResult*2 - 1 - k));

        if (indexB != 255) { b.set(rankB - indexB - 1, bits.get(rankResult - 1 - k));         if (i == thread_to_check){printf("b:"); print_bitset(b);} }
        else               a.set(rankA - indexA - 1, bits.get(rankResult - 1 - k));

        lane++;
    }

    for (int m = 0; m < (1 << connectionsSize); m++)
    {
        cuda_classes::bitset address_vacant(m);
        int cnt = 0;
        for (int c = 0; c < connectionsSize; c++)
        {
            int indexA = getIndexInSet(d_spanA, connections[c], rankA);
            int indexB = getIndexInSet(d_spanB, connections[c], rankB);
            a.set(rankA - indexA - 1, address_vacant.get(cnt));
            b.set(2*rankB - indexB - 1, address_vacant.get(cnt));
            cnt++;
        }

        if (i == thread_to_check)
        {
            print_bitset(a);
            print_bitset(b);
        }

        
        // cuda_classes::bitset indexes[4] = {row_a, col_a, row_b, col_b};
        // swapFirstNBits(indexes[0], rankA);
        // swapFirstNBits(indexes[1], rankA);
        // swapFirstNBits(indexes[2], rankB);
        // swapFirstNBits(indexes[3], rankB);
        
        // shiftBitsSx(indexes[0], rankA);
        // shiftBitsSx(indexes[2], rankB);

        // cuda_classes::bitset indexA = indexes[0] | indexes[1];
        // cuda_classes::bitset indexB = indexes[2] | indexes[3];

        // cpx value = d_valuesA[indexA.to_ulong()] * d_valuesB[indexB.to_ulong()];
        // atomicAdd(&(d_resultValues[i].real()), value.real());
        // atomicAdd(&d_resultValues[i].imag(), value.imag());
        // d_resultValues[i] += d_valuesA[indexA.to_ulong()] * d_valuesB[indexB.to_ulong()];
        // printf("indexA: %llu, indexB: %llu being added by thread %d\n", a.to_ulong(), b.to_ulong(), i);
        cpx value = cuCmulf(d_valuesA[a.to_ulong()], d_valuesB[b.to_ulong()]);
        atomicAdd(&(d_resultValues[i].x), value.x);
        atomicAdd(&(d_resultValues[i].y), value.y); 
    }
}

int round_div_up (int a, int b){
    return (a + b - 1)/b;
}

QTensor contractionGPU(QTensor A, QTensor B) 
{
    std::set<int> newSpan;
    newSpan.insert(A.span.begin(), A.span.end());
    newSpan.insert(B.span.begin(), B.span.end());

    // convert all sets to vectors
    std::vector<unsigned char> newSpanVec(newSpan.begin(), newSpan.end());
    std::vector<unsigned char> spanA(A.span.begin(), A.span.end());
    std::vector<unsigned char> spanB(B.span.begin(), B.span.end());

    QTensor result = QTensor(newSpan);
    std::vector<std::complex<float>> resultValues(1 << (result.rank*2), {0.0, 0.0});

    std::vector<unsigned char> connections = findCommonValues(A.span, B.span);

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
        cudaFree(d_spanA);
        cudaFree(d_spanB);
        cudaFree(d_newSpan);
        cudaFree(d_connections);
        cudaFree(d_valuesA);
        cudaFree(d_valuesB);
        cudaFree(d_resultValues);
    }

    result.setValues(resultValues);
    return result;
}