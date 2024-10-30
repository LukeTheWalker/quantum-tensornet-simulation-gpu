#pragma once

#include <stdio.h>

namespace cuda_classes{
    class bitset{
        public:
        unsigned char data[8];
        template<typename T>
        __host__ __device__ bitset(T value){ *(size_t *) data = (size_t)value; }
        __host__ __device__ bitset() : bitset(0ull) {}

        __host__ __device__ void set (size_t index, bool value){
            size_t byte_index = index >> 3;
            size_t bit_index = index & 7;
            if (value) data[byte_index] |= (1 << bit_index);
            else data[byte_index] &= ~(1 << bit_index);
        }

        __host__ __device__ bool get (size_t index){
            size_t byte_index = index >> 3;
            size_t bit_index = index & 7;
            return (data[byte_index] >> bit_index) & 1;
        }

        __host__ __device__ unsigned long long to_ulong(){
            return *(unsigned long long *) data;
        }

        __host__ __device__ bitset operator| (const bitset& other){
            return bitset(*(unsigned long long *) data | *(unsigned long long *) other.data);
        }

        __host__ __device__ void transfer_bit (size_t index, bitset& other){
            size_t byte_index = index >> 3;
            size_t bit_index = index & 7;
            if (get(index)) other.data[byte_index] |= (1 << bit_index);
            else other.data[byte_index] &= ~(1 << bit_index);
        }

        __host__ __device__ void xor_op (size_t other){
            *(size_t *) data ^= other;
        }
    };
}