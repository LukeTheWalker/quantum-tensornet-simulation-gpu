#pragma once

#include <stdio.h>

namespace cuda_classes{
    class bitset{
        public:
        unsigned char data[8];
        template<typename T>
        __host__ __device__ bitset(T value){
            size_t byte_size = sizeof(T);
            if (byte_size > 8) { printf("Error: the size of the input value is larger than the size of the bitset\n"); return; }
            for (size_t i = 0; i < byte_size; i++)
            {
                // TODO: compute all possible values for the bitset at compile time
                // big endian
                data[i] = (value >> (i*8)) & 0xFF;

                // little endian
                // data[byte_size-i-1] = (value >> (i*8)) & 0xFF;
            }
        }
        __host__ __device__ bitset(){
            for (size_t i = 0; i < 8; i++)
            {
                data[i] = 0;
            }
        }

        __host__ __device__ void set (size_t index, bool value){
            size_t byte_index = index/8;
            size_t bit_index = index%8;
            if (value) data[byte_index] |= (1 << bit_index);
            else data[byte_index] &= ~(1 << bit_index);
        }

        __host__ __device__ bool get (size_t index){
            size_t byte_index = index/8;
            size_t bit_index = index%8;
            return (data[byte_index] >> bit_index) & 1;
        }

        __host__ __device__ unsigned long long to_ulong(){
            unsigned long long value = 0;
            for (size_t i = 0; i < 8; i++)
            {
                value |= (data[i] << (i*8));
            }
            return value;
        }

        __host__ __device__ bitset operator| (const bitset& other){
            bitset result;
            for (size_t i = 0; i < 8; i++)
            {
                result.data[i] = data[i] | other.data[i];
            }
            return result;
        }

        __host__ __device__ void transfer_bit (size_t index, bitset& other){
            size_t byte_index = index/8;
            size_t bit_index = index%8;
            if (get(index)) other.data[byte_index] |= (1 << bit_index);
            else other.data[byte_index] &= ~(1 << bit_index);
        }
    };
}