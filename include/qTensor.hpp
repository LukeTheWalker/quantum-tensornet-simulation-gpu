#pragma once
#include <iostream>
#include <vector>
#include <bitset>
#include <cmath>
#include <algorithm>
#include <complex>
#include <set>
#include <array>

class QTensor
{
    public:
        std::set<int> span;
        std::vector<std::complex<float> > values;
        QTensor() {}
        QTensor(std::set<int> span): span(span){this->rank = span.size();}
        template <typename T>
        QTensor(std::vector<T> values, std::vector<int> span_) : span(span_.begin(), span_.end())
        {
            rank = span.size();
            for (int i = 0 ; i < values.size(); i+=2)
            {
                this->values.push_back({(float)values[i], (float)values[i+1]});
            }
        }

        void setValues(std::vector<std::complex<float>> values) 
        {
            if (values.size() != std::pow(2, rank*2))
            {
                std::cerr << "Error: the number of values is not consistent with the rank of the tensor" << std::endl;
            }else{
                for (int i = 0 ; i < values.size(); i++)
                {
                    this->values.push_back(values[i]);
                }
            } 
        }

        std::complex<float> getValue(int index) { return values[index]; }

        static QTensor contraction(QTensor A, QTensor B) 
        {
            auto keepNtoMbits = [](std::bitset<64>& bits, int n, int m) {
                for (int i = 0; i < 64; i++) {
                    if(i<n||i>=m)
                    bits.reset(i);
                }
            };

            auto shiftBitsDx = [](std::bitset<64>& bits, int x) 
            { 
                for (int i = 0; i < 64; i++) 
                { 
                    if(i+x > 63)
                    {
                        return;
                    }
                    bits[i] = bits[i+x];
                    bits[i+x] = 0;
                }  
            };

            auto swapFirstNBits = [](std::bitset<64>& bits, size_t n) {
                for (size_t i = 0; i < n / 2; ++i) {
                    bool temp = bits[i];
                    bits[i] = bits[n - i - 1];
                    bits[n - i - 1] = temp;
                }
            };

            auto getIndexInSet = [](std::set<int> set, int element) {
                int index = 0;
                for (auto it = set.begin(); it != set.end(); ++it) {
                    if (*it == element) {
                        return index;
                    }
                    index++;
                }
                return -1; // Element not found in the set
            };

            auto findCommonValues = [](std::set<int> set1, std::set<int> set2) -> std::vector<int> {
                std::vector<int> commonValues;
                for (auto value : set1) {
                    if (set2.find(value) != set2.end()) {
                        commonValues.push_back(value);
                    }
                }
                return commonValues;
            };

            auto shiftBitsSx = [](std::bitset<64>& bits, int x) 
            { 
                for (int i = 63; i >= x; i--) 
                { 
                    bits[i] = bits[i-x];
                    bits[i-x] = 0;
                }  
            };


            std::set<int> newSpan;
            newSpan.insert(A.span.begin(), A.span.end());
            newSpan.insert(B.span.begin(), B.span.end());
            QTensor result = QTensor(newSpan);
            std::vector<std::complex<float>> resultValues(1 << (result.rank*2), {0.0, 0.0});

            std::vector<int> connections = findCommonValues(A.span, B.span);

            #pragma omp parallel for
            for (int i = 0; i < (1 << (result.rank*2)); i++)
            {   
                std::bitset<64> bits(i);

                std::bitset<64> row_res = bits;
                keepNtoMbits(row_res, result.rank, 2*result.rank);
                shiftBitsDx(row_res, result.rank);
                swapFirstNBits(row_res, result.rank);

                std::bitset<64> col_res = bits;
                keepNtoMbits(col_res, 0, result.rank);
                swapFirstNBits(col_res, result.rank);
            
                std::bitset<64> row_a(0);
                std::bitset<64> col_a(0);

                std::bitset<64> row_b(0);
                std::bitset<64> col_b(0);

                auto lane = newSpan.begin();
                for (int k = 0 ; k < newSpan.size(); k++)
                {
                    int indexA = getIndexInSet(A.span, *lane);
                    int indexB = getIndexInSet(B.span, *lane);

                    if (indexA != -1) row_a[indexA] = row_res[k];
                    else              row_b[indexB] = row_res[k];

                    if (indexB != -1) col_b[indexB] = col_res[k];
                    else              col_a[indexA] = col_res[k];

                    lane++;
                }
                for (int m = 0; m < (1 << connections.size()); m++)
                {
                    std::bitset<64> address_vacant(m);
                    int cnt = 0;
                    for (auto& c: connections)
                    {
                        col_a[getIndexInSet(A.span, c)] = address_vacant[cnt];
                        row_b[getIndexInSet(B.span, c)] = address_vacant[cnt];
                        cnt++;
                    }
                    
                    std::array<std::bitset<64>, 4> indexes = {row_a, col_a, row_b, col_b};
                    swapFirstNBits(indexes[0], A.rank);
                    swapFirstNBits(indexes[1], A.rank);
                    swapFirstNBits(indexes[2], B.rank);
                    swapFirstNBits(indexes[3], B.rank);
                    
                    shiftBitsSx(indexes[0], A.rank);
                    shiftBitsSx(indexes[2], B.rank);
                    
                    std::bitset<64> indexA = indexes[0] | indexes[1];
                    std::bitset<64> indexB = indexes[2] | indexes[3];

                    resultValues.at(i) += A.getValue(indexA.to_ulong()) * B.getValue(indexB.to_ulong());
                }
            }
            result.setValues(resultValues);
            return result;
        }
            void printValues(std::ostream& os = std::cout) const
            {
                for (int i = 0 ; i < values.size(); i++)
                {
                    if (i != 0 && (i % (1 << rank)) == 0)
                    {
                        os << std::endl;
                    }
                    os << values[i] << ", ";
                }
            }

        int getRank() const { return rank; }
    public:
        int rank;
};