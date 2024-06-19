#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <complex>
#include <set>
#include <array>
#include "bitset.hpp"

class QTensor
{
    public:
        std::set<unsigned char> span;
        std::vector<std::complex<float> > values;
        QTensor() {}
        QTensor(std::set<unsigned char> span): span(span){this->rank = span.size();}
        template <typename T>
        QTensor(std::vector<T> values, std::vector<unsigned char> span_) : span(span_.begin(), span_.end())
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
                exit(1);
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
            auto getIndexInSet = [](std::set<unsigned char> set, unsigned char element) {
                int index = 0;
                for (auto it = set.begin(); it != set.end(); ++it) {
                    if (*it == element) {
                        return index;
                    }
                    index++;
                }
                return 255; // Element not found in the set
            };

            auto findCommonValues = [](std::set<unsigned char> set1, std::set<unsigned char> set2) -> std::vector<unsigned char> {
                std::vector<unsigned char> commonValues;
                for (auto value : set1) {
                    if (set2.find(value) != set2.end()) {
                        commonValues.push_back(value);
                    }
                }
                return commonValues;
            };

            std::set<unsigned char> newSpan;
            newSpan.insert(A.span.begin(), A.span.end());
            newSpan.insert(B.span.begin(), B.span.end());
            QTensor result = QTensor(newSpan);
            std::vector<std::complex<float>> resultValues(1 << (result.rank*2), {0.0, 0.0});

            std::vector<unsigned char> connections = findCommonValues(A.span, B.span);

            #pragma omp parallel for
            for (int i = 0; i < (1 << (result.rank*2)); i++)
            {   
                cpu_classes::bitset bits(i);

                cpu_classes::bitset a(0);
                cpu_classes::bitset b(0);

                auto lane = newSpan.begin();
                for (int k = 0 ; k < result.rank; k++)
                {
                    int indexA = getIndexInSet(A.span, *lane);
                    int indexB = getIndexInSet(B.span, *lane);

                    if (indexA != 255) a.set(2*A.rank - indexA - 1, bits.get(result.rank*2 - 1 - k));
                    else               b.set(2*B.rank - indexB - 1, bits.get(result.rank*2 - 1 - k));

                    if (indexB != 255) b.set(B.rank - indexB - 1, bits.get(result.rank - 1 - k));
                    else               a.set(A.rank - indexA - 1, bits.get(result.rank - 1 - k));

                    lane++;
                }

                for (int m = 0; m < (1 << connections.size()); m++)
                {
                    cpu_classes::bitset address_vacant(m);
                    int cnt = 0;
                    for (int c = 0; c < connections.size(); c++)
                    {
                        unsigned char indexA = getIndexInSet(A.span, connections[c]);
                        unsigned char indexB = getIndexInSet(B.span, connections[c]);
                        a.set(A.rank - indexA - 1, address_vacant.get(cnt));
                        b.set(2*B.rank - indexB - 1, address_vacant.get(cnt));
                        cnt++;
                    }

                    resultValues.at(i) += A.getValue(a.to_ulong()) * B.getValue(b.to_ulong());
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