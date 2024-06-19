#pragma once

#include <vector>
#include <string>

#include "qTensor.hpp"

struct Contraction {
    int id;
    int programId;
    std::vector<unsigned char> span;
    int leftId;
    int rightId;
    Contraction* left;
    Contraction* right;
    std::string kind;
    QTensor data;
};
