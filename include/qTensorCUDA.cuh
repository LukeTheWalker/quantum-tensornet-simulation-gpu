#include "qTensor.hpp"
#include "Contraction.hpp"

auto contractionGPU(QTensor A, QTensor B) -> QTensor;
auto contractTreeGPU(Contraction* root) -> void;
