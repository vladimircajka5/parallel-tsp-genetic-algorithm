#pragma once

#include "GeneticAlgorithm.h"

struct ExecutionResult {
    GAResult result;
    double elapsedSeconds = 0.0;
};

ExecutionResult runSolver(GeneticAlgorithm& algorithm, bool parallel);
