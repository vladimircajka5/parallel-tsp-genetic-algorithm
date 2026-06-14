#pragma once

#include "GeneticAlgorithm.h"

#include <string>

struct ExecutionResult {
    GAResult result;
    double elapsedSeconds = 0.0;
};

ExecutionResult runSolver(GeneticAlgorithm& algorithm, bool parallel);

ExecutionResult runSolverWithSetup(
    const std::string& dataPath,
    GAConfig config,
    bool parallel
);
