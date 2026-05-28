#include "ExecutionResult.h"

#include <chrono>

ExecutionResult runSolver(GeneticAlgorithm& algorithm, bool parallel) {
    auto start = std::chrono::high_resolution_clock::now();

    GAResult result = parallel
        ? algorithm.runParallel()
        : algorithm.runSerial();

    auto end = std::chrono::high_resolution_clock::now();

    double elapsedSeconds = std::chrono::duration<double>(end - start).count();

    return ExecutionResult{
        result,
        elapsedSeconds
    };
}