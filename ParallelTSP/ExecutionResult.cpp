#include "ExecutionResult.h"
#include "TspInstance.h"

#include <chrono>
#include <utility>

namespace {
    using Clock = std::chrono::high_resolution_clock;

    GAResult runAlgorithm(GeneticAlgorithm& algorithm, bool parallel) {
        return parallel
            ? algorithm.runParallel()
            : algorithm.runSerial();
    }

    double elapsedSecondsSince(Clock::time_point start) {
        auto end = Clock::now();
        return std::chrono::duration<double>(end - start).count();
    }
}

ExecutionResult runSolver(GeneticAlgorithm& algorithm, bool parallel) {
    auto start = Clock::now();
    GAResult result = runAlgorithm(algorithm, parallel);

    return ExecutionResult{
        result,
        elapsedSecondsSince(start)
    };
}

ExecutionResult runSolverWithSetup(
    const std::string& dataPath,
    GAConfig config,
    bool parallel
) {
    auto start = Clock::now();

    TspInstance instance(dataPath, parallel);
    GeneticAlgorithm algorithm(instance, std::move(config));
    GAResult result = runAlgorithm(algorithm, parallel);

    return ExecutionResult{
        result,
        elapsedSecondsSince(start)
    };
}
