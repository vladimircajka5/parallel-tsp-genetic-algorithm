#include "CommandLineOptions.h"
#include "ExecutionResult.h"
#include "GeneticAlgorithm.h"
#include "ResultWriter.h"
#include "TspInstance.h"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        CommandLineOptions options = parseArguments(argc, argv);

        TspInstance instance(options.dataPath, options.useParallel || options.benchmark);
        GeneticAlgorithm algorithm(instance, options.config);

        if (options.benchmark) {
            std::cout << "Running serial solver...\n";
            ExecutionResult serialResult = runSolver(algorithm, false);

            std::cout << "Running parallel solver...\n";
            ExecutionResult parallelResult = runSolver(algorithm, true);

            printBenchmarkResult(serialResult, parallelResult);
            saveBenchmarkResults(options, serialResult, parallelResult);

            std::cout << "Saved: results/benchmark.csv\n";

            return 0;
        }

        std::cout << "Finding the best route...\n\n";

        ExecutionResult execution = runSolver(algorithm, options.useParallel);

        std::vector<int> routeIds = convertRouteToCityIds(
            execution.result.bestRoute,
            instance
        );

        rotateRouteToStartId(routeIds, options.startId);

        printSolverResult(
            options.useParallel ? "PARALLEL" : "SERIAL",
            instance,
            execution.result,
            routeIds,
            execution.elapsedSeconds
        );

        saveResults(
            execution.result,
            routeIds,
            execution.elapsedSeconds
        );

        std::cout << "Saved: results/best_route.txt and results/history.csv\n";

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}