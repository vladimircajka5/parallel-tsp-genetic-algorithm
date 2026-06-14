#include "CommandLineOptions.h"
#include "ExecutionResult.h"
#include "GeneticAlgorithm.h"
#include "ResultWriter.h"
#include "TspInstance.h"

#include <exception>
#include <iostream>
#include <memory>
#include <tbb/global_control.h>

int main(int argc, char** argv) {
    try {
        CommandLineOptions options = parseArguments(argc, argv);

        std::unique_ptr<tbb::global_control> threadControl;

        if (options.threadCount > 0) {
            threadControl = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism,
                options.threadCount
            );
        }

        if (options.benchmark) {
            GAConfig serialConfig = options.config;
            GAConfig parallelConfig = options.config;

            if (options.config.islandCount > 1) {
                serialConfig.populationSize =
                    options.config.populationSize * options.config.islandCount;

                serialConfig.islandCount = 1;
                serialConfig.migrantsPerIsland = 0;

                std::cout << "=== ISLAND MODEL BENCHMARK ===\n";
                std::cout << "Running serial baseline solver...\n";
                std::cout << "Serial baseline population: "
                    << serialConfig.populationSize << '\n';

                std::cout << "Running island model solver...\n";
                std::cout << "Island count: " << parallelConfig.islandCount << '\n';
                std::cout << "Population per island: "
                    << parallelConfig.populationSize << '\n';
                std::cout << "Total island population: "
                    << parallelConfig.populationSize * parallelConfig.islandCount
                    << "\n\n";
            }
            else {
                std::cout << "=== STANDARD GA BENCHMARK ===\n";
                std::cout << "Running serial solver...\n";
                std::cout << "Running parallel solver...\n\n";
            }

            ExecutionResult serialResult = runSolverWithSetup(
                options.dataPath,
                serialConfig,
                false
            );

            ExecutionResult parallelResult = runSolverWithSetup(
                options.dataPath,
                parallelConfig,
                true
            );

            printBenchmarkResult(serialResult, parallelResult);
            saveBenchmarkResults(options, serialResult, parallelResult);

            std::cout << "Saved: results/benchmark.csv\n";

            return 0;
        }

        std::cout << "Finding the best route...\n\n";

        TspInstance instance(options.dataPath, options.useParallel);
        GeneticAlgorithm algorithm(instance, options.config);
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
