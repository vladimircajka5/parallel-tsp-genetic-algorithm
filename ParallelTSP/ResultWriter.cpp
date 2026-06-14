#include "ResultWriter.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <tbb/info.h>

namespace fs = std::filesystem;

namespace {
    int effectiveThreadCount(const CommandLineOptions& options) {
        if (options.threadCount > 0) {
            return options.threadCount;
        }

        return std::max(1, tbb::info::default_concurrency());
    }
}

std::vector<int> convertRouteToCityIds(
    const std::vector<int>& route,
    const TspInstance& instance
) {
    const auto& cities = instance.getCities();

    std::vector<int> ids;
    ids.reserve(route.size());

    for (int cityIndex : route) {
        ids.push_back(cities[cityIndex].id);
    }

    return ids;
}

void rotateRouteToStartId(std::vector<int>& routeIds, int startId) {
    auto it = std::find(routeIds.begin(), routeIds.end(), startId);

    if (it != routeIds.end()) {
        std::rotate(routeIds.begin(), it, routeIds.end());
    }
}

void printSolverResult(
    const std::string& label,
    const TspInstance& instance,
    const GAResult& result,
    const std::vector<int>& routeIds,
    double elapsedSeconds
) {
    std::cout << "=== GA TSP RESULT - " << label << " ===\n";
    std::cout << "Cities: " << instance.cityCount() << '\n';
    std::cout << "Used generations: " << result.usedGenerations << '\n';

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Best total distance: " << result.bestLength << '\n';

    std::cout << std::setprecision(6);
    std::cout << "Elapsed seconds: " << elapsedSeconds << '\n';

    std::cout << "Best tour (city IDs, closed tour returns to start):\n";

    for (size_t i = 0; i < routeIds.size(); ++i) {
        if (i > 0) {
            std::cout << " -> ";
        }

        std::cout << routeIds[i];
    }

    std::cout << " -> " << routeIds.front() << '\n';
}

void saveResults(
    const GAResult& result,
    const std::vector<int>& routeIds,
    double elapsedSeconds
) {
    fs::create_directories("results");

    {
        std::ofstream output("results/best_route.txt");

        output << "Used generations: " << result.usedGenerations << '\n';
        output << std::fixed << std::setprecision(6);
        output << "Best total distance: " << result.bestLength << '\n';
        output << "Elapsed seconds: " << elapsedSeconds << '\n';
        output << "Best tour (city IDs):\n";

        for (size_t i = 0; i < routeIds.size(); ++i) {
            if (i > 0) {
                output << ' ';
            }

            output << routeIds[i];
        }

        output << "\nClosed tour:\n";

        for (int id : routeIds) {
            output << id << ' ';
        }

        output << routeIds.front() << '\n';
    }

    {
        std::ofstream history("results/history.csv");

        history << "gen,best,mean\n";
        history << std::fixed << std::setprecision(6);

        for (const auto& [generation, best, mean] : result.history) {
            history << generation << ',' << best << ',' << mean << '\n';
        }
    }
}

void printBenchmarkResult(
    const ExecutionResult& serial,
    const ExecutionResult& parallel
) {
    double speedup = serial.elapsedSeconds / parallel.elapsedSeconds;

    std::cout << "\n=== BENCHMARK RESULT ===\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Serial time:   " << serial.elapsedSeconds << " s\n";
    std::cout << "Parallel time: " << parallel.elapsedSeconds << " s\n";
    std::cout << "Speedup:       " << speedup << "x\n";
    std::cout << "Serial best:   " << serial.result.bestLength << '\n';
    std::cout << "Parallel best: " << parallel.result.bestLength << '\n';
}

void saveBenchmarkResults(
    const CommandLineOptions& options,
    const ExecutionResult& serial,
    const ExecutionResult& parallel
) {
    fs::create_directories("results");

    const std::string filePath = "results/benchmark.csv";
    bool shouldWriteHeader = !fs::exists(filePath) || fs::file_size(filePath) == 0;

    std::ofstream output(filePath, std::ios::app);

    int islandCount = options.config.islandCount;
    int populationPerIsland = options.config.populationSize;
    int totalPopulation = populationPerIsland * islandCount;

    double speedup = serial.elapsedSeconds / parallel.elapsedSeconds;

    int measuredThreadCount = effectiveThreadCount(options);
    double efficiency = speedup / static_cast<double>(measuredThreadCount);

    std::string serialMode = islandCount > 1 ? "serial_baseline" : "serial";
    std::string parallelMode = islandCount > 1 ? "island_model" : "parallel";

    if (shouldWriteHeader) {
        output
            << "population_per_island,total_population,generations,patience,seed,threads,"
            << "islands,migration_interval,migrants,mode,used_generations,"
            << "best_length,elapsed_seconds,speedup,efficiency\n";
    }

    output << populationPerIsland << ','
        << totalPopulation << ','
        << options.config.generations << ','
        << options.config.patience << ','
        << options.config.seed << ','
        << measuredThreadCount << ','
        << islandCount << ','
        << options.config.migrationInterval << ','
        << options.config.migrantsPerIsland << ','
        << serialMode << ','
        << serial.result.usedGenerations << ','
        << std::fixed << std::setprecision(6)
        << serial.result.bestLength << ','
        << serial.elapsedSeconds << ','
        << "1.000000" << ','
        << "1.000000" << '\n';

    output << populationPerIsland << ','
        << totalPopulation << ','
        << options.config.generations << ','
        << options.config.patience << ','
        << options.config.seed << ','
        << measuredThreadCount << ','
        << islandCount << ','
        << options.config.migrationInterval << ','
        << options.config.migrantsPerIsland << ','
        << parallelMode << ','
        << parallel.result.usedGenerations << ','
        << std::fixed << std::setprecision(6)
        << parallel.result.bestLength << ','
        << parallel.elapsedSeconds << ','
        << speedup << ','
        << efficiency << '\n';
}
