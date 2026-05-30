#include "ResultWriter.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;

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

    double speedup = serial.elapsedSeconds / parallel.elapsedSeconds;

    double efficiency = 0.0;
    if (options.threadCount > 0) {
        efficiency = speedup / static_cast<double>(options.threadCount);
    }

    if (shouldWriteHeader) {
        output
            << "population,generations,patience,seed,threads,"
            << "serial_used_generations,parallel_used_generations,"
            << "serial_best_length,parallel_best_length,"
            << "serial_elapsed_seconds,parallel_elapsed_seconds,"
            << "speedup,efficiency\n";
    }

    output << options.config.populationSize << ','
        << options.config.generations << ','
        << options.config.patience << ','
        << options.config.seed << ','
        << options.threadCount << ','
        << serial.result.usedGenerations << ','
        << parallel.result.usedGenerations << ','
        << std::fixed << std::setprecision(6)
        << serial.result.bestLength << ','
        << parallel.result.bestLength << ','
        << serial.elapsedSeconds << ','
        << parallel.elapsedSeconds << ','
        << speedup << ','
        << efficiency << '\n';
}