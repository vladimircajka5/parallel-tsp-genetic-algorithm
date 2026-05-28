#include "GeneticAlgorithm.h"
#include "TspInstance.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

struct CommandLineOptions {
    std::string dataPath = "data_tsp.txt";
    int startId = 1;
    bool useParallel = false;
    GAConfig config;
};

void printUsage(const char* programName) {
    std::cout
        << "Usage: " << programName << " [options]\n\n"
        << "Options:\n"
        << "  --data <path>          Path to data_tsp.txt (default: data/data_tsp.txt)\n"
        << "  --pop <int>            Population size (default: 500)\n"
        << "  --gen <int>            Number of generations (default: 100)\n"
        << "  --pc <double>          Crossover probability (default: 0.9)\n"
        << "  --pm <double>          Mutation probability (default: 0.18)\n"
        << "  --elite <double>       Elite fraction (default: 0.03)\n"
        << "  --patience <int>       Early stop patience (default: 50)\n"
        << "  --seed <int>           Random seed (default: 42)\n"
        << "  --start_id <int>       Rotate output route to this city ID (default: 1)\n"
        << "  --help                 Show this help\n"
        << "  --parallel            Run TBB parallel version\n";
}

CommandLineOptions parseArguments(int argc, char** argv) {
    CommandLineOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto requireValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for argument: " + name);
            }
            return argv[++i];
        };

        if (arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--data") {
            options.dataPath = requireValue(arg);
        } else if (arg == "--pop") {
            options.config.populationSize = std::stoi(requireValue(arg));
        } else if (arg == "--gen") {
            options.config.generations = std::stoi(requireValue(arg));
        } else if (arg == "--pc") {
            options.config.crossoverProbability = std::stod(requireValue(arg));
        } else if (arg == "--pm") {
            options.config.mutationProbability = std::stod(requireValue(arg));
        } else if (arg == "--elite") {
            options.config.eliteFraction = std::stod(requireValue(arg));
        } else if (arg == "--patience") {
            options.config.patience = std::stoi(requireValue(arg));
        } else if (arg == "--seed") {
            options.config.seed = static_cast<unsigned int>(std::stoul(requireValue(arg)));
        } else if (arg == "--start_id") {
            options.startId = std::stoi(requireValue(arg));
        } else if (arg == "--parallel") {
            options.useParallel = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    return options;
}

std::vector<int> convertRouteToCityIds(const std::vector<int>& route, const TspInstance& instance) {
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

void saveResults(const GAResult& result, const std::vector<int>& routeIds, double elapsedSeconds) {
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

int main(int argc, char** argv) {
    try {
        CommandLineOptions options = parseArguments(argc, argv);

        TspInstance instance(options.dataPath, options.useParallel);
        GeneticAlgorithm algorithm(instance, options.config);

        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "Finding the best route...\n" << std::endl;
        GAResult result = options.useParallel ? algorithm.runParallel() : algorithm.runSerial();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsedSeconds = std::chrono::duration<double>(end - start).count();

        std::vector<int> routeIds = convertRouteToCityIds(result.bestRoute, instance);
        rotateRouteToStartId(routeIds, options.startId);

        std::cout << "=== GA TSP RESULT - "<< (options.useParallel ? "PARALLEL" : "SERIAL") << " ===\n";
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

        saveResults(result, routeIds, elapsedSeconds);
        std::cout << "Saved: results/best_route.txt and results/history.csv\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
