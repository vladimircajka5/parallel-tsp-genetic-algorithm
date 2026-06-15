#include "CommandLineOptions.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

void printUsage(const char* programName) {
    std::cout
        << "Usage: " << programName << " [options]\n\n"
        << "Options:\n"
        << "  --data <path>          Path to data_tsp.txt (default: data_tsp.txt)\n"
        << "  --pop <int>            Population size (default: 500)\n"
        << "  --gen <int>            Number of generations (default: 100)\n"
        << "  --pc <double>          Crossover probability (default: 0.9)\n"
        << "  --pm <double>          Mutation probability (default: 0.18)\n"
        << "  --elite <double>       Elite fraction (default: 0.03)\n"
        << "  --patience <int>       Early stop patience (default: 50)\n"
        << "  --seed <int>           Random seed (default: 42)\n"
        << "  --start_id <int>       Rotate output route to this city ID (default: 1)\n"
        << "  --parallel             Run TBB parallel version\n"
        << "  --benchmark            Run both serial and parallel versions and save comparison\n"
        << "  --threads <int>        Maximum number of TBB threads (default: automatic)\n"
        << "  --islands <int>       Number of islands for island model (default: 1)\n"
        << "  --migration <int>     Migration interval in generations (default: 25)\n"
        << "  --migrants <int>      Number of migrants per island (default: 5)\n"
        << "  --help                 Show this help\n";
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
        }
        else if (arg == "--data") {
            options.dataPath = requireValue(arg);
        }
        else if (arg == "--pop") {
            options.config.populationSize = std::stoi(requireValue(arg));
        }
        else if (arg == "--gen") {
            options.config.generations = std::stoi(requireValue(arg));
        }
        else if (arg == "--pc") {
            options.config.crossoverProbability = std::stod(requireValue(arg));
        }
        else if (arg == "--pm") {
            options.config.mutationProbability = std::stod(requireValue(arg));
        }
        else if (arg == "--elite") {
            options.config.eliteFraction = std::stod(requireValue(arg));
        }
        else if (arg == "--patience") {
            options.config.patience = std::stoi(requireValue(arg));
        }
        else if (arg == "--seed") {
            options.config.seed = static_cast<unsigned int>(std::stoul(requireValue(arg)));
        }
        else if (arg == "--start_id") {
            options.startId = std::stoi(requireValue(arg));
        }
        else if (arg == "--parallel") {
            options.useParallel = true;
        }
        else if (arg == "--benchmark") {
            options.benchmark = true;
        }
        else if (arg == "--threads") {
             options.threadCount = std::stoi(requireValue(arg));
        }
        else if (arg == "--islands") {
             options.config.islandCount = std::stoi(requireValue(arg));
        }
        else if (arg == "--migration") {
             options.config.migrationInterval = std::stoi(requireValue(arg));
        }
        else if (arg == "--migrants") {
             options.config.migrantsPerIsland = std::stoi(requireValue(arg));
        }
        else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    return options;
}