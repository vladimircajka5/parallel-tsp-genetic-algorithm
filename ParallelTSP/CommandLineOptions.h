#pragma once

#include "GeneticAlgorithm.h"

#include <string>

struct CommandLineOptions {
    std::string dataPath = "data_tsp.txt";
    int startId = 1;
    bool useParallel = false;
    bool benchmark = false;
    GAConfig config;
};

void printUsage(const char* programName);
CommandLineOptions parseArguments(int argc, char** argv);