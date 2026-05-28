#pragma once

#include "CommandLineOptions.h"
#include "ExecutionResult.h"
#include "TspInstance.h"

#include <string>
#include <vector>

std::vector<int> convertRouteToCityIds(
    const std::vector<int>& route,
    const TspInstance& instance
);

void rotateRouteToStartId(std::vector<int>& routeIds, int startId);

void printSolverResult(
    const std::string& label,
    const TspInstance& instance,
    const GAResult& result,
    const std::vector<int>& routeIds,
    double elapsedSeconds
);

void saveResults(
    const GAResult& result,
    const std::vector<int>& routeIds,
    double elapsedSeconds
);

void printBenchmarkResult(
    const ExecutionResult& serial,
    const ExecutionResult& parallel
);

void saveBenchmarkResults(
    const CommandLineOptions& options,
    const ExecutionResult& serial,
    const ExecutionResult& parallel
);