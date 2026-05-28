#pragma once

#include "Individual.h"
#include "TspInstance.h"

#include <random>
#include <tuple>
#include <utility>
#include <vector>

struct GAConfig {
    int populationSize = 500;
    int generations = 100;

    double crossoverProbability = 0.9;
    double mutationProbability = 0.18;
    double eliteFraction = 0.03;

    int patience = 50;
    unsigned int seed = 42;
};

struct GAResult {
    std::vector<int> bestRoute;
    double bestLength = 0.0;

    // generation, best, mean
    std::vector<std::tuple<int, double, double>> history;

    int usedGenerations = 0;
};

class GeneticAlgorithm {
public:
    GeneticAlgorithm(const TspInstance& instance, GAConfig config);

    GAResult runSerial();

private:
    const TspInstance& instance;
    GAConfig config;
    int n;

    std::vector<Individual> createInitialPopulation(std::mt19937& rng) const;

    void evaluatePopulation(std::vector<Individual>& population) const;
    double calculateRouteLength(const std::vector<int>& route) const;

    int pickParentByRank(
        std::mt19937& rng,
        std::discrete_distribution<int>& rankDistribution
    ) const;

    std::pair<std::vector<int>, std::vector<int>> scxCrossover(
        const std::vector<int>& parent1,
        const std::vector<int>& parent2,
        std::mt19937& rng
    ) const;

    std::vector<int> scxChild(
        bool useNext,
        const std::vector<int>& parent1,
        const std::vector<int>& parent2,
        std::mt19937& rng
    ) const;

    void inversionMutation(std::vector<int>& route, std::mt19937& rng) const;

    std::vector<double> buildRankWeights(int populationSize) const;
};