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
    GAResult runParallel();

private:
    const TspInstance& instance;
    GAConfig config;
    int n;

    GAResult run(bool useParallel);

    std::vector<Individual> createNextPopulationSerial(
        const std::vector<Individual>& population,
        int eliteCount,
        int generation,
        const std::vector<double>& rankWeights
    ) const;

    std::vector<Individual> createNextPopulationParallel(
        const std::vector<Individual>& population,
        int eliteCount,
        int generation,
        const std::vector<double>& rankWeights
    ) const;

    Individual createChild(
        const std::vector<Individual>& population,
        int generation,
        int childIndex,
        std::discrete_distribution<int>& rankDistribution
    ) const;

    std::vector<Individual> createInitialPopulation(std::mt19937& rng) const;

    void evaluatePopulation(std::vector<Individual>& population) const;
    void evaluatePopulationParallel(std::vector<Individual>& population) const;

    double calculateRouteLength(const std::vector<int>& route) const;

    int pickParentByRank(
        std::mt19937& rng,
        std::discrete_distribution<int>& rankDistribution
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