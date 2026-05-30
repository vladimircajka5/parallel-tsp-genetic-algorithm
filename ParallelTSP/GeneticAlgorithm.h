#pragma once

#include "Individual.h"
#include "TspInstance.h"

#include <random>
#include <tuple>
#include <utility>
#include <string>
#include <vector>

struct GAConfig {
    int populationSize = 500;
    int generations = 100;

    double crossoverProbability = 0.9;
    double mutationProbability = 0.18;
    double eliteFraction = 0.03;

    int patience = 50;
    unsigned int seed = 42;

    int islandCount = 1;
    int migrationInterval = 25;
    int migrantsPerIsland = 5;
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
        const std::vector<double>& rankWeights,
        int islandIndex = 0
    ) const;

    std::vector<Individual> createNextPopulationParallel(
        const std::vector<Individual>& population,
        int eliteCount,
        int generation,
        const std::vector<double>& rankWeights,
        int islandIndex = 0
    ) const;

    Individual createChild(
        const std::vector<Individual>& population,
        int generation,
        int childIndex,
        std::discrete_distribution<int>& rankDistribution,
        int islandIndex = 0
    ) const;

    std::vector<Individual> createInitialPopulationSerial(int islandIndex = 0) const;
    std::vector<Individual> createInitialPopulationParallel(int islandIndex = 0) const;

    Individual createInitialIndividual(int individualIndex, int islandIndex = 0) const;

    void evaluatePopulation(std::vector<Individual>& population) const;
    void evaluatePopulationParallel(std::vector<Individual>& population) const;

    void sortPopulation(std::vector<Individual>& population) const;
    void sortPopulationParallel(std::vector<Individual>& population) const;

    double calculateMeanLength(const std::vector<Individual>& population) const;
    double calculateMeanLengthParallel(const std::vector<Individual>& population) const;

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

    bool isValidRoute(const std::vector<int>& route) const;

    void validateRoute(const std::vector<int>& route, const std::string& context) const;

    GAResult runIslandModel();

    void evolveIsland(
        std::vector<Individual>& population,
        int islandIndex,
        int startGeneration,
        int generationCount
    ) const;

    void migrateBestIndividuals(std::vector<std::vector<Individual>>& islands) const;

    Individual findBestIndividual(const std::vector<std::vector<Individual>>& islands) const;

    double calculateMeanLengthAcrossIslands(const std::vector<std::vector<Individual>>& islands) const;

    void logIslandState(const std::vector<std::vector<Individual>>& islands, int generation, const std::string& stage) const;
};