#include "GeneticAlgorithm.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace {
    bool shorterIndividual(const Individual& a, const Individual& b) {
        return a.length < b.length;
    }
}

GeneticAlgorithm::GeneticAlgorithm(const TspInstance& instance, GAConfig config)
    : instance(instance), config(std::move(config)), n(instance.cityCount()) {
    if (this->config.populationSize <= 0) {
        throw std::invalid_argument("Population size must be positive.");
    }

    if (this->config.generations <= 0) {
        throw std::invalid_argument("Number of generations must be positive.");
    }

    if (this->config.crossoverProbability < 0.0 || this->config.crossoverProbability > 1.0) {
        throw std::invalid_argument("Crossover probability must be between 0 and 1.");
    }

    if (this->config.mutationProbability < 0.0 || this->config.mutationProbability > 1.0) {
        throw std::invalid_argument("Mutation probability must be between 0 and 1.");
    }

    if (this->config.eliteFraction < 0.0 || this->config.eliteFraction > 1.0) {
        throw std::invalid_argument("Elite fraction must be between 0 and 1.");
    }
}

GAResult GeneticAlgorithm::runSerial() {
    return run(false);
}

GAResult GeneticAlgorithm::runParallel() {
    return run(true);
}

GAResult GeneticAlgorithm::run(bool useParallelEvaluation) {
    std::mt19937 rng(config.seed);

    std::vector<Individual> population = createInitialPopulation(rng);
    evaluatePopulation(population);
    std::sort(population.begin(), population.end(), shorterIndividual);

    Individual bestOverall = population.front();
    std::vector<std::tuple<int, double, double>> history;

    int eliteCount = std::max(1, static_cast<int>(std::round(config.populationSize * config.eliteFraction)));
    eliteCount = std::min(eliteCount, config.populationSize);

    int noImprove = 0;
    int usedGenerations = config.generations;

    for (int generation = 1; generation <= config.generations; ++generation) {
        std::vector<Individual> nextPopulation(config.populationSize);

        for (int i = 0; i < eliteCount; ++i) {
            nextPopulation[i] = population[i];
        }

        std::vector<double> rankWeights = buildRankWeights(static_cast<int>(population.size()));
        std::discrete_distribution<int> rankDistribution(rankWeights.begin(), rankWeights.end());

        std::uniform_real_distribution<double> probability(0.0, 1.0);

        int fillIndex = eliteCount;
        while (fillIndex < config.populationSize) {
            const Individual& parent1 = population[pickParentByRank(rng, rankDistribution)];
            const Individual& parent2 = population[pickParentByRank(rng, rankDistribution)];

            std::vector<int> child1;
            std::vector<int> child2;

            if (probability(rng) < config.crossoverProbability) {
                auto children = scxCrossover(parent1.route, parent2.route, rng);
                child1 = std::move(children.first);
                child2 = std::move(children.second);
            } else {
                child1 = parent1.route;
                child2 = parent2.route;
            }

            if (probability(rng) < config.mutationProbability) {
                inversionMutation(child1, rng);
            }
            if (probability(rng) < config.mutationProbability) {
                inversionMutation(child2, rng);
            }

            nextPopulation[fillIndex++].route = std::move(child1);
            if (fillIndex < config.populationSize) {
                nextPopulation[fillIndex++].route = std::move(child2);
            }
        }

        population = std::move(nextPopulation);
        evaluatePopulation(population);
        std::sort(population.begin(), population.end(), shorterIndividual);

        double mean = 0.0;
        for (const Individual& individual : population) {
            mean += individual.length;
        }
        mean /= static_cast<double>(population.size());

        const Individual& generationBest = population.front();
        history.emplace_back(generation, generationBest.length, mean);

        if (generationBest.length + 1e-9 < bestOverall.length) {
            bestOverall = generationBest;
            noImprove = 0;
        } else {
            ++noImprove;
        }

        if (noImprove >= config.patience) {
            usedGenerations = generation;
            break;
        }
    }

    return GAResult{
        bestOverall.route,
        bestOverall.length,
        history,
        usedGenerations
    };
}

std::vector<Individual> GeneticAlgorithm::createInitialPopulation(std::mt19937& rng) const {
    std::vector<int> base(n);
    std::iota(base.begin(), base.end(), 0);

    std::vector<Individual> population(config.populationSize);
    for (Individual& individual : population) {
        individual.route = base;
        std::shuffle(individual.route.begin(), individual.route.end(), rng);
    }
    return population;
}

void GeneticAlgorithm::evaluatePopulation(std::vector<Individual>& population) const {
    for (Individual& individual : population) {
        individual.length = calculateRouteLength(individual.route);
    }
}

void GeneticAlgorithm::evaluatePopulationParallel(std::vector<Individual>& population) const {
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, population.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i != range.end(); ++i) {
                population[i].length = calculateRouteLength(population[i].route);
            }
        }
    );
}

double GeneticAlgorithm::calculateRouteLength(const std::vector<int>& route) const {
    double total = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        total += instance.distance(route[i], route[i + 1]);
    }
    total += instance.distance(route[n - 1], route[0]);
    return total;
}

int GeneticAlgorithm::pickParentByRank(
    std::mt19937& rng,
    std::discrete_distribution<int>& rankDistribution
) const {
    return rankDistribution(rng);
}

std::pair<std::vector<int>, std::vector<int>> GeneticAlgorithm::scxCrossover(
    const std::vector<int>& parent1,
    const std::vector<int>& parent2,
    std::mt19937& rng
) const {
    return {scxChild(true, parent1, parent2, rng), scxChild(false, parent1, parent2, rng)};
}

std::vector<int> GeneticAlgorithm::scxChild(
    bool useNext,
    const std::vector<int>& parent1,
    const std::vector<int>& parent2,
    std::mt19937& rng
) const {
    std::vector<int> px = parent1;
    std::vector<int> py = parent2;

    std::uniform_int_distribution<int> startDistribution(0, static_cast<int>(px.size()) - 1);
    int current = px[startDistribution(rng)];

    std::vector<int> child;
    child.reserve(n);
    child.push_back(current);

    while (px.size() > 1) {
        auto itX = std::find(px.begin(), px.end(), current);
        auto itY = std::find(py.begin(), py.end(), current);

        int ix = static_cast<int>(std::distance(px.begin(), itX));
        int iy = static_cast<int>(std::distance(py.begin(), itY));
        int sizeX = static_cast<int>(px.size());
        int sizeY = static_cast<int>(py.size());

        int candidateX = useNext ? px[(ix + 1) % sizeX] : px[(ix - 1 + sizeX) % sizeX];
        int candidateY = useNext ? py[(iy + 1) % sizeY] : py[(iy - 1 + sizeY) % sizeY];

        px.erase(px.begin() + ix);
        py.erase(py.begin() + iy);

        current = instance.distance(current, candidateX) < instance.distance(current, candidateY)
            ? candidateX
            : candidateY;
        child.push_back(current);
    }

    return child;
}


void GeneticAlgorithm::inversionMutation(std::vector<int>& route, std::mt19937& rng) const {
    std::uniform_int_distribution<int> indexDistribution(0, n - 1);

    int i = indexDistribution(rng);
    int j = indexDistribution(rng);

    if (i > j) {
        std::swap(i, j);
    }

    if (i != j) {
        std::reverse(route.begin() + i, route.begin() + j + 1);
    }
}


std::vector<double> GeneticAlgorithm::buildRankWeights(int populationSize) const {
    std::vector<double> weights(populationSize);
    for (int i = 0; i < populationSize; ++i) {
        weights[i] = static_cast<double>(populationSize - i);
    }
    return weights;
}