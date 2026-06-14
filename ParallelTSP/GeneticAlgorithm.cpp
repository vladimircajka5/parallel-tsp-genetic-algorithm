#include "GeneticAlgorithm.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <random>
#include <utility>
#include <functional>
#include <iostream>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/parallel_reduce.h>

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
    if (this->config.islandCount <= 0) {
        throw std::invalid_argument("Island count must be positive.");
    }

    if (this->config.migrationInterval <= 0) {
        throw std::invalid_argument("Migration interval must be positive.");
    }

    if (this->config.migrantsPerIsland < 0) {
        throw std::invalid_argument("Migrants per island cannot be negative.");
    }

    if (this->config.migrantsPerIsland > this->config.populationSize) {
        throw std::invalid_argument("Migrants per island cannot exceed population size.");
    }
}

GAResult GeneticAlgorithm::runSerial() {
    return run(false);
}

GAResult GeneticAlgorithm::runParallel() {
    if (config.islandCount > 1) {
        return runIslandModel();
    }

    return run(true);
}

GAResult GeneticAlgorithm::run(bool useParallel) {
    std::vector<Individual> population = useParallel ? createInitialPopulationParallel() : createInitialPopulationSerial();

    if (useParallel) {
        evaluatePopulationParallel(population);
    }
    else {
        evaluatePopulation(population);
    }

    if (useParallel) {
        sortPopulationParallel(population);
    }
    else {
        sortPopulation(population);
    }

    Individual bestOverall = population.front();
    std::vector<std::tuple<int, double, double>> history;

    int eliteCount = std::max(1, static_cast<int>(std::round(config.populationSize * config.eliteFraction)));
    eliteCount = std::min(eliteCount, config.populationSize);

    int noImprove = 0;
    int usedGenerations = config.generations;

    for (int generation = 1; generation <= config.generations; ++generation) {
        std::vector<double> rankWeights = buildRankWeights(static_cast<int>(population.size()));

        std::vector<Individual> nextPopulation = useParallel
            ? createNextPopulationParallel(population, eliteCount, generation, rankWeights)
            : createNextPopulationSerial(population, eliteCount, generation, rankWeights);

        population = std::move(nextPopulation);

        if (useParallel) {
            evaluatePopulationParallel(population);
        }
        else {
            evaluatePopulation(population);
        }

        if (useParallel) {
            sortPopulationParallel(population);
        }
        else {
            sortPopulation(population);
        }

        double mean = useParallel ? calculateMeanLengthParallel(population) : calculateMeanLength(population);

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

std::vector<Individual> GeneticAlgorithm::createNextPopulationSerial(
    const std::vector<Individual>& population,
    int eliteCount,
    int generation,
    const std::vector<double>& rankWeights,
    int islandIndex
) const {
    std::vector<Individual> nextPopulation(config.populationSize);

    for (int i = 0; i < eliteCount; ++i) {
        nextPopulation[i] = population[i];
    }

    std::discrete_distribution<int> rankDistribution(
        rankWeights.begin(),
        rankWeights.end()
    );

    for (int i = eliteCount; i < config.populationSize; ++i) {
        nextPopulation[i] = createChild(
            population,
            generation,
            i,
            rankDistribution,
            islandIndex
        );
    }

    return nextPopulation;
}

std::vector<Individual> GeneticAlgorithm::createNextPopulationParallel(
    const std::vector<Individual>& population,
    int eliteCount,
    int generation,
    const std::vector<double>& rankWeights,
    int islandIndex
) const {
    std::vector<Individual> nextPopulation(config.populationSize);

    for (int i = 0; i < eliteCount; ++i) {
        nextPopulation[i] = population[i];
    }

    tbb::parallel_for(
        tbb::blocked_range<int>(eliteCount, config.populationSize),
        [&](const tbb::blocked_range<int>& range) {
            std::discrete_distribution<int> localRankDistribution(
                rankWeights.begin(),
                rankWeights.end()
            );

            for (int i = range.begin(); i != range.end(); ++i) {
                nextPopulation[i] = createChild(
                    population,
                    generation,
                    i,
                    localRankDistribution,
                    islandIndex
                );
            }
        }
    );

    return nextPopulation;
}

Individual GeneticAlgorithm::createChild(
    const std::vector<Individual>& population,
    int generation,
    int childIndex,
    std::discrete_distribution<int>& rankDistribution,
    int islandIndex
) const {
    std::seed_seq seedSequence{
        config.seed,
        static_cast<unsigned int>(generation),
        static_cast<unsigned int>(childIndex + islandIndex * config.populationSize)
    };

    std::mt19937 rng(seedSequence);
    std::uniform_real_distribution<double> probability(0.0, 1.0);

    const Individual& parent1 = population[pickParentByRank(rng, rankDistribution)];
    const Individual& parent2 = population[pickParentByRank(rng, rankDistribution)];

    std::vector<int> childRoute;

    if (probability(rng) < config.crossoverProbability) {
        bool useNext = (childIndex % 2 == 0);
        childRoute = scxChild(useNext, parent1.route, parent2.route, rng);
    }
    else {
        childRoute = parent1.route;
    }

    if (probability(rng) < config.mutationProbability) {
        inversionMutation(childRoute, rng);
    }

    Individual child;
    child.route = std::move(childRoute);

#ifndef NDEBUG
    validateRoute(child.route, "createChild");
#endif

    return child;
}

Individual GeneticAlgorithm::createInitialIndividual(int individualIndex, int islandIndex) const {
    std::vector<int> route(n);
    std::iota(route.begin(), route.end(), 0);

    std::seed_seq seedSequence{
        config.seed,
        static_cast<unsigned int>(individualIndex + islandIndex * config.populationSize)
    };

    std::mt19937 rng(seedSequence);

    std::shuffle(route.begin(), route.end(), rng);

    Individual individual;
    individual.route = std::move(route);

#ifndef NDEBUG
    validateRoute(individual.route, "createInitialIndividual");
#endif

    return individual;
}

std::vector<Individual> GeneticAlgorithm::createInitialPopulationSerial(int islandIndex) const {
    std::vector<Individual> population(config.populationSize);

    for (int i = 0; i < config.populationSize; ++i) {
        population[i] = createInitialIndividual(i,islandIndex);
    }

    return population;
}

std::vector<Individual> GeneticAlgorithm::createInitialPopulationParallel(int islandIndex) const {
    std::vector<Individual> population(config.populationSize);

    tbb::parallel_for(
        tbb::blocked_range<int>(0, config.populationSize),
        [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                population[i] = createInitialIndividual(i,islandIndex);
            }
        }
    );

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

void GeneticAlgorithm::sortPopulation(std::vector<Individual>& population) const {
    std::sort(population.begin(), population.end(), shorterIndividual);
}

void GeneticAlgorithm::sortPopulationParallel(std::vector<Individual>& population) const {
    tbb::parallel_sort(population.begin(), population.end(), shorterIndividual);
}

double GeneticAlgorithm::calculateMeanLength(const std::vector<Individual>& population) const {
    double sum = 0.0;

    for (const Individual& individual : population) {
        sum += individual.length;
    }

    return sum / static_cast<double>(population.size());
}

double GeneticAlgorithm::calculateMeanLengthParallel(const std::vector<Individual>& population) const {
    double sum = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, population.size()),
        0.0,
        [&](const tbb::blocked_range<size_t>& range, double localSum) {
            for (size_t i = range.begin(); i != range.end(); ++i) {
                localSum += population[i].length;
            }
            return localSum;
        },
        std::plus<double>()
    );

    return sum / static_cast<double>(population.size());
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

// ROUTE VALIDATIONS
bool GeneticAlgorithm::isValidRoute(const std::vector<int>& route) const {
    if (static_cast<int>(route.size()) != n) {
        return false;
    }

    std::vector<bool> seen(n, false);

    for (int city : route) {
        if (city < 0 || city >= n) {
            return false;
        }

        if (seen[city]) {
            return false;
        }

        seen[city] = true;
    }

    return true;
}

void GeneticAlgorithm::validateRoute(
    const std::vector<int>& route,
    const std::string& context
) const {
    if (!isValidRoute(route)) {
        throw std::runtime_error("Invalid route generated in: " + context);
    }
}

// ISLAND FUNCTIONS
GAResult GeneticAlgorithm::runIslandModel() {
    int islandCount = config.islandCount;
    int migrationInterval = config.migrationInterval;

    std::vector<std::vector<Individual>> islands(islandCount);

    // The island model has two levels of parallel work:
    // 1) different islands are initialized/evolved independently;
    // 2) each island reuses the same parallel population operations as the
    //    normal parallel GA. TBB schedules both levels under the global thread
    //    limit configured in main.cpp.
    tbb::parallel_for(
        tbb::blocked_range<int>(0, islandCount),
        [&](const tbb::blocked_range<int>& range) {
            for (int island = range.begin(); island != range.end(); ++island) {
                islands[island] = createInitialPopulationParallel(island);
                evaluatePopulationParallel(islands[island]);
                sortPopulationParallel(islands[island]);
            }
        }
    );

    Individual bestOverall = findBestIndividual(islands);
    std::vector<std::tuple<int, double, double>> history;

    logIslandState(islands, 0, "initial populations");

    int noImprove = 0;
    int usedGenerations = config.generations;

    for (int generation = 1; generation <= config.generations; generation += migrationInterval) {
        int remainingGenerations = config.generations - generation + 1;
        int generationsThisCycle = std::min(migrationInterval, remainingGenerations);

        // Islands evolve independently between migrations. Inside each island,
        // evolveIsland() also uses parallel child creation, evaluation, and
        // sorting, so this keeps the full island cycle parallelized.
        tbb::parallel_for(
            tbb::blocked_range<int>(0, islandCount),
            [&](const tbb::blocked_range<int>& range) {
                for (int island = range.begin(); island != range.end(); ++island) {
                    evolveIsland(
                        islands[island],
                        island,
                        generation,
                        generationsThisCycle
                    );
                }
            }
        );

        int currentGeneration = generation + generationsThisCycle - 1;

        Individual currentBest = findBestIndividual(islands);
        double mean = calculateMeanLengthAcrossIslands(islands);

        history.emplace_back(currentGeneration, currentBest.length, mean);

        if (currentBest.length + 1e-9 < bestOverall.length) {
            bestOverall = currentBest;
            noImprove = 0;
        }
        else {
            noImprove += generationsThisCycle;
        }

        if (noImprove >= config.patience) {
            usedGenerations = currentGeneration;
            break;
        }

        if (currentGeneration < config.generations) {
            logIslandState(islands, currentGeneration, "before migration");

            migrateBestIndividuals(islands);

            logIslandState(islands, currentGeneration, "after migration");
        }
    }

    return GAResult{
        bestOverall.route,
        bestOverall.length,
        history,
        usedGenerations
    };
}

void GeneticAlgorithm::evolveIsland(
    std::vector<Individual>& population,
    int islandIndex,
    int startGeneration,
    int generationCount
) const {
    int eliteCount = std::max(
        1,
        static_cast<int>(std::round(config.populationSize * config.eliteFraction))
    );

    eliteCount = std::min(eliteCount, config.populationSize);

    for (int offset = 0; offset < generationCount; ++offset) {
        int generation = startGeneration + offset;

        std::vector<double> rankWeights = buildRankWeights(
            static_cast<int>(population.size())
        );

        population = createNextPopulationParallel(
            population,
            eliteCount,
            generation,
            rankWeights,
            islandIndex
        );

        evaluatePopulationParallel(population);
        sortPopulationParallel(population);
    }
}

void GeneticAlgorithm::migrateBestIndividuals(
    std::vector<std::vector<Individual>>& islands
) const {
    int islandCount = static_cast<int>(islands.size());

    if (islandCount <= 1 || config.migrantsPerIsland <= 0) {
        return;
    }

    int migrants = std::min(config.migrantsPerIsland, config.populationSize);

    std::cout << "\n[Island model] Migrating "
        << migrants
        << " best individuals per island using ring topology\n";

    std::vector<std::vector<Individual>> outgoing(islandCount);

    for (int island = 0; island < islandCount; ++island) {
        outgoing[island].assign(
            islands[island].begin(),
            islands[island].begin() + migrants
        );
    }

    for (int island = 0; island < islandCount; ++island) {
        int targetIsland = (island + 1) % islandCount;

        for (int migrant = 0; migrant < migrants; ++migrant) {
            int replaceIndex = config.populationSize - 1 - migrant;
            islands[targetIsland][replaceIndex] = outgoing[island][migrant];
        }
    }

    for (auto& islandPopulation : islands) {
        sortPopulationParallel(islandPopulation);
    }
}

Individual GeneticAlgorithm::findBestIndividual(
    const std::vector<std::vector<Individual>>& islands
) const {
    Individual best = islands.front().front();

    for (const auto& islandPopulation : islands) {
        if (islandPopulation.front().length < best.length) {
            best = islandPopulation.front();
        }
    }

    return best;
}

double GeneticAlgorithm::calculateMeanLengthAcrossIslands(
    const std::vector<std::vector<Individual>>& islands
) const {
    double sum = 0.0;
    int count = 0;

    for (const auto& islandPopulation : islands) {
        for (const Individual& individual : islandPopulation) {
            sum += individual.length;
            ++count;
        }
    }

    return sum / static_cast<double>(count);
}

void GeneticAlgorithm::logIslandState(
    const std::vector<std::vector<Individual>>& islands,
    int generation,
    const std::string& stage
) const {
    std::cout << "\n[Island model] Generation " << generation
        << " - " << stage << '\n';

    std::cout << std::fixed;

    for (size_t island = 0; island < islands.size(); ++island) {
        const auto& population = islands[island];

        double mean = calculateMeanLengthParallel(population);

        std::cout << "  Island " << island
            << " | best: " << population.front().length
            << " | mean: " << mean
            << '\n';
    }

    Individual globalBest = findBestIndividual(islands);

    std::cout << "  Global best: " << globalBest.length << "\n";
}
