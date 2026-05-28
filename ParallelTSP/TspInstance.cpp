#include "TspInstance.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

TspInstance::TspInstance(const std::string& filePath) {
    loadCities(filePath);
    buildDistanceMatrix();
}

int TspInstance::cityCount() const {
    return static_cast<int>(cities.size());
}

const std::vector<City>& TspInstance::getCities() const {
    return cities;
}

double TspInstance::distance(int fromIndex, int toIndex) const {
    int n = cityCount();
    return distances[fromIndex * n + toIndex];
}

void TspInstance::loadCities(const std::string& filePath) {
    std::ifstream input(filePath);
    if (!input) {
        throw std::runtime_error("Cannot open data file: " + filePath);
    }

    cities.clear();

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        std::istringstream iss(line);
        City city;
        if (iss >> city.id >> city.x >> city.y) {
            cities.push_back(city);
        }
    }

    if (cities.empty()) {
        throw std::runtime_error("No cities loaded from: " + filePath);
    }
}

void TspInstance::buildDistanceMatrix() {
    int n = cityCount();
    distances.assign(n * n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double dx = cities[i].x - cities[j].x;
            double dy = cities[i].y - cities[j].y;
            distances[i * n + j] = std::sqrt(dx * dx + dy * dy);
        }
    }
}
