#pragma once

#include "City.h"

#include <string>
#include <vector>

class TspInstance {
public:
    explicit TspInstance(const std::string& filePath);

    int cityCount() const;
    const std::vector<City>& getCities() const;
    double distance(int fromIndex, int toIndex) const;

private:
    std::vector<City> cities;
    std::vector<double> distances; // flattened n x n matrix

    void loadCities(const std::string& filePath);
    void buildDistanceMatrix();
};
