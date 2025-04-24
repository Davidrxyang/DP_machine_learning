#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

void load_csv_wdbc(const std::string& filename, std::vector<std::vector<float>>& features, std::vector<float>& labels) {
    std::ifstream file(filename);
    std::string line;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        std::getline(ss, value, ','); // skip ID
        std::getline(ss, value, ','); // diagnosis
        float label = (value == "M") ? 1.0f : 0.0f;
        labels.push_back(label);

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }
        features.push_back(row);
    }
}

void load_csv_votes(const std::string& filename, std::vector<std::vector<float>>& features, std::vector<float>& labels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<float> row;

        // Read party
        std::getline(ss, token, ',');
        float label = (token == "republican") ? 1.0f : 0.0f;
        labels.push_back(label);

        // Read 16 votes
        while (std::getline(ss, token, ',')) {
            if (token == "y") row.push_back(1.0f);
            else if (token == "n") row.push_back(0.0f);
            else row.push_back(-1.0f);  // Treat missing votes as -1.0
        }

        if (row.size() != 16) {
            std::cerr << "Invalid feature count: " << row.size() << " (expected 16)" << std::endl;
            exit(1);
        }

        features.push_back(row);
    }

    file.close();
}

void load_csv_iris(const std::string& filename, std::vector<std::vector<float>>& features, std::vector<float>& labels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<float> row;

        // Read 4 float features
        for (int i = 0; i < 4; ++i) {
            if (!std::getline(ss, token, ',')) {
                std::cerr << "Error reading feature on line: " << line << std::endl;
                exit(1);
            }
            row.push_back(std::stof(token));
        }

        // Read label
        if (!std::getline(ss, token, ',')) {
            std::cerr << "Missing label on line: " << line << std::endl;
            exit(1);
        }

        float label;
        if (token == "Iris-setosa") label = 0.0f;
        else if (token == "Iris-versicolor") label = 1.0f;
        else if (token == "Iris-virginica") label = 2.0f;
        else {
            std::cerr << "Unknown class label: " << token << std::endl;
            exit(1);
        }

        features.push_back(row);
        labels.push_back(label);
    }

    file.close();
}