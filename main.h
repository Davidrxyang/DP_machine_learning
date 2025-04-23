#ifndef MAIN_H
#define MAIN_H

#include "BPClassifier.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

// Function to load a real dataset from a CSV file
std::vector<std::vector<double>> loadDataset(const std::string& filename);

#endif // MAIN_H