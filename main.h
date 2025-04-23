#ifndef MAIN_H
#define MAIN_H

#include "BPClassifier.h"
#include <iostream>
#include <vector>
#include <random>

std::vector<std::vector<double>> generateDataset(int numSamples, int numFeatures, int numClasses);

#endif