#ifndef NN_H
#define NN_H

#include <vector>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <chrono>

#include "DataLoader.h"

using namespace std;

// Constants
// extern int INPUT_DIM;
const int HIDDEN_DIM = 16;
const float LEARNING_RATE = 0.01;
const int EPOCHS = 100;
const float TEST_RATIO = 0.2;

// Activation functions
float sigmoid(float x);
float sigmoid_derivative(float x);
float relu(float x);
float relu_derivative(float x);

// Data normalization
void normalize(std::vector<std::vector<float>>& train, std::vector<std::vector<float>>& test);

// Main function
int main();

#endif // NN_H