#ifndef DP_NN_H
#define DP_NN_H

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
#include "DPNoise.h"

using namespace std;

// Constants
const int HIDDEN_DIM = 16;
const float LEARNING_RATE = 0.01;
const int EPOCHS = 500;
const float TEST_RATIO = 0.2;
const float CLIP_THRESHOLD = 1.0; // Gradient clipping threshold

// Activation functions
float sigmoid(float x);
float sigmoid_derivative(float x);
float relu(float x);
float relu_derivative(float x);

// Gradient clipping
float clip_by_l2norm(float value, float norm, float clip_threshold);
float compute_l2_norm(const vector<float>& vec);

// Data normalization
void normalize(std::vector<std::vector<float>>& train, std::vector<std::vector<float>>& test);

// Main function
int main();

#endif // DP_NN_H