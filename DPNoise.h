// dp_noise.h
#ifndef DPNOISE_H
#define DPNOISE_H

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

const float SIGMA = 20.0; // Standard deviation for Gaussian noise

std::vector<float> add_gaussian_noise_to_vector(const std::vector<float>& gradients);

float add_gaussian_noise(float data);

#endif // DP_NOISE_H
