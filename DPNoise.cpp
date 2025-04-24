#include "DPNoise.h"

// Function to add Gaussian noise to a gradient vector
std::vector<float> add_gaussian_noise_to_vector(const std::vector<float>& gradients) {
    std::vector<float> noisy_gradients;
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<float> distribution(0.0, SIGMA);

    for (float grad : gradients) {
        float noise = distribution(generator);
        noisy_gradients.push_back(grad + noise);
    }

    return noisy_gradients;
}

float add_gaussian_noise(float data) {
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<float> distribution(0.0, SIGMA);
    float noise = distribution(generator);
    return data + noise;
}

