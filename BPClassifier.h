#ifndef BPCLASSIFIER_H
#define BPCLASSIFIER_H

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>

class BPClassifier {
public:
    BPClassifier(int inputSize, int hiddenSize, int outputSize, double learningRate);
    void train(const std::vector<std::vector<double>>& data, int epochs);
    double test(const std::vector<std::vector<double>>& data);
    int predict(const std::vector<double>& input);

private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    double learningRate;

    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;
    std::vector<double> hiddenLayer;
    std::vector<double> outputLayer;

    double sigmoid(double x);
    double sigmoidDerivative(double x);
    void forward(const std::vector<double>& input);
    void backward(const std::vector<double>& input, const std::vector<double>& target);
    void initializeWeights();
};

#endif // BPCLASSIFIER_H