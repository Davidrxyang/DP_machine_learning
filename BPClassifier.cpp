#include "BPClassifier.h"

BPClassifier::BPClassifier(int inputSize, int hiddenSize, int outputSize, double learningRate)
    : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize), learningRate(learningRate) {
    initializeWeights();
}

void BPClassifier::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
    weightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));

    for (auto& row : weightsInputHidden)
        for (auto& weight : row)
            weight = dis(gen);

    for (auto& row : weightsHiddenOutput)
        for (auto& weight : row)
            weight = dis(gen);
}

double BPClassifier::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double BPClassifier::sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

void BPClassifier::forward(const std::vector<double>& input) {
    hiddenLayer.resize(hiddenSize);
    outputLayer.resize(outputSize);

    for (int i = 0; i < hiddenSize; ++i) {
        hiddenLayer[i] = 0.0;
        for (int j = 0; j < inputSize; ++j) {
            hiddenLayer[i] += input[j] * weightsInputHidden[j][i];
        }
        hiddenLayer[i] = sigmoid(hiddenLayer[i]);
    }

    for (int i = 0; i < outputSize; ++i) {
        outputLayer[i] = 0.0;
        for (int j = 0; j < hiddenSize; ++j) {
            outputLayer[i] += hiddenLayer[j] * weightsHiddenOutput[j][i];
        }
        outputLayer[i] = sigmoid(outputLayer[i]);
    }
}

void BPClassifier::backward(const std::vector<double>& input, const std::vector<double>& target) {
    std::vector<double> outputError(outputSize);
    std::vector<double> hiddenError(hiddenSize);

    for (int i = 0; i < outputSize; ++i) {
        outputError[i] = (target[i] - outputLayer[i]) * sigmoidDerivative(outputLayer[i]);
    }

    for (int i = 0; i < hiddenSize; ++i) {
        hiddenError[i] = 0.0;
        for (int j = 0; j < outputSize; ++j) {
            hiddenError[i] += outputError[j] * weightsHiddenOutput[i][j];
        }
        hiddenError[i] *= sigmoidDerivative(hiddenLayer[i]);
    }

    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            weightsHiddenOutput[i][j] += learningRate * outputError[j] * hiddenLayer[i];
        }
    }

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            weightsInputHidden[i][j] += learningRate * hiddenError[j] * input[i];
        }
    }
}

void BPClassifier::train(const std::vector<std::vector<double>>& data, int epochs) {
    int trainSize = static_cast<int>(data.size() * 0.9);
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < trainSize; ++i) {
            const auto& row = data[indices[i]];
            std::vector<double> input(row.begin() + 1, row.end());
            std::vector<double> target(outputSize, 0.0);
            target[static_cast<int>(row[0])] = 1.0;

            forward(input);
            backward(input, target);
        }
    }
}

double BPClassifier::test(const std::vector<std::vector<double>>& data) {
    int testSize = static_cast<int>(data.size() * 0.1);
    int correct = 0;

    for (int i = data.size() - testSize; i < data.size(); ++i) {
        const auto& row = data[i];
        std::vector<double> input(row.begin() + 1, row.end());
        int actual = static_cast<int>(row[0]);
        int predicted = predict(input);

        if (actual == predicted) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / testSize;
}

int BPClassifier::predict(const std::vector<double>& input) {
    forward(input);
    return std::distance(outputLayer.begin(), std::max_element(outputLayer.begin(), outputLayer.end()));
}