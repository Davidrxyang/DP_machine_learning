#include "main.h"

// Function to generate a synthetic dataset
std::vector<std::vector<double>> generateDataset(int numSamples, int numFeatures, int numClasses) {
    std::vector<std::vector<double>> dataset;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> classDist(0, numClasses - 1);
    std::uniform_real_distribution<> featureDist(0.0, 1.0);

    for (int i = 0; i < numSamples; ++i) {
        int label = classDist(gen);
        std::vector<double> sample(numFeatures + 1);
        sample[0] = label; // First column is the class label
        for (int j = 1; j <= numFeatures; ++j) {
            sample[j] = featureDist(gen); // Remaining columns are features
        }
        dataset.push_back(sample);
    }

    return dataset;
}

int main() {
    // Parameters

    // for breast cancer wisoncsin dataset 

    int numSamples = 569;      // Number of samples in the dataset
    int numFeatures = 30;       // Number of features per sample
    int numClasses = 2;         // Number of output classes
    int hiddenSize = 64;         // Number of neurons in the hidden layer
    double learningRate = 0.05; // Learning rate
    int epochs = 2000;           // Number of training epochs

    // Generate synthetic dataset
    std::vector<std::vector<double>> dataset = generateDataset(numSamples, numFeatures, numClasses);

    // Create and train the classifier
    BPClassifier classifier(numFeatures, hiddenSize, numClasses, learningRate);
    classifier.train(dataset, epochs);

    // Test the classifier
    double accuracy = classifier.test(dataset);
    std::cout << "Classification accuracy: " << accuracy * 100.0 << "%" << std::endl;

    return 0;
}