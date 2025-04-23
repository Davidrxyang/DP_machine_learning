#include "main.h"

// Function to load a real dataset from a CSV file
std::vector<std::vector<double>> loadDataset(const std::string& filename) {
    std::vector<std::vector<double>> dataset;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return dataset;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        // Parse each value in the row
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value)); // Convert string to double
        }

        dataset.push_back(row);
    }

    file.close();
    return dataset;
}

int main() {
    // Parameters for the breast cancer Wisconsin dataset
    int numFeatures = 30;       // Number of features per sample
    int numClasses = 2;         // Number of output classes
    int hiddenSize = 64;        // Number of neurons in the hidden layer
    double learningRate = 0.05; // Learning rate
    int epochs = 2000;          // Number of training epochs

    // Load the dataset from a CSV file
    std::string filename = "datasets/breast+cancer+wisconsin+diagnostic/wdbc_cleaned.data";
    std::vector<std::vector<double>> dataset = loadDataset(filename);

    if (dataset.empty()) {
        std::cerr << "Error: Dataset is empty or could not be loaded." << std::endl;
        return 1;
    }

    // Create and train the classifier
    BPClassifier classifier(numFeatures, hiddenSize, numClasses, learningRate);
    classifier.train(dataset, epochs);

    // Test the classifier
    double accuracy = classifier.test(dataset);
    std::cout << "Classification accuracy: " << accuracy * 100.0 << "%" << std::endl;

    return 0;
}