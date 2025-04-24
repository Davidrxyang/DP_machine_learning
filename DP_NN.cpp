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

int INPUT_DIM = 0;

const int HIDDEN_DIM = 16;
const float LEARNING_RATE = 0.01;
const int EPOCHS = 500;
const float TEST_RATIO = 0.2;

// Sigmoid and ReLU
float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
float sigmoid_derivative(float x) { return x * (1 - x); }
float relu(float x) { return max(0.0f, x); }
float relu_derivative(float x) { return x > 0 ? 1.0f : 0.0f; }

// Normalize in-place using train-set statistics
void normalize(vector<vector<float>>& train, vector<vector<float>>& test) {

    int m = train[0].size();
    vector<float> mean(m, 0), stddev(m, 0);
    int n = train.size();

    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) mean[j] += train[i][j];
        mean[j] /= n;
        for (int i = 0; i < n; ++i) stddev[j] += pow(train[i][j] - mean[j], 2);
        stddev[j] = sqrt(stddev[j] / n);
    }

    for (auto& row : train)
        for (int j = 0; j < m; ++j)
            row[j] = (row[j] - mean[j]) / stddev[j];

    for (auto& row : test)
        for (int j = 0; j < m; ++j)
            row[j] = (row[j] - mean[j]) / stddev[j];
}

int main() {

    vector<vector<float>> X;
    vector<float> y;
    load_csv_wdbc("datasets/breast+cancer+wisconsin+diagnostic/wdbc.data", X, y);
    // load_csv_votes("datasets/congressional+voting+records/house-votes-84.data", X, y);

    // load_csv_iris("datasets/iris/iris.data", X, y);

    // get the input dimension
    INPUT_DIM = X[0].size(); // <-- set input dimension dynamically


    // Shuffle
    vector<size_t> indices(X.size());
    iota(indices.begin(), indices.end(), 0);
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(indices.begin(), indices.end(), default_random_engine(seed));

    vector<vector<float>> X_train, X_test;
    vector<float> y_train, y_test;
    size_t split = X.size() * (1 - TEST_RATIO);

    for (size_t i = 0; i < X.size(); ++i) {
        if (i < split) {
            X_train.push_back(X[indices[i]]);
            y_train.push_back(y[indices[i]]);
        } else {
            X_test.push_back(X[indices[i]]);
            y_test.push_back(y[indices[i]]);
        }
    }

    normalize(X_train, X_test);

    // Init weights
    mt19937 gen(seed);
    uniform_real_distribution<> dist(-1, 1);
    vector<vector<float>> W1(HIDDEN_DIM, vector<float>(INPUT_DIM));
    vector<float> B1(HIDDEN_DIM, 0.0f);
    vector<float> W2(HIDDEN_DIM);
    float B2 = 0.0f;

    for (auto& row : W1)
        for (float& w : row) w = dist(gen);
    for (float& w : W2) w = dist(gen);

    // Training
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < X_train.size(); ++i) {
            // Forward
            vector<float> hidden(HIDDEN_DIM);
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                hidden[j] = B1[j];
                for (int k = 0; k < INPUT_DIM; ++k)
                    hidden[j] += W1[j][k] * X_train[i][k];
                hidden[j] = relu(hidden[j]);
            }

            float out = B2;
            for (int j = 0; j < HIDDEN_DIM; ++j)
                out += W2[j] * hidden[j];
            out = sigmoid(out);

            // Loss
            float error = y_train[i] - out;
            total_loss += error * error;

            // Backprop
            float d_out = error * sigmoid_derivative(out);
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                float d_hidden = d_out * W2[j] * relu_derivative(hidden[j]);
                for (int k = 0; k < INPUT_DIM; ++k)
                    W1[j][k] += LEARNING_RATE * d_hidden * X_train[i][k];
                B1[j] += LEARNING_RATE * d_hidden;
                W2[j] += LEARNING_RATE * d_out * hidden[j];
            }
            B2 += LEARNING_RATE * d_out;
        }
        cout << "Epoch " << epoch + 1 << " Loss: " << total_loss / X_train.size() << endl;
    }

    // Evaluation
    int correct = 0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        vector<float> hidden(HIDDEN_DIM);
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            hidden[j] = B1[j];
            for (int k = 0; k < INPUT_DIM; ++k)
                hidden[j] += W1[j][k] * X_test[i][k];
            hidden[j] = relu(hidden[j]);
        }

        float out = B2;
        for (int j = 0; j < HIDDEN_DIM; ++j)
            out += W2[j] * hidden[j];
        out = sigmoid(out);
        int pred = out >= 0.5 ? 1 : 0;
        if (pred == y_test[i]) correct++;
    }

    cout << "\nTest Accuracy: " << (float)correct / X_test.size() * 100.0f << "%" << endl;
    return 0;
}
