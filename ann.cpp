#include <iostream>
#include <fstream>
#include <vector>
#include <getopt.h>
#include <cfloat>
#include "load_data.h"
#include "neuron.h"
#include "network.h"
#include "CycleTimer.h"


#define ROW_SIZE (28)
#define COL_SIZE (28)
#define INPUT_LAYER_SIZE (ROW_SIZE * COL_SIZE)
#define HIDDEN_LAYER_SIZE 300
#define OUTPUT_LAYER_SIZE 10

#define CROSS_VALIDATION (5)
#define LEARNING_RATE (0.01)
#define ACCURACY_THRESHOLD (0.10)

using namespace std;

char* images = NULL;
char* labels = NULL;

vector<Card> testing_set;
vector<Card> training_set;


void loadInputData(const char* filename, unsigned long max_size) {
    std::cout<<"Filename is "<<filename<<", max_size is "<<max_size<<std::endl;
    ifstream ifs(filename, ios::binary|ios::ate);
    if (!ifs.is_open()) {
      std::cout<<"File is not opened"<<std::endl;
    }
    ifstream::pos_type pos = ifs.tellg();

    if (pos < max_size) {
      std::cout<<"Incorrect size for input data. Pos is "<<pos<<std::endl;
      exit(1);
    }

    images = new char[max_size];

    ifs.seekg(0, ios::beg);
    ifs.read(&images[0], max_size);
}

void loadLabelData(const char* filename, unsigned long max_size) {

    std::cout<<"Filename is "<<filename<<", max_size is "<<max_size<<std::endl;
    ifstream ifs(filename, ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();

    if (pos < max_size) {
      std::cout<<"Incorrect size for input data. Pos is "<<pos<<std::endl;
      exit(1);
    }

    labels = new char[max_size];

    ifs.seekg(0, ios::beg);
    ifs.read(&labels[0], max_size);
}


void loadData(vector<Card> &data_set, int data_size, char *dataFile, char *labelFile) {

    unsigned long cardSize = ROW_SIZE * COL_SIZE;

    unsigned long inputSize = 16 + data_size * cardSize;
    unsigned long labelSize = 8 + (unsigned long)data_size;

    loadInputData(dataFile, inputSize);
    loadLabelData(labelFile, labelSize);

    for (int i = 0; i < data_size; i++) {
      Card c(cardSize);
      c.imageLoad(images, 16 + i * cardSize); // There is an offset of 16 bytes
      c.labelLoad(labels, 8 + i);  // There is an offset of 8 bytes
      data_set.push_back(c);
    }

    delete images;
    delete labels;
}

void trainData() {
}

int main(int argc, char **argv) {
    int     opt;
    char *trainingInputFile = NULL;
    char *trainingLabelFile = NULL;

    char *testInputFile = NULL;
    char *testLabelFile = NULL;

    int training_size = 0;
    int test_size = 0;

    int maxEpoch = 0;

    while ((opt = getopt(argc, argv, "d:l:t:o:m:n:e:")) != EOF) {
      switch (opt) {
        case 'd': trainingInputFile = optarg;
          break;
        case 'l': trainingLabelFile = optarg;
          break;
        case 't': testInputFile = optarg;
          break;
        case 'o': testLabelFile = optarg;
          break;
        case 'm': training_size = atoi(optarg);
          break;
        case 'n': test_size = atoi(optarg);
          break;
        case 'e': maxEpoch = atoi(optarg);
          break;
        default: std::cout<<"Incorrect arguments"<<std::endl;
          break;
      }
    }
    if (trainingInputFile == NULL || trainingLabelFile == NULL || testInputFile == NULL || testLabelFile == NULL) {
      std::cout<<"Incorrect arguments"<<std::endl;
    }

    if (training_size == 0 || test_size == 0 || maxEpoch == 0) {
      std::cout<<"Incorrect arguments (Size)"<<std::endl;
    }
    double loadStartTime = currentSeconds();

    setupSigmoid();
    // Load training Data
    loadData(training_set, training_size, trainingInputFile, trainingLabelFile);
    std::cout<<"Completed loading of training data"<<std::endl;

    // Load test Data
    loadData(testing_set, test_size, testInputFile, testLabelFile);
    std::cout<<"Completed loading of test data"<<std::endl;
    double loadEndTime = currentSeconds();
    std::cout << "Load time is " << loadEndTime - loadStartTime << std::endl;


    Network *neuralNetwork = new Network(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
    Network *bestNeuralNetwork = new Network(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);

    int crossFoldValidationTestSize = training_set.size()/CROSS_VALIDATION;

    for(int section = 0 ; section < CROSS_VALIDATION; section++) {
      int testStartIndex = crossFoldValidationTestSize * section;

      double maxAccuracy = 0;

      // Start with the best network from previous phase
      bestNeuralNetwork->copyTo(neuralNetwork);

      for (int epoch = 1; epoch <= maxEpoch; epoch++) {
        double trainingStartTime = currentSeconds();
        // Training with all training data
        for (int i = 0; i < training_set.size(); i++) {
          // Skip the testing set for cross fold validation
          if (i == testStartIndex) {
            i += crossFoldValidationTestSize - 1;
            continue;
          }
          neuralNetwork->respond(training_set[i]);
          neuralNetwork->train(LEARNING_RATE, training_set[i].outputs);
        }
        double trainingEndTime = currentSeconds();

        int totalRight = 0;
        // Testing for Accuracy
        for (int i = testStartIndex; i < testStartIndex + crossFoldValidationTestSize; i++) {
          neuralNetwork->respond(training_set[i]);
          int out = neuralNetwork->getOutput();
          if (out == training_set[i].output) {
            totalRight++;
          }
        }
        double testEndTime = currentSeconds();

        double accuracy = (double) (totalRight) / crossFoldValidationTestSize;
        if (accuracy > maxAccuracy) {
          maxAccuracy = accuracy;
          neuralNetwork->copyTo(bestNeuralNetwork);
        }
        if (accuracy < (1 - ACCURACY_THRESHOLD) * maxAccuracy) {
          std::cout << "Breaking because of threshold:maxAccuracy=" << maxAccuracy << ",accuracy=" << accuracy << std::endl;
          break;
        }

        std::cout<<"Cross::"<<section<<",EPOCH:" << epoch << ", Accuracy is " << accuracy<< std::endl;
        std::cout << "Training time is " << trainingEndTime - trainingStartTime << std::endl;
        std::cout << "Test time is " << testEndTime - trainingEndTime << std::endl;
      }
      std::cout<<"Cross::"<<section<<",Best Accuracy is " <<maxAccuracy<< std::endl;
    }

    int correctPrediction = 0;
    // Testing for Accuracy on test data
    for (int i = 0; i < testing_set.size(); i++) {
      bestNeuralNetwork->respond(testing_set[i]);
      int out = bestNeuralNetwork->getOutput();
      if (out == testing_set[i].output) {
        correctPrediction++;
      }
    }

    std::cout << "Accuracy::"<< (float) (correctPrediction)/(testing_set.size())<<std::endl;
    return 0;
}
