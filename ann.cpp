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
#define RMSE_THRESHOLD (0.10)

using namespace std;

vector<char> images;
vector<char> labels;

vector<Card> testing_set;
vector<Card> training_set;


void loadInputData(const char* filename, unsigned long max_size) {
    images.clear();

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

    images.resize(max_size);

    ifs.seekg(0, ios::beg);
    ifs.read(&images[0], max_size);
}

void loadLabelData(const char* filename, unsigned long max_size) {
    labels.clear();

    std::cout<<"Filename is "<<filename<<", max_size is "<<max_size<<std::endl;
    ifstream ifs(filename, ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();

    if (pos < max_size) {
      std::cout<<"Incorrect size for input data. Pos is "<<pos<<std::endl;
      exit(1);
    }

    labels.resize(max_size);

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

    int epoch = 0;

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
        case 'e': epoch = atoi(optarg);
          break;
        default: std::cout<<"Incorrect arguments"<<std::endl;
          break;
      }
    }
    if (trainingInputFile == NULL || trainingLabelFile == NULL || testInputFile == NULL || testLabelFile == NULL) {
      std::cout<<"Incorrect arguments"<<std::endl;
    }

    if (training_size == 0 || test_size == 0 || epoch == 0) {
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

    Network *neuralnet = new Network(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
    double loadEndTime = currentSeconds();
    int crossFoldValidationTestSize = training_set.size()/CROSS_VALIDATION;

    for(int section = 0 ; section < CROSS_VALIDATION; section++) {
      int testStartIndex = crossFoldValidationTestSize * section;

      double bestRate= 0.01;
      double bestRight = 0;
      double minRMSE = DBL_MAX;

      for (int e = 1; e < epoch; e++) {
        double trainingStartTime = currentSeconds();
        // Training with all training data
        for (int i = 0; i < training_set.size(); i++) {
          // Skip the testing set for cross fold validation
          if (i == testStartIndex) {
            i += crossFoldValidationTestSize - 1;
            continue;
          }
          neuralnet->respond(training_set[i]);
          neuralnet->train(LEARNING_RATE, training_set[i].outputs);
        }
        double trainingEndTime = currentSeconds();

        int totalRight = 0;
        double totalDiff = 0;

        // Testing for RMSE
        for (int i = testStartIndex; i < testStartIndex+crossFoldValidationTestSize; i++) {
          neuralnet->respond(training_set[i]);
          int out = neuralnet->getOutput();
          if (out == training_set[i].output) {
            totalRight++;
          } else {
            totalDiff += pow(out - training_set[i].output, 2);
          }
        }

        double rmse = sqrt(totalDiff / crossFoldValidationTestSize);
        if (rmse < minRMSE) {
          minRMSE = rmse;
        }

        double testEndTime = currentSeconds();
        std::cout<<"Cross::"<<section<<",EPOCH:" << e << ", Accuracy is " << (float) (totalRight) / crossFoldValidationTestSize<< ", RMSE::"<< rmse << std::endl;

        if (rmse > (1 + RMSE_THRESHOLD) * minRMSE) {
          std::cout << "Breaking because of threshold:MinRMSE=" << minRMSE << ", RMSE=" << rmse << std::endl;
          break;
        }

        std::cout << "Training time is " << trainingEndTime - trainingStartTime << std::endl;
        std::cout << "Test time is " << testEndTime - trainingEndTime << std::endl;
      }
      std::cout << "Best Learning Rate:"<<bestRate<<", Accuracy " << (float) (bestRight)/crossFoldValidationTestSize<<std::endl;
    }

  int correctPrediction = 0;

  // Testing for RMSE
  for (int i = 0; i < testing_set.size(); i++) {
    neuralnet->respond(testing_set[i]);
    int out = neuralnet->getOutput();
    if (out == testing_set[i].output) {
      correctPrediction++;
    }
  }

  std::cout << "Load time is " << loadEndTime - loadStartTime << std::endl;
  std::cout << "Accuracy " << (float) (correctPrediction)/(testing_set.size())<<std::endl;
  return 0;
}
