#include <iostream>
#include <fstream>
#include <vector>
#include <getopt.h>
#include <cfloat>
#include "load_data.h"
#include "neuron.h"
#include "network.h"
#include "CycleTimer.h"

#define INPUT_LAYER_SIZE 196
#define HIDDEN_LAYER_SIZE 300
#define OUTPUT_LAYER_SIZE 10

#define LEARNING_RATE (0.01)
#define RMSE_THRESHOLD (0.10)

using namespace std;

vector<char> images;
vector<char> labels;

vector<Card> testing_set;
vector<Card> training_set;


void loadInputData(const char* filename) {
    std::cout<<"Filename is "<<filename<<std::endl;
    ifstream ifs(filename, ios::binary|ios::ate);
    if (!ifs.is_open()) {
      std::cout<<"File is not opened"<<std::endl;
    }
    ifstream::pos_type pos = ifs.tellg();
    std::cout<<"Position is "<<pos<<std::endl;

    images.resize(pos);

    ifs.seekg(0, ios::beg);
    ifs.read(&images[0], pos);
}

void loadLabelData(const char* filename) {
    ifstream ifs(filename, ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();

    labels.resize(pos);

    ifs.seekg(0, ios::beg);
    ifs.read(&labels[0], pos);
}


void loadData(vector<Card> &data_set, int data_size, char *dataFile, char *labelFile) {

    loadInputData(dataFile);
    loadLabelData(labelFile);

    for (int i = 0; i < data_size; i++) {
      Card c;
      c.imageLoad(images, 16 + i * 196); // There is an offset of 16 bytes
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
    double bestRate= 0.01;
    double bestRight = 0;

    double minRMSE = DBL_MAX;

    double loadEndTime = currentSeconds();

    for(int e=1; e < epoch; e++) {
      double traingStartTime = currentSeconds();
      // Training with all training data
      for (int i = 0; i < training_set.size(); i++) {
        neuralnet->respond(training_set[i]);
        neuralnet->train(LEARNING_RATE, training_set[i].outputs);
      }
      double traingEndTime = currentSeconds();

      int totalRight = 0;

      double totalDiff = 0;

      // Testing
      for (int i = 0; i < testing_set.size(); i++) {
        //    for (int i = 0; i < 10; i++) {
        neuralnet->respond(testing_set[i]);
        int out = neuralnet->getOutput();
        if (out == testing_set[i].output) {
          totalRight++;
        } else {
          totalDiff += pow(out - testing_set[i].output, 2);
        }
      }

      double rmse = sqrt(totalDiff/test_size);

      double testEndTime = currentSeconds();

      std::cout << "EPOCH:"<<e<<", Accuracy is " << (float) (totalRight) / (testing_set.size()) <<", RMSE::"<<rmse<<std::endl;

      if (rmse < minRMSE) {
        minRMSE = rmse;
      }

      if (minRMSE < RMSE_THRESHOLD * rmse) {
        std::cout<<"Breaking because of threshold"<<std::endl;
        break;
      }

      std::cout << "Training time is " << traingEndTime - traingStartTime << std::endl;
      std::cout << "Test time is " << testEndTime - traingEndTime << std::endl;
    }
  std::cout << "Load time is " << loadEndTime - loadStartTime << std::endl;
  std::cout << "Best Learning Rate:"<<bestRate<<", Accuracy " << (float) (bestRight)/(testing_set.size())<<std::endl;
    return 0;
}
