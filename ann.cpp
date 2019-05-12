#include <iostream>
#include <fstream>
#include <vector>
#include <getopt.h>
#include "load_data.h"
#include "neuron.h"
#include "network.h"

using namespace std;

vector<char> images;
vector<char> labels;

vector<Card> testing_set;
vector<Card> training_set;


void loadTrainingData(const char* filename) {
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

void loadTrainingLabel(const char* filename) {
    ifstream ifs(filename, ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();

    labels.resize(pos);

    ifs.seekg(0, ios::beg);
    ifs.read(&labels[0], pos);
}


void loadData(char *trainingFile, char *labelFile) {

    loadTrainingData(trainingFile);
    loadTrainingLabel(labelFile);

    //training_set.resize(8000);
    //int tr_pos = 0;
    //testing_set.resize(2000);
    //int te_pos = 0;

  for (int i = 0; i < 10000; i++) {
    Card c;
    if (i % 5 != 0) {
      //training_set[tr_pos] = c;
      //training_set[tr_pos].imageLoad(images, 16 + i * 196); // There is an offset of 16 bytes
      //training_set[tr_pos].labelLoad(labels, 8 + i);  // There is an offset of 8 bytes
      //tr_pos++;
      c.imageLoad(images, 16 + i * 196); // There is an offset of 16 bytes
      c.labelLoad(labels, 8 + i);  // There is an offset of 8 bytes
      training_set.push_back(c);
    } else {
      //testing_set[te_pos] = c;
      //testing_set[te_pos].imageLoad(images, 16 + i * 196);  // There is an offset of 16 bytes
      //testing_set[te_pos].labelLoad(labels, 8 + i);  // There is an offset of 8 bytes
      //te_pos++;
      c.imageLoad(images, 16 + i * 196); // There is an offset of 16 bytes
      c.labelLoad(labels, 8 + i);  // There is an offset of 8 bytes
      testing_set.push_back(c);
    }
  }

  //cout << testing_set[1088].output << endl;

}

void trainData() {
}

int main(int argc, char **argv) {
    int     opt;
    char *inputFile = NULL;
    char *labelFile = NULL;

    while ((opt = getopt(argc, argv, "l:i:")) != EOF) {
      switch (opt) {
        case 'i': inputFile = optarg;
          break;
        case 'l': labelFile = optarg;
          break;
        default: std::cout<<"Incorrect arguments"<<std::endl;
          break;
      }
    }
    if (inputFile == NULL || labelFile == NULL) {
      std::cout<<"Incorrect arguments"<<std::endl;
    }

    loadData(inputFile, labelFile);
    setupSigmoid();
    Network *neuralnet = new Network(196, 49, 10);


    // Training with all training data
    for (int i = 0; i < training_set.size(); i++) {
      neuralnet->respond(training_set[i]);
      neuralnet->train(training_set[i].outputs);
    }

    int totalRight = 0;
    // Testing
    for (int i = 0; i < testing_set.size(); i++) {
      neuralnet->respond(testing_set[i]);
      if(neuralnet->bestIndex == testing_set[i].output) totalRight ++;
      std::cout<<"TestCard  "<<i<<"::"<<"Got ::"<<neuralnet->bestIndex<<", Expected::"<<testing_set[i].output<<std::endl;
    }

    std::cout<<"Accuracy is "<<(float)(totalRight)/(testing_set.size())<<std::endl;
    return 0;
}
