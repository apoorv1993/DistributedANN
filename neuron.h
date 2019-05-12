#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include "sigmoid.h"
using namespace std;

float LEARNING_RATE = 0.10;

class Neuron {
public:
    vector<Neuron>* inputs;
    vector<float> weights;
    float output;
    float error;

    Neuron () {
        error = 0.0;
    }

    Neuron(vector<Neuron> *p_inputs) {

        error = 0.0;
        inputs = p_inputs;
        for (int i = 0; i < p_inputs->size(); i++) {
            // Note:
            // Need to change to rand() between -1.0 and 1.0
            weights.push_back(RandomFloat(-1.0f, 1.0f));
        }
    }

    void respond() {

        float input = 0.0;
        for (int i = 0; i < inputs->size(); i++) {
            input += (*inputs)[i].output * weights[i];
        }
    
        output = lookupSigmoid(input);
        error = 0.0;
    }

    void setError(float desired) {
        error = desired - output;
    }

    void train(float learning_rate) {
        // Back propagation
        float delta = (1.0f - output) * (1.0f + output) * error * learning_rate;
        for (int i = 0; i < inputs->size(); i++) {
            (*inputs)[i].error += weights[i] * error;
            weights[i] += (*inputs)[i].output * delta;
        }
    }

    float RandomFloat(float a, float b) {
      float random = ((float) rand()) / (float) RAND_MAX;
      float diff = b - a;
      float r = random * diff;
      return a + r;
    }

};
