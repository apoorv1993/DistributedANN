#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include "sigmoid.h"
using namespace std;

float LEARNING_RATE = 0.01;

class Neuron {
public:
    vector<Neuron> inputs; 
    vector<float> weights;
    float output;
    float error;

    Neuron () {
        error = 0.0;
    }

    Neuron(vector<Neuron> p_inputs) {

        error = 0.0;

        for (int i = 0; i < p_inputs.size(); i++) {
            inputs.push_back(p_inputs[i]);
            // Note:
            // Need to change to rand() between -1.0 and 1.0
            weights.push_back(((float) rand()) / (float) RAND_MAX);
        }
    }

    void respond() {

        float input = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            input += inputs[i].output * weights[i];
        }
    
        output = lookupSigmoid(input);
        error = 0.0;
    }

    void setError(float desired) {
        error = desired - output;
    }

    void train() {
        // Back propagation
        float delta = (1.0 - output) * (1.0 + output) * error * LEARNING_RATE;
        for (int i = 0; i < inputs.size(); i++) {
            inputs[i].error += weights[i] * error;
            weights[i] += inputs[i].output * delta;
        }
    }

};
