
#ifndef _SIGMOID_H
#define _SIGMOID_H

#include <vector>
#include <cmath>
vector<float> g_sigmoid;


void setupSigmoid() {
  
  for (int i = 0; i < 200; i++) {
    float x = (i / 20.0f) - 5.0f;
    g_sigmoid.push_back(2.0f / (1.0f + exp(-2.0f * x)) - 1.0f);
  }
}

// once the sigmoid has been set up, this function accesses it:
float lookupSigmoid(float x) {
    int value =  (int) floor((x + 5.0f) * 20.0f);
    if (value > 199)
        value = 199;
    else if (value < 0)
        value = 0;

    return g_sigmoid[value];
}

#endif //_SIGMOID_H