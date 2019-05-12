#include <vector>
#include <math.h>
//float [] g_sigmoid = new float [200];
vector<float> g_sigmoid;


void setupSigmoid() {
  
  for (int i = 0; i < 200; i++) {
    float x = (i / 20.0) - 5.0;
    g_sigmoid.push_back(2.0 / (1.0 + exp(-2.0 * x)) - 1.0);
  }
}

// once the sigmoid has been set up, this function accesses it:
float lookupSigmoid(float x) {
    int value =  (int) floor((x + 5.0) * 20.0);
    if (value > 199)
        value = 199;
    else if (value < 0)
        value = 0;

    return g_sigmoid[value];
}
