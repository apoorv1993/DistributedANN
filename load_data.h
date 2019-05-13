#ifndef _LOADDATA_H
#define _LOADDATA_H

#include <vector>
using namespace std;

class Card {
public:
    float* inputs;
    float* outputs;

    unsigned long size;

    int output;
    Card(unsigned long size) {
        this->size = size;
        inputs = new float [size];
        outputs = new float [10];
    }


    void imageLoad(char* images, unsigned long offset) {
        for (int i = 0; i < size; i++) {
            inputs[i] = ((unsigned char)images[i+offset])/128.0f - 1.0f;
        }
    }

    void labelLoad(char* labels, unsigned long offset) {
        output = (int)((unsigned char)labels[offset]);
        for (int i = 0 ; i < 10; i++) {
            if (output == i) {
                outputs[i] = 1.0;
            }
            else {
                outputs[i] = -1.0f;
            }
        }
    }
};

#endif //_LOADDATA_H
