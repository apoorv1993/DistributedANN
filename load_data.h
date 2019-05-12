#include <vector>
using namespace std;

class Card {
public:
    float* inputs;
    float* outputs;

    int output;
    Card() {
        inputs = new float [196];
        outputs = new float [10];
    }


    void imageLoad(vector<char> images, int offset) {
        for (int i = 0; i < 196; i++) {
            inputs[i] = ((unsigned char)images[i+offset])/128.0f - 1.0f;
        }
    }

    void labelLoad(vector<char> labels, int offset) {
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
