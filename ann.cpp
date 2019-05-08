#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class Card {
private:
    float* inputs;
    float* outputs;

public:
    int output;
    Card() {
        inputs = new float [196];
        outputs = new float [10];
    }


    void imageLoad(vector<char> images, int offset) {
        for (int i = 0; i < 196; i++) {
            inputs[i] = ((unsigned char)images[i+offset])/128.0 - 1.0; 
            //cout << (int)((unsigned char)images[i+offset]) << endl;
            //cout << inputs[i] << endl; 
        }
    }

    void labelLoad(vector<char> labels, int offset) {
        output = (int)((unsigned char)labels[offset]);
        for (int i = 0 ; i < 10; i++) {
            if (output == i) {
                outputs[i] = 1.0;
            }
            else {
                outputs[i] = -1.0;
            }
        }
    }
};

vector<char> images;
vector<char> labels;

vector<Card> testing_set;
vector<Card> training_set;


void loadTrainingData(const char* filename) {
    ifstream ifs(filename, ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();

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

int main() {

    loadTrainingData("data/t10k-images-14x14.idx3-ubyte");
    loadTrainingLabel("data/t10k-labels.idx1-ubyte");

    //training_set.resize(8000);
    //int tr_pos = 0;
    //testing_set.resize(2000);
    //int te_pos = 0;

  for (int i = 0; i < 10000; i++) {
    Card c;
    cout << i << endl;
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

  cout << testing_set[1088].output << endl;

}