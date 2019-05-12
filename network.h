#include <vector>

class Network {

    vector<Neuron> input_layer;
    vector<Neuron> hidden_layer;
    vector<Neuron> output_layer;

 public:
  int bestIndex = -1;
  Network(int inputs, int hidden, int outputs) {

        for (int i = 0; i < inputs; i++) {
            Neuron neuron;
            input_layer.push_back(neuron);
        }

        for (int j = 0; j < hidden; j++) {
            Neuron neuron (&input_layer);
            hidden_layer.push_back(neuron);
        }

        for (int k = 0; k < outputs; k++) {
            Neuron neuron (&hidden_layer);
            output_layer.push_back(neuron);
        }
    }

    void respond(Card card) {

        for (int i = 0; i < input_layer.size(); i++) {
            input_layer[i].output = card.inputs[i];
        }

        // now feed forward through the hidden layer
        for (int j = 0; j < hidden_layer.size(); j++) {
            hidden_layer[j].respond();
        }

        for (int k = 0; k < output_layer.size(); k++) {
            output_layer[k].respond();
        }
    }


    void train(float learning_rate, float* outputs) {
        // adjust the output layer
        for (int k = 0; k < output_layer.size(); k++) {
            output_layer[k].setError(outputs[k]);
            output_layer[k].train(learning_rate);
        }
        float best = -1.0f;
        for (int i = 0; i < output_layer.size(); i++) {
            if (output_layer[i].output > best) {
              bestIndex = i;
            }
        }
        // propagate back to the hidden layer
        for (int j = 0; j < hidden_layer.size(); j++) {
          hidden_layer[j].train(learning_rate);
        }

    // The input layer doesn't learn: it is the input and only that
    }

    int getOutput() {
      float best = -1.0f;
      for (int i = 0; i < output_layer.size(); i++) {
        if (output_layer[i].output > best) {
          best = output_layer[i].output;
          bestIndex = i;
        }
      }
      return bestIndex;
    }
};
