#ifndef _NETWORK_H
#define _NETWORK_H

#include <vector>
#include "neuron.h"
#include "mpi.h"
#include "CycleTimer.h"

class Network {

    vector<Neuron*> input_layer;
    vector<Neuron*> hidden_layer;
    vector<Neuron*> output_layer;

    float *weightSendBuffer;
    float *weightReceiveBuffer;

    size_t bufferSize;

 public:
    int bestIndex;
    double communicationTime;
    Network(int inputs, int hidden, int outputs, int rank) {
          for (int i = 0; i < inputs; i++) {
              Neuron *neuron = new Neuron();
              input_layer.push_back(neuron);
          }

          for (int j = 0; j < hidden; j++) {
              Neuron *neuron = new Neuron(&input_layer, rank == 0);
              hidden_layer.push_back(neuron);
          }

          for (int k = 0; k < outputs; k++) {
              Neuron *neuron = new Neuron(&hidden_layer, rank == 0);
              output_layer.push_back(neuron);
          }

          bufferSize = hidden * inputs + outputs * hidden;
          weightSendBuffer = new float[bufferSize];
          weightReceiveBuffer = new float[bufferSize];
          communicationTime = 0;
    }

    ~Network() {
      for (int i = 0; i < input_layer.size(); i++) {
        delete input_layer[i];
      }

      for (int j = 0; j < hidden_layer.size(); j++) {
        delete hidden_layer[j];
      }

      for (int k = 0; k < output_layer.size(); k++) {
        delete output_layer[k];
      }
    }

    void copyTo(Network *otherNetwork) {
      // Copy weights for hidden layer
      for (int j = 0; j < hidden_layer.size(); j++) {
        otherNetwork->hidden_layer[j]->setWeights(hidden_layer[j]->getWeights());
      }
      // Copy weights for output layer
      for (int k = 0; k < output_layer.size(); k++) {
        otherNetwork->output_layer[k]->setWeights(output_layer[k]->getWeights());
      }
    }

  void respond(Card card) {

        for (int i = 0; i < input_layer.size(); i++) {
            input_layer[i]->output = card.inputs[i];
        }

        // now feed forward through the hidden layer
        for (int j = 0; j < hidden_layer.size(); j++) {
            hidden_layer[j]->respond();
        }

        for (int k = 0; k < output_layer.size(); k++) {
            output_layer[k]->respond();
        }
    }


    void train(float learning_rate, float* outputs) {
        // adjust the output layer
        for (int k = 0; k < output_layer.size(); k++) {
            output_layer[k]->setError(outputs[k]);
            output_layer[k]->train(learning_rate);
        }
        float best = -1.0f;
        for (int i = 0; i < output_layer.size(); i++) {
            if (output_layer[i]->output > best) {
              bestIndex = i;
            }
        }
        // propagate back to the hidden layer
        for (int j = 0; j < hidden_layer.size(); j++) {
          hidden_layer[j]->train(learning_rate);
        }

        // The input layer doesn't learn: it is the input and only that
    }

    int getOutput() {
      float best = -1.0f;
      for (int i = 0; i < output_layer.size(); i++) {
        if (output_layer[i]->output > best) {
          best = output_layer[i]->output;
          bestIndex = i;
        }
      }
      return bestIndex;
    }

    void GetParameters(float *buffer) {
      int offset = 0;
      // First the hidden layer
      for (int j = 0; j < hidden_layer.size(); j++) {
        memcpy(buffer + offset, hidden_layer[j]->getWeights(), sizeof(float) * input_layer.size());
        offset += input_layer.size();
      }

      // Then the output layer
      for (int i = 0; i < output_layer.size(); i++) {
        memcpy(buffer + offset, output_layer[i]->getWeights(), sizeof(float) * hidden_layer.size());
        offset += hidden_layer.size();
      }
    }

    void UpdateParameters(float *buffer, int clusterSize, bool doAverage) {
      int offset = 0;
      // First the hidden layer
      for (int j = 0; j < hidden_layer.size(); j++) {
        float *updateWeights = buffer + offset;
        if (doAverage) {
          for (int k = 0; k < input_layer.size(); k++) {
            updateWeights[k] /= clusterSize;
          }
        }
        hidden_layer[j]->setWeights(updateWeights);
        offset += input_layer.size();
      }

      // Then the output layer
      for (int i = 0; i < output_layer.size(); i++) {
        float *updateWeights = buffer + offset;
        if (doAverage) {
          for (int k = 0; k < hidden_layer.size(); k++) {
            updateWeights[k] /= clusterSize;
          }
        }
        output_layer[i]->setWeights(updateWeights);
        offset += hidden_layer.size();
      }
    }

    void BroadcastParameters(int rank, int clusterSize) {
      double start = currentSeconds();
      if (rank == 0) {
        GetParameters(weightSendBuffer);
      }
      MPI_Bcast(weightSendBuffer, bufferSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
      if (rank != 0) {
        UpdateParameters(weightSendBuffer, clusterSize, false);
      }
      communicationTime += (currentSeconds() - start);
    }

    void AverageParameters(int rank, int clusterSize) {
      double start = currentSeconds();
      GetParameters(weightSendBuffer);
      MPI_Allreduce(weightSendBuffer, weightReceiveBuffer, bufferSize, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      UpdateParameters(weightReceiveBuffer, clusterSize, true);
      communicationTime += (currentSeconds() - start);
    }
};

#endif //_NETWORK_H
