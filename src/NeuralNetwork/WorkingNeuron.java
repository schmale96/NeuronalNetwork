package NeuralNetwork;

import NeuralNetwork.ActivationFunctions.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public class WorkingNeuron extends Neuron {

    private List<Connection> connections = new ArrayList<>();
    private ActivationFunction activationFunction = ActivationFunction.ActivationHyperbolicTangent;


    @Override
    public float getValue() {

        float sum = 0;

        for(Connection c : connections) {

            sum += c.getValue();

        }

        return activationFunction.activation(sum);
    }

    public void addConnection(Connection c) {

        connections.add(c);
    }

}
