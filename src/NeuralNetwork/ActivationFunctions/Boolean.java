package NeuralNetwork.ActivationFunctions;

public class Boolean implements ActivationFunction {

    @Override
    public float activation(float input) {
        if (input < 0) return 0;
        else return 1;
    }
}
