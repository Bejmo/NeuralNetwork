package neuralstypes.networks;

import neuralstypes.Layer;

public class XORNetwork extends NeuralNetwork {

    // Creates the networks with a selected seed
    public XORNetwork(long seed) {
        super(seed);
        creatorXORNetwork();
    }
    
    // Creates the networks with a random seed
    public XORNetwork() {
        super();
        creatorXORNetwork();
    }

    // Define the network architecture in the constructor
    private void creatorXORNetwork() {
        // The XOR network has 2 inputs, 2 hidden neurons, and 1 output.

        // --- 1. Hidden Layer ---
        // Input: 2 (X1, X2), Neurons: 2, Activation: Sigmoid
        Layer hiddenLayer = new Layer(2, 2, "Sigmoid", this.randomGenerator);
        this.addLayer(hiddenLayer);

        // --- 2. Output Layer ---
        // Input: 2 (from previous hidden layer), Neurons: 1, Activation: Sigmoid
        Layer outputLayer = new Layer(2, 1, "Sigmoid", this.randomGenerator);
        this.addLayer(outputLayer);
        
        System.out.println("XOR Network initialized: 2 inputs -> 2 hidden -> 1 output.");
    }
}