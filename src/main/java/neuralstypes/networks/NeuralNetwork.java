package neuralstypes.networks;

import java.util.ArrayList;
import java.util.Random;

import neuralstypes.Layer;

public abstract class NeuralNetwork {

    protected final Random randomGenerator;
    private final ArrayList<Layer> layers = new ArrayList<>();



    // ------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------
    public NeuralNetwork() {
        this.randomGenerator = new Random();
    }

    public NeuralNetwork(long seed) {
        this.randomGenerator = new Random(seed);
    }

    // ------------------------------------------------------------
    // Network building
    // ------------------------------------------------------------
    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public int size() {
        return layers.size();
    }



    // --- FORWARD (BATCH) ---
    public float[][] forwardBatch(float[][] inputs) {
        float[][] currentInputs = inputs;
        for (Layer layer : layers) {
            currentInputs = layer.forward(currentInputs);
        }
        return currentInputs;
    }

    // ------------------------------------------------------------
    // Backpropagation for batch
    // ------------------------------------------------------------
    private void backwardBatch(float[][] targets) {

        int L = layers.size();

        // Output layer
        layers.get(L - 1).computeOutputDeltas(targets);

        // Hidden layers (back to front)
        for (int i = L - 2; i >= 0; i--) {
            Layer current = layers.get(i);
            Layer next = layers.get(i + 1);

            current.computeHiddenDeltas(next);
        }
    }




    // ------------------------------------------------------------
    // Update weights for all layers
    // ------------------------------------------------------------
    private void updateWeights(float learningRate,float momentum ) {

        for (Layer layer : layers) {
            layer.updateWeights(learningRate, momentum);
        }
    }

    // ------------------------------------------------------------
    // Reset gradients for a new batch
    // ------------------------------------------------------------
    private void resetAllGradients() {

        for (Layer layer : layers) {
            layer.resetGradients();
        }
    }

    // ------------------------------------------------------------
    // Train one batch
    // ------------------------------------------------------------
    public void trainBatch(float[][] inputsBatch, float[][] targetsBatch,float learningRate,float momentum) {

        resetAllGradients();

        forwardBatch(inputsBatch);

        backwardBatch(targetsBatch);

        updateWeights(learningRate,momentum);
    }


    public float[][] getFinalOutputs() {
        if (layers.isEmpty()) return new float[0][0];
        return layers.get(layers.size() - 1).getOutputs();
    }
}