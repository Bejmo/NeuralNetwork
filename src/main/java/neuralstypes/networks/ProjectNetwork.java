package neuralstypes.networks;

import neuralstypes.Layer;

public class ProjectNetwork extends NeuralNetwork {

    private final int pictureSize = 28;
    private final int outputNumber = 10;
    
    private static final int HIDDEN_SIZE_1 = 128;
    private static final int HIDDEN_SIZE_2 = 64;

    public ProjectNetwork() {
        super();
        createProjectNetwork();
    }

    public ProjectNetwork(long seed) {
        super(seed);
        createProjectNetwork();
    }

    private void createProjectNetwork() {
        int inputPixels = pictureSize * pictureSize;

        // Hidden Layer 1: Input -> 128
        this.addLayer(new Layer(inputPixels, HIDDEN_SIZE_1, "RELU", this.randomGenerator));

        // Hidden Layer 2: 128 -> 64
        this.addLayer(new Layer(HIDDEN_SIZE_1, HIDDEN_SIZE_2, "RELU", this.randomGenerator));

        // Output Layer: 64 -> 10 (Softmax)
        this.addLayer(new Layer(HIDDEN_SIZE_2, outputNumber, "SOFTMAX", this.randomGenerator));



        System.out.println("Network initialized: Inputs(784) -> Hidden1(128) -> Hidden2(64) -> Output(10).");
    }

    // Renamed to 'predict' to avoid conflict with NeuralNetwork.compute(float[]) returning float[]
    public float predict(float[] inputs) {
        float[][] batchInput = new float[1][inputs.length];
        batchInput[0] = inputs;
        float[][] result = forwardBatch(batchInput);

        // Select the index with the highest probability
        return outputSelector(result[0]);
    }

    private int outputSelector(float[] outputs) {
        float max = -1.0f; 
        int argmax = 0;
        for (int i = 0; i < outputs.length; i++) {
            if (outputs[i] > max) {
                argmax = i;
                max = outputs[i];
            }
        }
        return argmax;
    }
}