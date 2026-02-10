package neuralstypes;

import java.util.Random;

import utility.MatrixMath;
import utility.functions.Function;
import utility.functions.FunctionNames;
import utility.functions.Softmax;

public class Layer {

    private final int inputSize;
    private final int neuronCount;

    // --- Parameters ---
    private final float[][] weights;            // [Neuron][Input]
    private final float[][] weightsTransposed;  // [Input][Neuron] (Cached for forward pass)
    private final float[] biases;               // [Neuron]

    // --- Momentum ---
    private final float[][] weightVelocities;
    private final float[] biasVelocities;


    // --- Gradients for batch accumulation ---
    private final float[][] dBiases;          // [neuron]
    private final float[][] dWeights;       // [neuron][input]


    // --- Batch Cache (Now 2D Arrays) ---
    private float[][] batchInputs;         // [BatchSize][InputSize]
    private float[][] batchPreActivations; // [BatchSize][NeuronCount]
    private float[][] batchOutputs;        // [BatchSize][NeuronCount]
    private float[][] batchDeltas;         // [BatchSize][NeuronCount]

    private final Function activationFunction;

    public Layer(int inputSize, int neuronCount, String activationName, Random randomGenerator) {
        this.inputSize = inputSize;
        this.neuronCount = neuronCount;

        Function found = FunctionNames.getFunctionByName(activationName);
        this.activationFunction = (found != null) ? found : FunctionNames.RELU.getFunction();

        this.weights = new float[neuronCount][inputSize];
        this.weightsTransposed = new float[inputSize][neuronCount];
        this.biases = new float[neuronCount];
        this.weightVelocities = new float[neuronCount][inputSize];
        this.biasVelocities = new float[neuronCount];

        this.dBiases = new float[neuronCount][1];
        this.dWeights = new float[neuronCount][inputSize];


        // Xavier init
        float limit = (float) (1.0 / Math.sqrt(inputSize));
        for (int i = 0; i < neuronCount; i++) {
            for (int j = 0; j < inputSize; j++) {
                float w = (randomGenerator.nextFloat() * 2 - 1) * limit;
                weights[i][j] = w;
                weightsTransposed[j][i] = w;
            }
            biases[i] = 0.0f;
        }
    }


    // ================================================================
    // Reset accumulated gradients before each batch
    // ================================================================
    public void resetGradients() {
        for (int i = 0; i < neuronCount; i++) {
            dBiases[i][0] = 0f;
            for (int j = 0; j < inputSize; j++) {
                dWeights[i][j] = 0f;
            }
        }
    }


    // ================================================================
    // Forward pass (batch)
    // ================================================================
    public float[][] forward(float[][] inputs) {

        int batchSize = inputs.length;
        this.batchInputs = inputs;

        // 1. Z = Inputs * W^T
        // Inputs: [Batch x In], W^T: [In x Neurons] -> Result: [Batch x Neurons]
        this.batchPreActivations = MatrixMath.multiply(inputs, this.weightsTransposed);

        // 2. Add Biases (Broadcast to every row in batch)
        MatrixMath.addBiasToBatch(this.batchPreActivations, this.biases);

        // 3. Apply Activation

        this.batchOutputs = new float[batchSize][neuronCount];

        for (int i = 0; i < batchSize; i++) {
            // Apply activation to each vector in the batch
            this.batchOutputs[i] = activationFunction.evaluate(batchPreActivations[i]);
        }

        return this.batchOutputs;
    }

    // --- BACKPROPAGATION: CALCULATE DELTAS (BATCH) ---

    public void computeOutputDeltas(float[][] targets) {

        int batch = batchOutputs.length;
        batchDeltas = new float[batch][neuronCount];

        for (int b = 0; b < batch; b++) {
            for (int n = 0; n < neuronCount; n++) {

                float out = batchOutputs[b][n];

                // For softmax cross entropy: delta = out - target
                float delta;
                if (activationFunction instanceof Softmax) {
                    delta = out - targets[b][n];
                } else {
                    delta = (out - targets[b][n]) * activationFunction.derivative(out);
                }

                batchDeltas[b][n] = delta;

                // Accumulate bias gradient
                dBiases[n][0] += delta;

                // Accumulate weight gradient
                for (int j = 0; j < inputSize; j++) {
                    dWeights[n][j] += batchInputs[b][j] * delta;
                }
            }
        }
    }


    // ================================================================
// Compute deltas for hidden layers
// ================================================================
    public void computeHiddenDeltas(Layer nextLayer) {

        int batch = batchOutputs.length;
        batchDeltas = new float[batch][this.neuronCount];
        // Next Deltas: [Batch x NextNeurons]
        // Next Weights: [NextNeurons x CurrentNeurons]
        // Result: [Batch x CurrentNeurons]
        float[][] nextDeltas = nextLayer.batchDeltas;
        float[][] nextWeights = nextLayer.weights;

        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < neuronCount; i++) {

                float sum = 0;

                for (int k = 0; k < nextLayer.neuronCount; k++) {
                    sum += nextDeltas[b][k] * nextWeights[k][i];
                }

                float activated = batchOutputs[b][i];
                float delta = sum * activationFunction.derivative(activated);

                batchDeltas[b][i] = delta;

                dBiases[i][0] += delta;

                for (int j = 0; j < inputSize; j++) {
                    dWeights[i][j] += batchInputs[b][j] * delta;
                }
            }
        }


    }

    // --- UPDATE WEIGHTS (AVERAGED OVER BATCH) ---

    public void updateWeights(float learningRate, float momentum) {
        int batchSize = batchInputs.length;
        float scale = learningRate / batchSize;

        for (int i = 0; i < neuronCount; i++) {

            // --- Update bias ---
            float db = dBiases[i][0] * scale;
            biasVelocities[i] = momentum * biasVelocities[i] + db;
            biases[i] -= biasVelocities[i];

            // Update Weights using calculated matrix gradients
            for (int j = 0; j < inputSize; j++) {

                float grad = dWeights[i][j] * scale;

                float velocity = momentum * weightVelocities[i][j] + grad;
                weightVelocities[i][j] = velocity;

                weights[i][j] -= velocity;
                weightsTransposed[j][i] = weights[i][j];
            }
        }
    }


    // --- GETTERS ---
    public float[][] getWeights() {
        return weights;
    }

    public float[][] getTransposedWeights() {
        return weightsTransposed;
    }

    public float[][] getDeltas() {
        return batchDeltas;
    }

    public float[][] getOutputs() {
        return batchOutputs;
    }
}