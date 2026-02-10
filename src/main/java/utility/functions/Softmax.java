package utility.functions;

public class Softmax extends Function {

    // --- SCALAR METHODS (Invalid for Softmax) ---

    @Override
    public float evaluate(float x) {
        throw new UnsupportedOperationException("Softmax requires a vector input.");
    }

    @Override
    public float derivative(float x) {
        // The derivative is handled via the (A - Y) simplification in Layer.java.
        return 1.0f;
    }

    // --- VECTOR METHODS ---

    @Override
    public float[] evaluate(float[] inputs) {
        float[] outputs = new float[inputs.length];
        
        // 1. Find max for numerical stability
        float max = inputs[0];
        for (float val : inputs) {
            if (val > max) max = val;
        }

        // 2. Compute Exponentials and Sum
        float sum = 0.0f;
        for (int i = 0; i < inputs.length; i++) {
            // Subtract max to prevent overflow
            outputs[i] = (float) Math.exp(inputs[i] - max);
            sum += outputs[i];
        }

        // 3. Normalize (Divide by sum)
        for (int i = 0; i < inputs.length; i++) {
            outputs[i] /= sum;
        }

        return outputs;
    }
}