package utility.functions;

public abstract class Function {
    
    // --- SCALAR METHODS ---
    public abstract float evaluate(float x);
    public abstract float derivative(float x);

    // --- VECTOR METHODS (Added for Softmax) ---
    
    // Computes the function for an entire layer.
    public float[] evaluate(float[] inputs) {
        float[] outputs = new float[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = evaluate(inputs[i]);
        }
        return outputs;
    }

    // Computes derivatives for an entire layer.
    public float[] derivative(float[] inputs) {
        float[] derivs = new float[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            derivs[i] = derivative(inputs[i]);
        }
        return derivs;
    }
}