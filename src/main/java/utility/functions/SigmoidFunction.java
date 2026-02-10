package utility.functions;

public class SigmoidFunction extends Function { // Sigmoid Function

    // Sigmoid function: f(x) = 1 / (1 + e^(-x))
    @Override
    public float evaluate(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }
    
    // Derivative of Sigmoid: f'(x) = f(x) * (1 - f(x))
    @Override
    public float derivative(float x) {
        float output = evaluate(x);
        return output * (1.0f - output);
    }
}