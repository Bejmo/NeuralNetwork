package utility.functions;

public class ReluFunction extends Function {

    // RELU function
    @Override
    public float evaluate(float x) {
        return x > 0 ? x : 0;
    }
    
    // Derivative of RELU
    @Override
    public float derivative(float x) {
        return x > 0 ? 1 : 0;
    }
}