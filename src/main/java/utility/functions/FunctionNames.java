package utility.functions;

public enum FunctionNames {
    // Singleton
    SIGMOID("Sigmoid", new SigmoidFunction()),
    RELU("Relu", new ReluFunction()),
    SOFTMAX("Softmax", new Softmax());

    private final String name;
    private final Function function;

    FunctionNames(String name, Function function) {
        this.name = name;
        this.function = function;
    }

    public String getName() {
        return name;
    }

    // Return the instance
    public Function getFunction() {
        return function;
    }

    public static Function getFunctionByName(String name) {
        for (FunctionNames fn : FunctionNames.values()) {
            if (fn.getName().equalsIgnoreCase(name)) {
                return fn.getFunction();
            }
        }
        return null; // Throws exception
    }
}