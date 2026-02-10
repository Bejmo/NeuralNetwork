package utility;

public class MatrixMath {

    private MatrixMath() {}

    // Optimized Matrix-Matrix Multiplication (Standard)
    // A (RowsA x ColsA) * B (ColsA x ColsB) = Result (RowsA x ColsB)
    public static float[][] multiply(float[][] A, float[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;

        if (colsA != B.length) { // colsA = rowsB
            throw new IllegalArgumentException("Matrix dimensions mismatch for multiplication.");
        }

        float[][] result = new float[rowsA][colsB];

        // IKJ loop optimization for cache locality
        for (int i = 0; i < rowsA; i++) {
            float[] rowA = A[i];
            float[] rowC = result[i];
            for (int k = 0; k < colsA; k++) {
                float valA = rowA[k];
                if (valA == 0.0f) continue; // Stop early
                
                float[] rowB = B[k];
                for (int j = 0; j < colsB; j++) {
                    rowC[j] += valA * rowB[j];
                }
            }
        }
        return result;
    }

    // Transpose a Matrix
    public static float[][] transpose(float[][] M) {
        int rows = M.length;
        int cols = M[0].length;
        float[][] result = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = M[i][j];
            }
        }
        return result;
    }

    // Add a bias vector to every row of a matrix
    public static void addBiasToBatch(float[][] matrix, float[] bias) {
        int rows = matrix.length;    // Batch Size
        int cols = matrix[0].length; // Neuron Count

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] += bias[j];
            }
        }
    }
}