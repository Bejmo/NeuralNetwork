package datatreatments;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FashionMNISTDataLoader {

    private static final String TRAIN_VECTORS = "data/fashion_mnist_train_vectors.csv";
    private static final String TRAIN_LABELS = "data/fashion_mnist_train_labels.csv";
    private static final String TEST_VECTORS = "data/fashion_mnist_test_vectors.csv";
    private static final String TEST_LABELS = "data/fashion_mnist_test_labels.csv";

    private static final int INPUT_SIZE = 784;
    private static final int OUTPUT_SIZE = 10;

    public static class DataSet {
        public float[][] inputs;
        public float[][] targets;
        public int[] labels;

        public DataSet(float[][] inputs, float[][] targets, int[] labels) {
            this.inputs = inputs;
            this.targets = targets;
            this.labels = labels;
        }

        public int size() {
            return inputs.length;
        }
    }

    // Load the Training set
    public DataSet loadTrainingData() throws IOException {
        System.out.println("Loading TRAINING data...");
        return loadData(TRAIN_VECTORS, TRAIN_LABELS);
    }

    // Load the Test set
    public DataSet loadTestData() throws IOException {
        System.out.println("Loading TEST data...");
        return loadData(TEST_VECTORS, TEST_LABELS);
    }

    private DataSet loadData(String vectorsFile, String labelsFile) throws IOException {
        List<Integer> tempLabels = new ArrayList<>();
        List<float[]> tempVectors = new ArrayList<>();

        loadLabels(labelsFile, tempLabels);
        loadVectors(vectorsFile, tempVectors);

        if (tempVectors.size() != tempLabels.size()) {
            throw new IllegalStateException("Size mismatch in file: " + vectorsFile);
        }

        int dataSize = tempLabels.size();
        float[][] inputsArray = tempVectors.toArray(new float[0][0]);
        int[] labelsArray = new int[dataSize];
        float[][] targetsArray = new float[dataSize][OUTPUT_SIZE];

        for (int i = 0; i < dataSize; i++) {
            int label = tempLabels.get(i);
            labelsArray[i] = label;
            if (label >= 0 && label < OUTPUT_SIZE) {
                targetsArray[i][label] = 1.0f;
            }
        }

        return new DataSet(inputsArray, targetsArray, labelsArray);
    }

    private void loadLabels(String filePath, List<Integer> targetList) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    targetList.add(Integer.valueOf(line.trim()));
                }
            }
        }
    }

    private void loadVectors(String filePath, List<float[]> targetList) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                if (values.length < INPUT_SIZE) continue;

                float[] vector = new float[INPUT_SIZE];
                for (int i = 0; i < INPUT_SIZE; i++) {
                    // IMPORTANT: normalize the data so we don't have overflow.
                    vector[i] = Float.parseFloat(values[i].trim()) / 255.0f;
                }
                targetList.add(vector);
            }
        }
    }
}