import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

import datatreatments.FashionMNISTDataLoader;
import datatreatments.FashionMNISTDataLoader.DataSet;
import neuralstypes.networks.ProjectNetwork;
import neuralstypes.networks.XORNetwork;

public class Main {

    public static void main(String[] args) {
        boolean input = false;

        if (input) {
            try (Scanner scanner = new Scanner(System.in)) {
                System.out.println("==========================================");
                System.out.println("   NEURAL NETWORK SELECTION MENU");
                System.out.println("==========================================");
                System.out.println("1. Run XOR Network Example");
                System.out.println("2. Run Fashion MNIST (ProjectNetwork)");
                System.out.print("Select an option [1-2]: ");
    
                String choice = scanner.nextLine().trim();
    
                switch (choice) {
                    case "1" -> {
                        System.out.println("\n--- Starting Neural Network Training for XOR ---");
                        Long seed = askForSeed(scanner);
                        XORNet(seed);
                    }
                    case "2" -> {
                        System.out.println("\n--- Starting Neural Network Training for Fashion MNIST ---");
                        Long seed = askForSeed(scanner);
                        ProjectNet(seed);
                    }
                    default -> System.out.println("Invalid option selected. Exiting.");
                }
            }
        }
        else ProjectNet(null);
    }

    // Saves the predictions in a .csv file
    private static void savePredictions(ProjectNetwork network, DataSet testData, String filename) {
        System.out.println("Generating prediction file: " + filename + "...");

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            for (int i = 0; i < testData.size(); i++) {
                float predictionIndex = network.predict(testData.inputs[i]);

                writer.write(String.valueOf((int)predictionIndex));
                writer.newLine();
            }

            System.out.println("File generated.");

        } catch (IOException e) {
            System.err.println("Error writing in the CSV file: " + e.getMessage());
        }
    }

    // --- OPTION 1: XOR NETWORK ---
    private static void XORNet(Long seed) {
        XORNetwork xorNet = (seed != null) ? new XORNetwork(seed) : new XORNetwork();

        float[][] trainingInputs = {
            {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}
        };
        float[][] expectedOutputs = {
            {0.0f}, {1.0f}, {1.0f}, {0.0f}
        };

        float learningRate = 0.5f;
        float momentum = 0.9f;
        int epochs = 10000;

        System.out.println("Training (Epochs: " + epochs + ", LR: " + learningRate + ")...");

        for (int e = 0; e < epochs; e++) {
            xorNet.trainBatch(trainingInputs, expectedOutputs, learningRate, momentum);

            if ((e + 1) % 1000 == 0) {
                float[][] results = xorNet.forwardBatch(trainingInputs);
                float totalError = 0.0f;
                for(int i=0; i<4; i++) {
                    float err = expectedOutputs[i][0] - results[i][0];
                    totalError += err * err;
                }
                System.out.printf("Epoch %d: MSE = %.6f%n", (e + 1), totalError / 4);
            }
        }

        System.out.println("\n--- Testing Results ---");
        float[][] outputs = xorNet.forwardBatch(trainingInputs);
        for (int i = 0; i < 4; i++) {
            System.out.printf("Input: [%.0f, %.0f], Expected: %.0f, Predicted: %.4f%n",
                trainingInputs[i][0], trainingInputs[i][1], expectedOutputs[i][0], outputs[i][0]);
        }
    }

    // --- OPTION 2: PROJECT NETWORK ---
    private static void ProjectNet(Long seed) {
        ProjectNetwork projectNet = (seed != null) ? new ProjectNetwork(seed) : new ProjectNetwork();

        FashionMNISTDataLoader loader = new FashionMNISTDataLoader();
        DataSet trainData;
        DataSet testData;

        System.out.println("Loading datasets...");
        try {
            trainData = loader.loadTrainingData();
            testData = loader.loadTestData();
        } catch (IOException e) {
            System.err.println("There was an error loading the datasets. Please check that the files exist and that the format is correct." +
                    "error:"+e.getMessage());


            return;
        }

        int batchSize = 64;
        float learningRate = 0.015f;
        float momentum = 0.9f;
        int epochs = 20;

        int trainSize = trainData.size();
        int testSize = testData.size();

        System.out.println("Training on " + trainSize + " images (Batch Size: " + batchSize + ")");

        long startTime = System.currentTimeMillis();

        for (int e = 0; e < epochs; e++) {

            if (e == 10 || e == 13 || e == 16 || e == 19 ) {
                learningRate /= 2.0f;
                System.out.printf(">>> LR Decayed to: %.6f%n", learningRate);
            }

            // --- TRAINING PHASE (MINI-BATCH) ---
            for (int i = 0; i < trainSize; i += batchSize) {
                int end = Math.min(i + batchSize, trainSize);
                int currentBatchSize = end - i;

                float[][] batchInputs = new float[currentBatchSize][];
                float[][] batchTargets = new float[currentBatchSize][];

                for (int j = 0; j < currentBatchSize; j++) {
                    batchInputs[j] = trainData.inputs[i + j];
                    batchTargets[j] = trainData.targets[i + j];
                }

                projectNet.trainBatch(batchInputs, batchTargets, learningRate, momentum);
            }

            // --- VALIDATION PHASE ---
            int validationCorrect = 0;
            for (int i = 0; i < testSize; i += batchSize) {
                int end = Math.min(i + batchSize, testSize);
                int currentBatchSize = end - i;

                float[][] batchInputs = new float[currentBatchSize][];
                for (int j = 0; j < currentBatchSize; j++) batchInputs[j] = testData.inputs[i + j];

                float[][] batchOutputs = projectNet.forwardBatch(batchInputs);

                for (int j = 0; j < currentBatchSize; j++) {
                    int predicted = getMaxIndex(batchOutputs[j]);
                    int actual = getMaxIndex(testData.targets[i + j]);
                    if (predicted == actual) validationCorrect++;
                }
            }

            long currentTime = System.currentTimeMillis();
            float validationAccuracy = (float) validationCorrect / testSize * 100.0f;

            System.out.printf("Epoch %d/%d - Validation Accuracy: %.2f%% | Time: %.2fs%n",
                    (e + 1), epochs, validationAccuracy, (currentTime - startTime) / 1000.0f);
        }

        System.out.println("Training Complete.");

        savePredictions(projectNet, testData, "test_predictions.csv");
    }

    private static Long askForSeed(Scanner scanner) {
        System.out.print("Introduce a seed (if left blank, we will use a random seed): ");
        String seedInput = scanner.nextLine().trim();
        if (!seedInput.isEmpty()) {
            try {
                return Long.valueOf(seedInput);
            } catch (NumberFormatException e) {
                System.err.println("Invalid seed.");
            }
        }
        return null;
    }

    private static int getMaxIndex(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) maxIndex = i;
        }
        return maxIndex;
    }
}