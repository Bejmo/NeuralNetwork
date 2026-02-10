package datatreatments;

import java.util.*;

/**
 * Creates mini-batches grouped by label, and provides
 * an epoch-based randomized iterator that guarantees
 * no repetition until all batches are consumed.
 */
public class MiniBatchesSamples {

    public static class MiniBatch {
        public final float[][] inputs;
        public final int[] labels;

        public MiniBatch(float[][] inputs, int[] labels) {
            this.inputs = inputs;
            this.labels = labels;
        }
    }

    // All batches grouped by label
    private final Map<Integer, List<MiniBatch>> batchesPerLabel = new HashMap<>();

    // Flattened and shuffled at each epoch
    private final List<MiniBatch> epochBatches = new ArrayList<>();

    private int epochPointer = 0;
    private final Random random = new Random();

    public MiniBatchesSamples(FashionMNISTDataLoader.DataSet data, int batchSize) {

        //Group data by label
        Map<Integer, List<float[]>> vectorsByLabel = new HashMap<>();
        for (int i = 0; i < 10; i++) vectorsByLabel.put(i, new ArrayList<>());
        for (int i = 0; i < data.labels.length; i++) {
            vectorsByLabel.get(data.labels[i]).add(data.inputs[i]);
        }

        //Create mini-batches per label
        for (int label = 0; label < 10; label++) {
            List<float[]> vecs = vectorsByLabel.get(label);
            List<MiniBatch> list = new ArrayList<>();

            int index = 0;
            while (index < vecs.size()) {
                int end = Math.min(index + batchSize, vecs.size());
                List<float[]> batchVecs = vecs.subList(index, end);

                float[][] inputs = new float[batchVecs.size()][];
                int[] lbls = new int[batchVecs.size()];
                for (int j = 0; j < batchVecs.size(); j++) {
                    inputs[j] = batchVecs.get(j);
                    lbls[j] = label;
                }

                list.add(new MiniBatch(inputs, lbls));
                index = end;
            }

            batchesPerLabel.put(label, list);
        }

        // Initialize first epoch
        startNewEpoch();
    }

    /**
     * Flattens all mini-batches into a single list and shuffles them.
     * This guarantees that all batches are used exactly once per epoch.
     */
    public void startNewEpoch() {
        epochBatches.clear();

        // Flatten all batches into epoch list
        for (List<MiniBatch> list : batchesPerLabel.values()) {
            epochBatches.addAll(list);
        }

        // Shuffle for randomness
        Collections.shuffle(epochBatches, random);

        // Restart pointer
        epochPointer = 0;
    }

    /** Returns true if there are still unused batches in the epoch. */
    public boolean hasMoreBatchesInEpoch() {
        return epochPointer < epochBatches.size();
    }

    /** Returns the next batch and moves the pointer. */
    public MiniBatch getNextBatch() {
        if (!hasMoreBatchesInEpoch()) return null;
        return epochBatches.get(epochPointer++);
    }


}
