/**
 * data-loader.js
 * MNISTDataLoader — identical API to the original course file.
 * Parses MNIST CSV files in the browser using FileReader (no network requests).
 * CSV format: label (0-9) followed by 784 pixel values (0-255), no header row.
 */
class MNISTDataLoader {
    constructor() {
        this.trainData = null;
        this.testData  = null;
    }

    /**
     * Parse a single CSV file and return normalised tensors.
     * @param {File} file  — user-selected .csv File object
     * @returns {Promise<{xs: Tensor4D, ys: Tensor2D, count: number}>}
     *   xs  — shape [N, 28, 28, 1], float32 in [0, 1]
     *   ys  — shape [N, 10], one-hot encoded labels
     */
    async loadCSVFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (event) => {
                try {
                    const content = event.target.result;
                    // Split into non-empty lines
                    const lines = content.split('\n').filter(line => line.trim() !== '');

                    const labels = [];
                    const pixels = [];

                    for (const line of lines) {
                        const values = line.split(',').map(Number);
                        // Each row must have exactly: 1 label + 784 pixels = 785 values
                        if (values.length !== 785) continue;

                        labels.push(values[0]);           // integer class label
                        pixels.push(values.slice(1));     // 784 pixel intensities
                    }

                    if (labels.length === 0) {
                        reject(new Error('No valid data rows found in CSV'));
                        return;
                    }

                    // Build xs: normalize pixels to [0,1] and reshape to [N, 28, 28, 1]
                    const xs = tf.tidy(() =>
                        tf.tensor2d(pixels)
                          .div(255)
                          .reshape([labels.length, 28, 28, 1])
                    );

                    // Build ys: one-hot encode labels to depth 10
                    const ys = tf.tidy(() => tf.oneHot(labels, 10));

                    resolve({ xs, ys, count: labels.length });

                } catch (error) {
                    reject(error);
                }
            };

            reader.onerror = () => reject(new Error('FileReader failed to read file'));
            reader.readAsText(file);
        });
    }

    /** Load and store training data from a user-selected file. */
    async loadTrainFromFiles(file) {
        this.trainData = await this.loadCSVFile(file);
        return this.trainData;
    }

    /** Load and store test data from a user-selected file. */
    async loadTestFromFiles(file) {
        this.testData = await this.loadCSVFile(file);
        return this.testData;
    }

    /**
     * Split xs/ys into train and validation subsets.
     * Uses tf.tidy so the original tensors are NOT consumed.
     * @param {number} valRatio  — fraction reserved for validation (default 0.1)
     * @returns {{ trainXs, trainYs, valXs, valYs }}
     */
    splitTrainVal(xs, ys, valRatio = 0.1) {
        return tf.tidy(() => {
            const numVal   = Math.floor(xs.shape[0] * valRatio);
            const numTrain = xs.shape[0] - numVal;

            return {
                trainXs: xs.slice([0,       0, 0, 0], [numTrain, 28, 28, 1]),
                trainYs: ys.slice([0,       0],        [numTrain, 10]),
                valXs:   xs.slice([numTrain,0, 0, 0], [numVal,   28, 28, 1]),
                valYs:   ys.slice([numTrain,0],        [numVal,   10]),
            };
        });
    }

    /**
     * Sample k random images from the test set using a shuffled index array.
     * Returns new tensors (not slices of the originals) so callers can dispose safely.
     * @returns {{ batchXs: Tensor4D, batchYs: Tensor2D, indices: number[] }}
     */
    getRandomTestBatch(xs, ys, k = 5) {
        return tf.tidy(() => {
            // tf.util.createShuffledIndices returns a Uint32Array
            const shuffled  = tf.util.createShuffledIndices(xs.shape[0]);
            const selected  = Array.from(shuffled.slice(0, k));

            return {
                batchXs:  tf.gather(xs, selected),
                batchYs:  tf.gather(ys, selected),
                indices:  selected,
            };
        });
    }

    /**
     * Render a single [28, 28, 1] float32 tensor onto a canvas element.
     * Uses a temporary 28×28 canvas then scales up with drawImage for sharpness.
     * @param {Tensor}  tensor  — shape [28,28,1] or [28,28], values in [0,1]
     * @param {HTMLCanvasElement} canvas
     * @param {number}  scale   — pixel multiplier (default 4 → 112×112 px)
     */
    draw28x28ToCanvas(tensor, canvas, scale = 4) {
        tf.tidy(() => {
            const imageData = new ImageData(28, 28);

            // Flatten to [784] and denormalize to [0, 255]
            const data = tensor.reshape([28, 28]).mul(255).clipByValue(0, 255).dataSync();

            for (let i = 0; i < 784; i++) {
                const v = Math.round(data[i]);
                imageData.data[i * 4]     = v;   // R
                imageData.data[i * 4 + 1] = v;   // G
                imageData.data[i * 4 + 2] = v;   // B
                imageData.data[i * 4 + 3] = 255; // A (fully opaque)
            }

            // Set dimensions first, then get context (avoids cleared/invalid context)
            canvas.width  = 28 * scale;
            canvas.height = 28 * scale;
            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = false;  // keep pixels sharp

            const tmp  = document.createElement('canvas');
            tmp.width  = 28;
            tmp.height = 28;
            tmp.getContext('2d').putImageData(imageData, 0, 0);

            ctx.drawImage(tmp, 0, 0, 28 * scale, 28 * scale);
        });
    }

    /** Dispose stored tensors to prevent memory leaks. */
    dispose() {
        if (this.trainData) {
            this.trainData.xs.dispose();
            this.trainData.ys.dispose();
            this.trainData = null;
        }
        if (this.testData) {
            this.testData.xs.dispose();
            this.testData.ys.dispose();
            this.testData = null;
        }
    }
}
