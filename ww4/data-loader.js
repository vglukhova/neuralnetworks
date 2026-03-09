/**
 * data-loader.js
 * MNISTDataLoader — исправленная версия
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
     */
    async loadCSVFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (event) => {
                try {
                    const content = event.target.result;
                    console.log('File loaded, length:', content.length);
                    
                    // Split into lines and filter out empty lines
                    const lines = content.split('\n').filter(line => line.trim() !== '');
                    console.log('Number of lines:', lines.length);

                    const labels = [];
                    const pixels = [];

                    for (let i = 0; i < lines.length; i++) {
                        const line = lines[i].trim();
                        if (line === '') continue;
                        
                        const values = line.split(',').map(Number);
                        
                        // Check if we have the right number of values
                        if (values.length !== 785) {
                            console.warn(`Line ${i + 1}: expected 785 values, got ${values.length}`);
                            continue;
                        }

                        labels.push(values[0]);
                        pixels.push(values.slice(1));
                    }

                    if (labels.length === 0) {
                        reject(new Error('No valid data rows found in CSV'));
                        return;
                    }

                    console.log(`Loaded ${labels.length} valid samples`);

                    // Create tensors
                    const xs = tf.tidy(() => {
                        // Create tensor from pixels array
                        const pixelTensor = tf.tensor2d(pixels);
                        // Normalize to [0,1] and reshape to [N, 28, 28, 1]
                        return pixelTensor
                            .div(255)
                            .reshape([labels.length, 28, 28, 1]);
                    });

                    const ys = tf.tidy(() => tf.oneHot(labels, 10));

                    resolve({ 
                        xs, 
                        ys, 
                        count: labels.length 
                    });

                } catch (error) {
                    console.error('Error parsing CSV:', error);
                    reject(error);
                }
            };

            reader.onerror = () => reject(new Error('FileReader failed to read file'));
            reader.readAsText(file);
        });
    }

    /** Load and store training data from a user-selected file. */
    async loadTrainFromFiles(file) {
        try {
            console.log('Loading train file:', file.name);
            const data = await this.loadCSVFile(file);
            this.trainData = data;
            console.log('Train data loaded:', data.count, 'samples');
            return data;
        } catch (error) {
            console.error('Error loading train data:', error);
            throw error;
        }
    }

    /** Load and store test data from a user-selected file. */
    async loadTestFromFiles(file) {
        try {
            console.log('Loading test file:', file.name);
            const data = await this.loadCSVFile(file);
            this.testData = data;
            console.log('Test data loaded:', data.count, 'samples');
            return data;
        } catch (error) {
            console.error('Error loading test data:', error);
            throw error;
        }
    }

    /**
     * Split xs/ys into train and validation subsets.
     */
    splitTrainVal(xs, ys, valRatio = 0.1) {
        return tf.tidy(() => {
            const total = xs.shape[0];
            const numVal = Math.floor(total * valRatio);
            const numTrain = total - numVal;

            return {
                trainXs: xs.slice([0, 0, 0, 0], [numTrain, 28, 28, 1]),
                trainYs: ys.slice([0, 0], [numTrain, 10]),
                valXs: xs.slice([numTrain, 0, 0, 0], [numVal, 28, 28, 1]),
                valYs: ys.slice([numTrain, 0], [numVal, 10]),
            };
        });
    }

    /**
     * Sample k random images from the test set.
     */
    getRandomTestBatch(xs, ys, k = 5) {
        return tf.tidy(() => {
            const total = xs.shape[0];
            // Generate random indices
            const indices = [];
            for (let i = 0; i < k; i++) {
                indices.push(Math.floor(Math.random() * total));
            }
            
            return {
                batchXs: tf.gather(xs, indices),
                batchYs: tf.gather(ys, indices),
                indices: indices,
            };
        });
    }

    /**
     * Render a single [28, 28, 1] tensor onto a canvas element.
     */
    draw28x28ToCanvas(tensor, canvas, scale = 4) {
        tf.tidy(() => {
            // Ensure tensor is 2D [28, 28]
            let tensor2d = tensor;
            if (tensor.shape.length === 4) {
                tensor2d = tensor.reshape([28, 28]);
            } else if (tensor.shape.length === 3) {
                tensor2d = tensor.reshape([28, 28]);
            }
            
            // Get the data as a flat array
            const data = tensor2d.mul(255).clipByValue(0, 255).dataSync();
            
            // Create ImageData
            const imageData = new ImageData(28, 28);
            
            for (let i = 0; i < 784; i++) {
                const value = Math.round(data[i]);
                imageData.data[i * 4] = value;     // R
                imageData.data[i * 4 + 1] = value; // G
                imageData.data[i * 4 + 2] = value; // B
                imageData.data[i * 4 + 3] = 255;   // A
            }
            
            // Set canvas size
            canvas.width = 28 * scale;
            canvas.height = 28 * scale;
            
            // Create temporary canvas
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(imageData, 0, 0);
            
            // Draw scaled
            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
        });
    }

    /** Dispose stored tensors */
    dispose() {
        if (this.trainData) {
            if (this.trainData.xs) this.trainData.xs.dispose();
            if (this.trainData.ys) this.trainData.ys.dispose();
            this.trainData = null;
        }
        if (this.testData) {
            if (this.testData.xs) this.testData.xs.dispose();
            if (this.testData.ys) this.testData.ys.dispose();
            this.testData = null;
        }
    }
}
