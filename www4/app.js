class MNISTApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();
        this.modelMaxPool = null;
        this.modelAvgPool = null;
        this.isTraining = false;
        this.trainData = null;
        this.testData = null;
        this.currentComparisonType = 'maxpool'; // 'maxpool' or 'avgpool'
        
        this.initializeUI();
    }

    initializeUI() {
        // Bind button events
        document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
        document.getElementById('trainMaxPoolBtn').addEventListener('click', () => this.onTrain('maxpool'));
        document.getElementById('trainAvgPoolBtn').addEventListener('click', () => this.onTrain('avgpool'));
        document.getElementById('denoiseTestBtn').addEventListener('click', () => this.onDenoiseTest());
        document.getElementById('comparePoolingBtn').addEventListener('click', () => this.onComparePooling());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.onSaveDownload());
        document.getElementById('loadModelBtn').addEventListener('click', () => this.onLoadFromFiles());
        document.getElementById('resetBtn').addEventListener('click', () => this.onReset());
        document.getElementById('toggleVisorBtn').addEventListener('click', () => this.toggleVisor());
    }

    async onLoadData() {
        try {
            const trainFile = document.getElementById('trainFile').files[0];
            const testFile = document.getElementById('testFile').files[0];
            
            if (!trainFile || !testFile) {
                this.showError('Please select both train and test CSV files');
                return;
            }

            this.showStatus('Loading training data...');
            const trainData = await this.dataLoader.loadTrainFromFiles(trainFile);
            
            this.showStatus('Loading test data...');
            const testData = await this.dataLoader.loadTestFromFiles(testFile);

            this.trainData = trainData;
            this.testData = testData;

            this.updateDataStatus(trainData.count, testData.count);
            this.showStatus('Data loaded successfully!');
            
        } catch (error) {
            this.showError(`Failed to load data: ${error.message}`);
        }
    }

    async onTrain(poolingType) {
        if (!this.trainData) {
            this.showError('Please load training data first');
            return;
        }

        if (this.isTraining) {
            this.showError('Training already in progress');
            return;
        }

        try {
            this.isTraining = true;
            this.showStatus(`Starting training with ${poolingType} pooling...`);
            
            // Split training data
            const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
                this.trainData.xs, this.trainData.ys, 0.1
            );

            // Add noise to input images (autoencoder uses noisy input, clean output)
            const noisyTrainXs = this.dataLoader.addNoise(trainXs);
            const noisyValXs = this.dataLoader.addNoise(valXs);

            // Create or get model
            if (poolingType === 'maxpool' && !this.modelMaxPool) {
                this.modelMaxPool = this.createDenoisingAutoencoder('maxpool');
            } else if (poolingType === 'avgpool' && !this.modelAvgPool) {
                this.modelAvgPool = this.createDenoisingAutoencoder('avgpool');
            }

            const model = poolingType === 'maxpool' ? this.modelMaxPool : this.modelAvgPool;

            // Train with tfjs-vis callbacks
            const startTime = Date.now();
            const history = await model.fit(noisyTrainXs, trainXs, {
                epochs: 10,
                batchSize: 128,
                validationData: [noisyValXs, valXs],
                shuffle: true,
                callbacks: tfvis.show.fitCallbacks(
                    { name: `Training Performance (${poolingType} pooling)` },
                    ['loss', 'val_loss'],
                    { callbacks: ['onEpochEnd'] }
                )
            });

            const duration = (Date.now() - startTime) / 1000;
            const finalLoss = history.history.loss[history.history.loss.length - 1];
            
            this.showStatus(`${poolingType} pooling training completed in ${duration.toFixed(1)}s. Final loss: ${finalLoss.toFixed(4)}`);
            
            this.updateModelInfo();
            
            // Clean up
            trainXs.dispose();
            trainYs.dispose();
            valXs.dispose();
            valYs.dispose();
            noisyTrainXs.dispose();
            noisyValXs.dispose();
            
        } catch (error) {
            this.showError(`Training failed: ${error.message}`);
        } finally {
            this.isTraining = false;
        }
    }

    async onDenoiseTest() {
        if ((!this.modelMaxPool && !this.modelAvgPool) || !this.testData) {
            this.showError('Please train or load at least one model and test data first');
            return;
        }

        try {
            const { batchXs, indices } = this.dataLoader.getRandomTestBatch(
                this.testData.xs, this.testData.ys, 5
            );
            
            // Add noise to test images
            const noisyBatchXs = this.dataLoader.addNoise(batchXs);
            
            // Get denoised images from both models if available
            let denoisedMaxPool = null;
            let denoisedAvgPool = null;
            
            if (this.modelMaxPool) {
                denoisedMaxPool = this.modelMaxPool.predict(noisyBatchXs);
            }
            if (this.modelAvgPool) {
                denoisedAvgPool = this.modelAvgPool.predict(noisyBatchXs);
            }
            
            // Render comparison
            await this.renderDenoisingComparison(batchXs, noisyBatchXs, denoisedMaxPool, denoisedAvgPool);
            
            // Clean up
            batchXs.dispose();
            noisyBatchXs.dispose();
            if (denoisedMaxPool) denoisedMaxPool.dispose();
            if (denoisedAvgPool) denoisedAvgPool.dispose();
            
        } catch (error) {
            this.showError(`Denoising test failed: ${error.message}`);
        }
    }

    async onComparePooling() {
        if (!this.modelMaxPool || !this.modelAvgPool || !this.testData) {
            this.showError('Please train both MaxPool and AvgPool models first');
            return;
        }

        try {
            // Get three specific digits for comparison
            const digitIndices = this.findDigitIndices(this.testData.ys, [2, 3, 5]);
            
            if (digitIndices.length < 3) {
                this.showError('Could not find required digits (2, 3, 5) in test set');
                return;
            }

            const comparisonContainer = document.getElementById('comparisonContainer');
            comparisonContainer.innerHTML = '<h3>Max Pooling vs Average Pooling Comparison</h3>';
            
            for (let i = 0; i < digitIndices.length; i++) {
                const idx = digitIndices[i];
                const original = tf.gather(this.testData.xs, [idx]);
                const noisy = this.dataLoader.addNoise(original);
                
                const denoisedMax = this.modelMaxPool.predict(noisy);
                const denoisedAvg = this.modelAvgPool.predict(noisy);
                
                const canvas = document.createElement('canvas');
                canvas.className = 'comparison-canvas';
                
                await this.renderTripleComparison(original, noisy, denoisedMax, denoisedAvg, canvas);
                
                const label = document.createElement('div');
                label.className = 'comparison-label';
                label.textContent = `Digit ${i === 0 ? '2' : i === 1 ? '3' : '5'} | Left: MaxPool, Right: AvgPool`;
                
                const container = document.createElement('div');
                container.className = 'comparison-item';
                container.appendChild(canvas);
                container.appendChild(label);
                comparisonContainer.appendChild(container);
                
                // Clean up
                original.dispose();
                noisy.dispose();
                denoisedMax.dispose();
                denoisedAvg.dispose();
            }
            
        } catch (error) {
            this.showError(`Comparison failed: ${error.message}`);
        }
    }

    async onSaveDownload() {
        if (!this.modelMaxPool && !this.modelAvgPool) {
            this.showError('No models to save');
            return;
        }

        try {
            if (this.modelMaxPool) {
                await this.modelMaxPool.save('downloads://mnist-denoiser-maxpool');
            }
            if (this.modelAvgPool) {
                await this.modelAvgPool.save('downloads://mnist-denoiser-avgpool');
            }
            this.showStatus('Models saved successfully!');
        } catch (error) {
            this.showError(`Failed to save models: ${error.message}`);
        }
    }

    async onLoadFromFiles() {
        const jsonFile = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];
        
        if (!jsonFile || !weightsFile) {
            this.showError('Please select both model.json and weights.bin files');
            return;
        }

        try {
            this.showStatus('Loading model...');
            
            // Determine which model is being loaded based on filename
            const model = await tf.loadLayersModel(
                tf.io.browserFiles([jsonFile, weightsFile])
            );
            
            if (jsonFile.name.includes('maxpool')) {
                if (this.modelMaxPool) this.modelMaxPool.dispose();
                this.modelMaxPool = model;
                this.showStatus('MaxPool model loaded successfully!');
            } else if (jsonFile.name.includes('avgpool')) {
                if (this.modelAvgPool) this.modelAvgPool.dispose();
                this.modelAvgPool = model;
                this.showStatus('AvgPool model loaded successfully!');
            } else {
                this.showStatus('Model loaded, but could not determine pooling type');
            }
            
            this.updateModelInfo();
            
        } catch (error) {
            this.showError(`Failed to load model: ${error.message}`);
        }
    }

    onReset() {
        if (this.modelMaxPool) {
            this.modelMaxPool.dispose();
            this.modelMaxPool = null;
        }
        if (this.modelAvgPool) {
            this.modelAvgPool.dispose();
            this.modelAvgPool = null;
        }
        
        this.dataLoader.dispose();
        this.trainData = null;
        this.testData = null;
        
        this.updateDataStatus(0, 0);
        this.updateModelInfo();
        this.clearPreviews();
        this.showStatus('Reset completed');
    }

    toggleVisor() {
        tfvis.visor().toggle();
    }

    createDenoisingAutoencoder(poolingType = 'maxpool') {
        const model = tf.sequential();
        
        // Encoder
        model.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same',
            inputShape: [28, 28, 1]
        }));
        
        if (poolingType === 'maxpool') {
            model.add(tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' }));
        } else {
            model.add(tf.layers.averagePooling2d({ poolSize: 2, padding: 'same' }));
        }
        
        model.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        
        if (poolingType === 'maxpool') {
            model.add(tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' }));
        } else {
            model.add(tf.layers.averagePooling2d({ poolSize: 2, padding: 'same' }));
        }
        
        // Bottleneck
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        
        // Decoder
        model.add(tf.layers.upSampling2d({ size: [2, 2] }));
        model.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        
        model.add(tf.layers.upSampling2d({ size: [2, 2] }));
        model.add(tf.layers.conv2d({
            filters: 1,
            kernelSize: 3,
            activation: 'sigmoid',
            padding: 'same'
        }));
        
        model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError'
        });
        
        return model;
    }

    findDigitIndices(ys, digits) {
        const indices = [];
        const ysArray = ys.argMax(-1).arraySync();
        
        for (const digit of digits) {
            const idx = ysArray.findIndex((val, index) => 
                val === digit && !indices.includes(index)
            );
            if (idx !== -1) indices.push(idx);
        }
        
        return indices;
    }

    async renderDenoisingComparison(originals, noisy, denoisedMax, denoisedAvg) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '<h3>Denoising Results (Original | Noisy | Denoised)</h3>';
        
        const originalArray = originals.arraySync();
        
        for (let i = 0; i < originalArray.length; i++) {
            const item = document.createElement('div');
            item.className = 'preview-item';
            
            const canvas = document.createElement('canvas');
            
            const originalTensor = tf.tensor(originalArray[i]);
            const noisyTensor = tf.gather(noisy, [i]).squeeze();
            
            // Use available denoised model
            let denoisedTensor;
            if (denoisedMax && denoisedAvg) {
                // Show both? For simplicity, use MaxPool for now
                denoisedTensor = tf.gather(denoisedMax, [i]).squeeze();
            } else if (denoisedMax) {
                denoisedTensor = tf.gather(denoisedMax, [i]).squeeze();
            } else {
                denoisedTensor = tf.gather(denoisedAvg, [i]).squeeze();
            }
            
            this.dataLoader.drawComparison(
                originalTensor, 
                noisyTensor, 
                denoisedTensor, 
                canvas, 
                2
            );
            
            item.appendChild(canvas);
            container.appendChild(item);
            
            // Clean up
            originalTensor.dispose();
            noisyTensor.dispose();
            denoisedTensor.dispose();
        }
    }

    async renderTripleComparison(original, noisy, denoisedMax, denoisedAvg, canvas) {
        return tf.tidy(() => {
            const ctx = canvas.getContext('2d');
            canvas.width = 28 * 5 * 2; // 5 images * scale 2
            canvas.height = 28 * 2;
            
            // Create temp canvases
            const tempCanvas1 = document.createElement('canvas');
            const tempCanvas2 = document.createElement('canvas');
            const tempCanvas3 = document.createElement('canvas');
            const tempCanvas4 = document.createElement('canvas');
            
            this.dataLoader.draw28x28ToCanvas(original.squeeze(), tempCanvas1, 1);
            this.dataLoader.draw28x28ToCanvas(noisy.squeeze(), tempCanvas2, 1);
            this.dataLoader.draw28x28ToCanvas(denoisedMax.squeeze(), tempCanvas3, 1);
            this.dataLoader.draw28x28ToCanvas(denoisedAvg.squeeze(), tempCanvas4, 1);
            
            ctx.imageSmoothingEnabled = false;
            
            // Row 1: Original, Noisy
            ctx.drawImage(tempCanvas1, 0, 0, 56, 56);
            ctx.drawImage(tempCanvas2, 56, 0, 56, 56);
            
            // Row 2: MaxPool, AvgPool
            ctx.drawImage(tempCanvas3, 0, 56, 56, 56);
            ctx.drawImage(tempCanvas4, 56, 56, 56, 56);
            
            // Add labels
            ctx.font = '16px Arial';
            ctx.fillStyle = 'white';
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 2;
            
            ctx.strokeText('Original', 5, 20);
            ctx.fillText('Original', 5, 20);
            ctx.strokeText('Noisy', 61, 20);
            ctx.fillText('Noisy', 61, 20);
            ctx.strokeText('MaxPool', 5, 76);
            ctx.fillText('MaxPool', 5, 76);
            ctx.strokeText('AvgPool', 61, 76);
            ctx.fillText('AvgPool', 61, 76);
        });
    }

    clearPreviews() {
        document.getElementById('previewContainer').innerHTML = '';
        document.getElementById('comparisonContainer').innerHTML = '';
    }

    updateDataStatus(trainCount, testCount) {
        const statusEl = document.getElementById('dataStatus');
        statusEl.innerHTML = `
            <h3>Data Status</h3>
            <p>Train samples: ${trainCount}</p>
            <p>Test samples: ${testCount}</p>
        `;
    }

    updateModelInfo() {
        const infoEl = document.getElementById('modelInfo');
        
        if (!this.modelMaxPool && !this.modelAvgPool) {
            infoEl.innerHTML = '<h3>Model Info</h3><p>No models loaded</p>';
            return;
        }
        
        let info = '<h3>Models Info</h3>';
        
        if (this.modelMaxPool) {
            info += '<p><b>MaxPool Autoencoder:</b> Loaded</p>';
        }
        if (this.modelAvgPool) {
            info += '<p><b>AvgPool Autoencoder:</b> Loaded</p>';
        }
        
        infoEl.innerHTML = info;
    }

    showStatus(message) {
        const logs = document.getElementById('trainingLogs');
        const entry = document.createElement('div');
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logs.appendChild(entry);
        logs.scrollTop = logs.scrollHeight;
    }

    showError(message) {
        this.showStatus(`ERROR: ${message}`);
        console.error(message);
    }
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    new MNISTApp();
});
