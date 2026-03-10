class MNISTApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();
        this.classifierModel = null;
        this.autoencoderMax = null;   // Autoencoder with Max Pooling
        this.autoencoderAvg = null;   // Autoencoder with Average Pooling
        this.isTraining = false;
        this.trainData = null;
        this.testData = null;
        this.noiseStddev = 0.3;

        this.initializeUI();
    }

    initializeUI() {
        document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
        document.getElementById('trainBtn').addEventListener('click', () => this.onTrain());
        document.getElementById('trainAutoencoderBtn').addEventListener('click', () => this.onTrainAutoencoder());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.onEvaluate());
        document.getElementById('testFiveBtn').addEventListener('click', () => this.onTestFive());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.onSaveDownload());
        document.getElementById('loadModelBtn').addEventListener('click', () => this.onLoadFromFiles());
        document.getElementById('resetBtn').addEventListener('click', () => this.onReset());
        document.getElementById('toggleVisorBtn').addEventListener('click', () => this.toggleVisor());

        const noiseSlider = document.getElementById('noiseSlider');
        const noiseVal = document.getElementById('noiseVal');
        noiseSlider.addEventListener('input', () => {
            this.noiseStddev = parseFloat(noiseSlider.value);
            noiseVal.textContent = this.noiseStddev.toFixed(2);
        });
        noiseVal.textContent = this.noiseStddev.toFixed(2);
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

    // ─── CNN Classifier ────────────────────────────────────────────────────────

    async onTrain() {
        if (!this.trainData) { this.showError('Please load training data first'); return; }
        if (this.isTraining) { this.showError('Training already in progress'); return; }

        try {
            this.isTraining = true;
            this.showStatus('Starting classifier training...');

            const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
                this.trainData.xs, this.trainData.ys, 0.1
            );

            if (!this.classifierModel) {
                this.classifierModel = this.createClassifierModel();
                this.updateModelInfo();
            }

            const startTime = Date.now();
            const history = await this.classifierModel.fit(trainXs, trainYs, {
                epochs: 5,
                batchSize: 128,
                validationData: [valXs, valYs],
                shuffle: true,
                callbacks: tfvis.show.fitCallbacks(
                    { name: 'Classifier Training' },
                    ['loss', 'val_loss', 'acc', 'val_acc'],
                    { callbacks: ['onEpochEnd'] }
                )
            });

            const duration = (Date.now() - startTime) / 1000;
            const bestValAcc = Math.max(...history.history.val_acc);
            this.showStatus(`Classifier training done in ${duration.toFixed(1)}s. Best val_acc: ${bestValAcc.toFixed(4)}`);

            trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose();
        } catch (error) {
            this.showError(`Training failed: ${error.message}`);
        } finally {
            this.isTraining = false;
        }
    }

    // ─── CNN Autoencoder Training ───────────────────────────────────────────────

    async onTrainAutoencoder() {
        if (!this.trainData) { this.showError('Please load training data first'); return; }
        if (this.isTraining) { this.showError('Training already in progress'); return; }

        try {
            this.isTraining = true;
            this.showStatus(`Training autoencoders with noise stddev=${this.noiseStddev}...`);

            // Prepare noisy training data (input) vs clean (target)
            const cleanXs = this.trainData.xs;
            const noisyXs = this.dataLoader.addNoise(cleanXs, this.noiseStddev);

            // Validation split from clean
            const numVal = Math.floor(cleanXs.shape[0] * 0.1);
            const numTrain = cleanXs.shape[0] - numVal;

            const trainClean = cleanXs.slice([0, 0, 0, 0], [numTrain, 28, 28, 1]);
            const trainNoisy = noisyXs.slice([0, 0, 0, 0], [numTrain, 28, 28, 1]);
            const valClean   = cleanXs.slice([numTrain, 0, 0, 0], [numVal, 28, 28, 1]);
            const valNoisy   = noisyXs.slice([numTrain, 0, 0, 0], [numVal, 28, 28, 1]);

            // Build autoencoders
            if (this.autoencoderMax) this.autoencoderMax.dispose();
            if (this.autoencoderAvg) this.autoencoderAvg.dispose();

            this.autoencoderMax = this.createAutoencoder('max');
            this.autoencoderAvg = this.createAutoencoder('avg');

            // Train Max Pooling Autoencoder
            this.showStatus('Training Max Pooling Autoencoder...');
            const t1 = Date.now();
            await this.autoencoderMax.fit(trainNoisy, trainClean, {
                epochs: 10,
                batchSize: 128,
                validationData: [valNoisy, valClean],
                shuffle: true,
                callbacks: tfvis.show.fitCallbacks(
                    { name: 'Autoencoder – Max Pooling', tab: 'Autoencoders' },
                    ['loss', 'val_loss'],
                    { callbacks: ['onEpochEnd'] }
                )
            });
            this.showStatus(`Max Pooling AE done in ${((Date.now() - t1) / 1000).toFixed(1)}s`);

            // Train Average Pooling Autoencoder
            this.showStatus('Training Avg Pooling Autoencoder...');
            const t2 = Date.now();
            await this.autoencoderAvg.fit(trainNoisy, trainClean, {
                epochs: 10,
                batchSize: 128,
                validationData: [valNoisy, valClean],
                shuffle: true,
                callbacks: tfvis.show.fitCallbacks(
                    { name: 'Autoencoder – Avg Pooling', tab: 'Autoencoders' },
                    ['loss', 'val_loss'],
                    { callbacks: ['onEpochEnd'] }
                )
            });
            this.showStatus(`Avg Pooling AE done in ${((Date.now() - t2) / 1000).toFixed(1)}s`);

            this.updateModelInfo();

            // Clean up
            noisyXs.dispose();
            trainClean.dispose(); trainNoisy.dispose();
            valClean.dispose();   valNoisy.dispose();

            this.showStatus('Both autoencoders trained! Click "Test 5 Random" to see denoising comparison.');
        } catch (error) {
            this.showError(`Autoencoder training failed: ${error.message}`);
        } finally {
            this.isTraining = false;
        }
    }

    // ─── Evaluate Classifier ───────────────────────────────────────────────────

    async onEvaluate() {
        if (!this.classifierModel) { this.showError('No classifier model. Please train first.'); return; }
        if (!this.testData) { this.showError('No test data available'); return; }

        try {
            this.showStatus('Evaluating classifier...');

            const testXs = this.testData.xs;
            const testYs = this.testData.ys;

            const predictions = this.classifierModel.predict(testXs);
            const predictedLabels = predictions.argMax(-1);
            const trueLabels = testYs.argMax(-1);

            const accuracy = await this.calculateAccuracy(predictedLabels, trueLabels);
            const confusionMatrix = await this.createConfusionMatrix(predictedLabels, trueLabels);

            const metricsContainer = { name: 'Test Metrics', tab: 'Evaluation' };

            tfvis.show.modelSummary(metricsContainer, this.classifierModel);
            tfvis.show.perClassAccuracy(
                metricsContainer,
                { values: this.calculatePerClassAccuracy(confusionMatrix) },
                ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            );
            tfvis.render.confusionMatrix(metricsContainer, {
                values: confusionMatrix,
                tickLabels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            });

            this.showStatus(`Test accuracy: ${(accuracy * 100).toFixed(2)}%`);

            predictions.dispose();
            predictedLabels.dispose();
            trueLabels.dispose();
        } catch (error) {
            this.showError(`Evaluation failed: ${error.message}`);
        }
    }

    // ─── Test 5 Random – Denoising Comparison ─────────────────────────────────

    async onTestFive() {
        if (!this.testData) { this.showError('Please load test data first'); return; }
        if (!this.autoencoderMax || !this.autoencoderAvg) {
            this.showError('Please train autoencoders first (Train Autoencoder button)');
            return;
        }

        try {
            const { batchXs, batchYs, indices } = this.dataLoader.getRandomTestBatch(
                this.testData.xs, this.testData.ys, 5
            );

            // Add noise to test batch
            const noisyBatch = this.dataLoader.addNoise(batchXs, this.noiseStddev);

            // Denoise with both models
            const denoisedMax = this.autoencoderMax.predict(noisyBatch);
            const denoisedAvg = this.autoencoderAvg.predict(noisyBatch);

            const trueLabels = batchYs.argMax(-1);
            const trueArray  = await trueLabels.array();

            await this.renderDenoisingPreview(noisyBatch, denoisedMax, denoisedAvg, trueArray);

            // Clean up
            batchXs.dispose(); batchYs.dispose(); noisyBatch.dispose();
            denoisedMax.dispose(); denoisedAvg.dispose(); trueLabels.dispose();
        } catch (error) {
            this.showError(`Test preview failed: ${error.message}`);
        }
    }

    async renderDenoisingPreview(noisy, denoisedMax, denoisedAvg, trueLabels) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '';

        const noisyArr     = await noisy.array();
        const maxArr       = await denoisedMax.array();
        const avgArr       = await denoisedAvg.array();

        // Row headers
        const makeLabel = (text, color) => {
            const el = document.createElement('div');
            el.className = 'row-label';
            el.style.color = color;
            el.textContent = text;
            return el;
        };

        const rows = [
            { label: 'Noisy Input',           color: '#555',  data: noisyArr },
            { label: 'Max Pooling Denoised',   color: '#1565C0', data: maxArr },
            { label: 'Avg Pooling Denoised',   color: '#1565C0', data: avgArr },
        ];

        rows.forEach(({ label, color, data }) => {
            const rowWrapper = document.createElement('div');
            rowWrapper.className = 'preview-row-block';

            const rowLabel = makeLabel(label, color);
            rowWrapper.appendChild(rowLabel);

            const imagesRow = document.createElement('div');
            imagesRow.className = 'preview-row';

            data.forEach((image, i) => {
                const item = document.createElement('div');
                item.className = 'preview-item';

                const canvas = document.createElement('canvas');
                const caption = document.createElement('div');
                caption.className = 'img-caption';
                caption.textContent = `Label: ${trueLabels[i]}`;

                this.dataLoader.draw28x28ToCanvas(tf.tensor(image), canvas, 4);

                item.appendChild(canvas);
                item.appendChild(caption);
                imagesRow.appendChild(item);
            });

            rowWrapper.appendChild(imagesRow);
            container.appendChild(rowWrapper);
        });

        // Comparison header
        const compHeader = document.createElement('div');
        compHeader.className = 'comparison-header';
        compHeader.textContent = 'Compare Max Pooling vs Average Pooling';
        container.insertBefore(compHeader, container.children[1]);
    }

    // ─── Save / Load ───────────────────────────────────────────────────────────

    async onSaveDownload() {
        if (!this.autoencoderMax && !this.classifierModel) {
            this.showError('No model to save'); return;
        }
        try {
            if (this.autoencoderMax) {
                await this.autoencoderMax.save('downloads://mnist-autoencoder-max');
                this.showStatus('Max Pooling Autoencoder saved!');
            }
            if (this.autoencoderAvg) {
                await this.autoencoderAvg.save('downloads://mnist-autoencoder-avg');
                this.showStatus('Avg Pooling Autoencoder saved!');
            }
            if (this.classifierModel) {
                await this.classifierModel.save('downloads://mnist-classifier');
                this.showStatus('Classifier saved!');
            }
        } catch (error) {
            this.showError(`Failed to save model: ${error.message}`);
        }
    }

    async onLoadFromFiles() {
        const jsonFile    = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];

        if (!jsonFile || !weightsFile) {
            this.showError('Please select both model.json and weights.bin files');
            return;
        }

        try {
            this.showStatus('Loading model...');
            if (this.autoencoderMax) this.autoencoderMax.dispose();
            this.autoencoderMax = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
            this.updateModelInfo();
            this.showStatus('Model loaded as Max Pooling Autoencoder successfully!');
        } catch (error) {
            this.showError(`Failed to load model: ${error.message}`);
        }
    }

    onReset() {
        if (this.classifierModel)  { this.classifierModel.dispose();  this.classifierModel  = null; }
        if (this.autoencoderMax)   { this.autoencoderMax.dispose();   this.autoencoderMax   = null; }
        if (this.autoencoderAvg)   { this.autoencoderAvg.dispose();   this.autoencoderAvg   = null; }

        this.dataLoader.dispose();
        this.trainData = null;
        this.testData  = null;

        this.updateDataStatus(0, 0);
        this.updateModelInfo();
        this.clearPreview();
        this.showStatus('Reset completed');
    }

    toggleVisor() { tfvis.visor().toggle(); }

    // ─── Model Architectures ──────────────────────────────────────────────────

    createClassifierModel() {
        const model = tf.sequential();
        model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [28, 28, 1] }));
        model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        model.add(tf.layers.dropout({ rate: 0.25 }));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
        model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
        return model;
    }

    /**
     * Create CNN Autoencoder for image denoising.
     * @param {'max'|'avg'} poolingType - Type of pooling in the encoder
     */
    createAutoencoder(poolingType = 'max') {
        const model = tf.sequential();

        // ── Encoder ──
        model.add(tf.layers.conv2d({
            filters: 32, kernelSize: 3, activation: 'relu',
            padding: 'same', inputShape: [28, 28, 1]
        }));

        if (poolingType === 'max') {
            model.add(tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' })); // [14,14,32]
        } else {
            model.add(tf.layers.averagePooling2d({ poolSize: 2, padding: 'same' })); // [14,14,32]
        }

        model.add(tf.layers.conv2d({
            filters: 64, kernelSize: 3, activation: 'relu', padding: 'same'
        }));

        if (poolingType === 'max') {
            model.add(tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' })); // [7,7,64]
        } else {
            model.add(tf.layers.averagePooling2d({ poolSize: 2, padding: 'same' })); // [7,7,64]
        }

        // ── Decoder ──
        model.add(tf.layers.conv2dTranspose({
            filters: 64, kernelSize: 3, strides: 2,
            activation: 'relu', padding: 'same'
        })); // [14,14,64]

        model.add(tf.layers.conv2dTranspose({
            filters: 32, kernelSize: 3, strides: 2,
            activation: 'relu', padding: 'same'
        })); // [28,28,32]

        model.add(tf.layers.conv2d({
            filters: 1, kernelSize: 3, activation: 'sigmoid', padding: 'same'
        })); // [28,28,1]

        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
        return model;
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    async calculateAccuracy(predicted, trueLabels) {
        const equals   = predicted.equal(trueLabels);
        const accuracy = equals.mean();
        const result   = await accuracy.data();
        equals.dispose(); accuracy.dispose();
        return result[0];
    }

    async createConfusionMatrix(predicted, trueLabels) {
        const predArray = await predicted.array();
        const trueArray = await trueLabels.array();
        const matrix    = Array(10).fill(null).map(() => Array(10).fill(0));
        for (let i = 0; i < predArray.length; i++) matrix[trueArray[i]][predArray[i]]++;
        return matrix;
    }

    calculatePerClassAccuracy(confusionMatrix) {
        return confusionMatrix.map((row, i) => {
            const correct = row[i];
            const total   = row.reduce((s, v) => s + v, 0);
            return total > 0 ? correct / total : 0;
        });
    }

    clearPreview() { document.getElementById('previewContainer').innerHTML = ''; }

    updateDataStatus(trainCount, testCount) {
        document.getElementById('dataStatus').innerHTML = `
            <h3>Data Status</h3>
            <p>Train samples: ${trainCount}</p>
            <p>Test samples: ${testCount}</p>
        `;
    }

    updateModelInfo() {
        const infoEl = document.getElementById('modelInfo');
        const count = m => {
            let p = 0;
            m.layers.forEach(l => l.getWeights().forEach(w => p += w.size));
            return p;
        };
        let html = '<h3>Model Info</h3>';
        if (this.classifierModel) html += `<p>Classifier – params: ${count(this.classifierModel).toLocaleString()}</p>`;
        if (this.autoencoderMax)  html += `<p>AE Max Pooling – params: ${count(this.autoencoderMax).toLocaleString()}</p>`;
        if (this.autoencoderAvg)  html += `<p>AE Avg Pooling – params: ${count(this.autoencoderAvg).toLocaleString()}</p>`;
        if (!this.classifierModel && !this.autoencoderMax && !this.autoencoderAvg) html += '<p>No model loaded</p>';
        infoEl.innerHTML = html;
    }

    showStatus(message) {
        const logs  = document.getElementById('trainingLogs');
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

document.addEventListener('DOMContentLoaded', () => { new MNISTApp(); });
