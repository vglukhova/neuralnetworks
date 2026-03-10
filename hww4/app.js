class MNISTApp {
    constructor() {
        this.loader       = new MNISTDataLoader();
        this.trainData    = null;
        this.testData     = null;
        this.classifier   = null;
        this.aeMax        = null;   // autoencoder with max pooling
        this.aeAvg        = null;   // autoencoder with avg pooling
        this.isTraining   = false;
        this.noiseStddev  = 0.3;
        this.initUI();
    }

    initUI() {
        this.bind('loadDataBtn',          () => this.onLoadData());
        this.bind('trainClassifierBtn',   () => this.onTrainClassifier());
        this.bind('evaluateBtn',          () => this.onEvaluate());
        this.bind('trainAutoencoderBtn',  () => this.onTrainAutoencoder());
        this.bind('testFiveBtn',          () => this.onTestFive());
        this.bind('saveBtn',              () => this.onSave());
        this.bind('loadModelBtn',         () => this.onLoadModel());
        this.bind('resetBtn',             () => this.onReset());
        this.bind('toggleVisorBtn',       () => tfvis.visor().toggle());

        const slider = document.getElementById('noiseSlider');
        const label  = document.getElementById('noiseVal');
        slider.addEventListener('input', () => {
            this.noiseStddev = parseFloat(slider.value);
            label.textContent = this.noiseStddev.toFixed(2);
        });
    }

    bind(id, fn) {
        document.getElementById(id).addEventListener('click', fn);
    }

    // ── Load Data ─────────────────────────────────────────────────────────────

    async onLoadData() {
        const trainFile = document.getElementById('trainFile').files[0];
        const testFile  = document.getElementById('testFile').files[0];
        if (!trainFile || !testFile) {
            return this.log('ERROR: Select both Train and Test CSV files.');
        }
        try {
            this.log('Loading train CSV…');
            this.trainData = await this.loader.loadTrainFromFiles(trainFile);
            this.log('Loading test CSV…');
            this.testData  = await this.loader.loadTestFromFiles(testFile);
            this.log('Data loaded — train: ' + this.trainData.count + ', test: ' + this.testData.count);
            document.getElementById('dataStatus').innerHTML =
                '<b>Train:</b> ' + this.trainData.count + ' samples<br>' +
                '<b>Test:</b> '  + this.testData.count  + ' samples';
        } catch (err) {
            this.log('ERROR loading data: ' + err.message);
        }
    }

    // ── Classifier ────────────────────────────────────────────────────────────

    async onTrainClassifier() {
        if (!this.trainData)  return this.log('ERROR: Load data first.');
        if (this.isTraining)  return this.log('ERROR: Already training.');
        this.isTraining = true;
        try {
            if (this.classifier) { this.classifier.dispose(); this.classifier = null; }
            this.classifier = this.buildClassifier();
            this.log('Training classifier…');

            const split = this.loader.splitTrainVal(this.trainData.xs, this.trainData.ys, 0.1);
            const t0 = Date.now();
            const hist = await this.classifier.fit(split.trainXs, split.trainYs, {
                epochs: 5, batchSize: 128,
                validationData: [split.valXs, split.valYs],
                shuffle: true,
                callbacks: tfvis.show.fitCallbacks(
                    { name: 'Classifier', tab: 'Training' },
                    ['loss', 'val_loss', 'acc', 'val_acc'],
                    { callbacks: ['onEpochEnd'] }
                )
            });
            split.trainXs.dispose(); split.trainYs.dispose();
            split.valXs.dispose();   split.valYs.dispose();

            const best = Math.max.apply(null, hist.history.val_acc);
            this.log('Classifier done in ' + ((Date.now()-t0)/1000).toFixed(1) + 's  |  best val_acc: ' + best.toFixed(4));
            this.updateModelInfo();
        } catch (err) {
            this.log('ERROR: ' + err.message);
        } finally {
            this.isTraining = false;
        }
    }

    async onEvaluate() {
        if (!this.classifier) return this.log('ERROR: Train or load a classifier first.');
        if (!this.testData)   return this.log('ERROR: Load test data first.');
        try {
            this.log('Evaluating…');
            const preds  = this.classifier.predict(this.testData.xs);
            const predLb = preds.argMax(-1);
            const trueLb = this.testData.ys.argMax(-1);

            const eq  = predLb.equal(trueLb);
            const acc = (await eq.mean().data())[0];
            this.log('Test accuracy: ' + (acc * 100).toFixed(2) + '%');

            // Confusion matrix
            const predArr = await predLb.array();
            const trueArr = await trueLb.array();
            const cm = Array.from({length:10}, () => new Array(10).fill(0));
            for (let i = 0; i < predArr.length; i++) cm[trueArr[i]][predArr[i]]++;
            tfvis.render.confusionMatrix(
                { name: 'Confusion Matrix', tab: 'Evaluation' },
                { values: cm, tickLabels: ['0','1','2','3','4','5','6','7','8','9'] }
            );

            preds.dispose(); predLb.dispose(); trueLb.dispose(); eq.dispose();
        } catch (err) {
            this.log('ERROR: ' + err.message);
        }
    }

    // ── Autoencoders ──────────────────────────────────────────────────────────

    async onTrainAutoencoder() {
        if (!this.trainData) return this.log('ERROR: Load data first.');
        if (this.isTraining) return this.log('ERROR: Already training.');
        this.isTraining = true;
        try {
            if (this.aeMax) { this.aeMax.dispose(); this.aeMax = null; }
            if (this.aeAvg) { this.aeAvg.dispose(); this.aeAvg = null; }

            this.aeMax = this.buildAutoencoder('max');
            this.aeAvg = this.buildAutoencoder('avg');

            const clean = this.trainData.xs;
            const noisy = this.loader.addNoise(clean, this.noiseStddev);

            const total  = clean.shape[0];
            const numVal = Math.floor(total * 0.1);
            const numTrn = total - numVal;

            const trnClean = clean.slice([0,      0,0,0], [numTrn, 28,28,1]);
            const trnNoisy = noisy.slice([0,      0,0,0], [numTrn, 28,28,1]);
            const valClean = clean.slice([numTrn, 0,0,0], [numVal, 28,28,1]);
            const valNoisy = noisy.slice([numTrn, 0,0,0], [numVal, 28,28,1]);

            const fitOpts = (name) => ({
                epochs: 10, batchSize: 128,
                validationData: [valNoisy, valClean],
                shuffle: true,
                callbacks: tfvis.show.fitCallbacks(
                    { name: name, tab: 'Autoencoders' },
                    ['loss', 'val_loss'],
                    { callbacks: ['onEpochEnd'] }
                )
            });

            this.log('Training Max-Pooling Autoencoder…');
            const t1 = Date.now();
            await this.aeMax.fit(trnNoisy, trnClean, fitOpts('AE Max Pooling'));
            this.log('Max-AE done in ' + ((Date.now()-t1)/1000).toFixed(1) + 's');

            this.log('Training Avg-Pooling Autoencoder…');
            const t2 = Date.now();
            await this.aeAvg.fit(trnNoisy, trnClean, fitOpts('AE Avg Pooling'));
            this.log('Avg-AE done in ' + ((Date.now()-t2)/1000).toFixed(1) + 's');

            noisy.dispose();
            trnClean.dispose(); trnNoisy.dispose();
            valClean.dispose(); valNoisy.dispose();

            this.updateModelInfo();
            this.log('Both autoencoders trained. Click "Test 5 Random" to see results.');
        } catch (err) {
            this.log('ERROR: ' + err.message);
        } finally {
            this.isTraining = false;
        }
    }

    async onTestFive() {
        if (!this.testData)             return this.log('ERROR: Load test data first.');
        if (!this.aeMax || !this.aeAvg) return this.log('ERROR: Train autoencoders first.');
        try {
            const { batchXs, batchYs } = this.loader.getRandomTestBatch(
                this.testData.xs, this.testData.ys, 5
            );
            const noisy       = this.loader.addNoise(batchXs, this.noiseStddev);
            const denoisedMax = this.aeMax.predict(noisy);
            const denoisedAvg = this.aeAvg.predict(noisy);

            // Get labels BEFORE disposing tensors
            const lblTensor = batchYs.argMax(-1);
            const trueArr   = await lblTensor.array();
            lblTensor.dispose();

            // Convert to plain JS arrays (values in [0,1])
            const noisyData = await noisy.array();
            const maxData   = await denoisedMax.array();
            const avgData   = await denoisedAvg.array();

            // Render using plain arrays — no tensors involved
            this.renderPreview(noisyData, maxData, avgData, trueArr);

            batchXs.dispose(); batchYs.dispose();
            noisy.dispose(); denoisedMax.dispose(); denoisedAvg.dispose();
        } catch (err) {
            this.log('ERROR: ' + err.message);
        }
    }

    renderPreview(noisyData, maxData, avgData, trueLabels) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '';

        const rows = [
            { label: 'Noisy Input',            color: '#555',    data: noisyData },
            { label: 'Max Pooling – Denoised', color: '#1565C0', data: maxData   },
            { label: 'Avg Pooling – Denoised', color: '#1565C0', data: avgData   },
        ];

        // Comparison header between noisy and denoised rows
        rows.forEach((row, ri) => {
            if (ri === 1) {
                const hdr = document.createElement('div');
                hdr.className = 'cmp-header';
                hdr.textContent = 'Compare Max Pooling vs Average Pooling';
                container.appendChild(hdr);
            }
            const block = document.createElement('div');
            block.className = 'preview-block';

            const lbl = document.createElement('div');
            lbl.className = 'row-label';
            lbl.style.color = row.color;
            lbl.textContent = row.label;
            block.appendChild(lbl);

            const rowEl = document.createElement('div');
            rowEl.className = 'preview-row';

            row.data.forEach((img, i) => {
                const item = document.createElement('div');
                item.className = 'preview-item';

                const canvas = document.createElement('canvas');
                // img is a plain JS array [28][28][1] — drawToCanvas handles flattening
                this.loader.drawToCanvas(img, canvas, 4);

                const cap = document.createElement('div');
                cap.className = 'caption';
                cap.textContent = 'Label: ' + trueLabels[i];

                item.appendChild(canvas);
                item.appendChild(cap);
                rowEl.appendChild(item);
            });

            block.appendChild(rowEl);
            container.appendChild(block);
        });
    }

    // ── Save / Load ───────────────────────────────────────────────────────────

    async onSave() {
        if (!this.aeMax && !this.aeAvg && !this.classifier) {
            return this.log('ERROR: No model to save.');
        }
        try {
            if (this.classifier) {
                await this.classifier.save('downloads://mnist-classifier');
                this.log('Classifier saved.');
            }
            if (this.aeMax) {
                await this.aeMax.save('downloads://mnist-ae-max');
                this.log('Max-Pooling AE saved.');
            }
            if (this.aeAvg) {
                await this.aeAvg.save('downloads://mnist-ae-avg');
                this.log('Avg-Pooling AE saved.');
            }
        } catch (err) {
            this.log('ERROR saving: ' + err.message);
        }
    }

    async onLoadModel() {
        const jsonFile    = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];
        if (!jsonFile || !weightsFile) {
            return this.log('ERROR: Select .json and .bin files.');
        }
        try {
            this.log('Loading model…');
            if (this.aeMax) { this.aeMax.dispose(); }
            this.aeMax = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
            this.updateModelInfo();
            this.log('Model loaded as Max-Pooling AE. You can now click "Test 5 Random".');
        } catch (err) {
            this.log('ERROR loading model: ' + err.message);
        }
    }

    onReset() {
        if (this.classifier) { this.classifier.dispose(); this.classifier = null; }
        if (this.aeMax)      { this.aeMax.dispose();      this.aeMax      = null; }
        if (this.aeAvg)      { this.aeAvg.dispose();      this.aeAvg      = null; }
        this.loader.dispose();
        this.trainData = null;
        this.testData  = null;
        document.getElementById('dataStatus').textContent = 'No data loaded';
        document.getElementById('modelInfo').textContent  = 'No model';
        document.getElementById('previewContainer').innerHTML = '';
        this.log('Reset done.');
    }

    // ── Model builders ────────────────────────────────────────────────────────

    buildClassifier() {
        const m = tf.sequential();
        m.add(tf.layers.conv2d({ inputShape:[28,28,1], filters:32, kernelSize:3, activation:'relu', padding:'same' }));
        m.add(tf.layers.conv2d({ filters:64, kernelSize:3, activation:'relu', padding:'same' }));
        m.add(tf.layers.maxPooling2d({ poolSize:2 }));
        m.add(tf.layers.dropout({ rate:0.25 }));
        m.add(tf.layers.flatten());
        m.add(tf.layers.dense({ units:128, activation:'relu' }));
        m.add(tf.layers.dropout({ rate:0.5 }));
        m.add(tf.layers.dense({ units:10, activation:'softmax' }));
        m.compile({ optimizer:'adam', loss:'categoricalCrossentropy', metrics:['accuracy'] });
        return m;
    }

    buildAutoencoder(poolType) {
        const m = tf.sequential();

        // Encoder
        m.add(tf.layers.conv2d({ inputShape:[28,28,1], filters:32, kernelSize:3, activation:'relu', padding:'same' }));
        if (poolType === 'max') {
            m.add(tf.layers.maxPooling2d({ poolSize:2, padding:'same' }));   // -> [14,14,32]
        } else {
            m.add(tf.layers.averagePooling2d({ poolSize:2, padding:'same' }));
        }
        m.add(tf.layers.conv2d({ filters:64, kernelSize:3, activation:'relu', padding:'same' }));
        if (poolType === 'max') {
            m.add(tf.layers.maxPooling2d({ poolSize:2, padding:'same' }));   // -> [7,7,64]
        } else {
            m.add(tf.layers.averagePooling2d({ poolSize:2, padding:'same' }));
        }

        // Decoder
        m.add(tf.layers.conv2dTranspose({ filters:64, kernelSize:3, strides:2, activation:'relu',    padding:'same' })); // -> [14,14,64]
        m.add(tf.layers.conv2dTranspose({ filters:32, kernelSize:3, strides:2, activation:'relu',    padding:'same' })); // -> [28,28,32]
        m.add(tf.layers.conv2d({          filters:1,  kernelSize:3,            activation:'sigmoid', padding:'same' })); // -> [28,28,1]

        m.compile({ optimizer:'adam', loss:'meanSquaredError' });
        return m;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    updateModelInfo() {
        const countParams = (m) => {
            let p = 0;
            m.layers.forEach(l => l.getWeights().forEach(w => { p += w.size; }));
            return p;
        };
        let html = '';
        if (this.classifier) html += 'Classifier: ' + countParams(this.classifier).toLocaleString() + ' params<br>';
        if (this.aeMax)      html += 'AE Max-Pool: ' + countParams(this.aeMax).toLocaleString() + ' params<br>';
        if (this.aeAvg)      html += 'AE Avg-Pool: ' + countParams(this.aeAvg).toLocaleString() + ' params<br>';
        if (!html) html = 'No model';
        document.getElementById('modelInfo').innerHTML = html;
    }

    log(msg) {
        const el   = document.getElementById('logs');
        const line = document.createElement('div');
        const time = new Date().toLocaleTimeString();
        line.textContent = '[' + time + '] ' + msg;
        if (msg.startsWith('ERROR')) line.style.color = '#ff6b6b';
        el.appendChild(line);
        el.scrollTop = el.scrollHeight;
    }
}

document.addEventListener('DOMContentLoaded', () => { new MNISTApp(); });
