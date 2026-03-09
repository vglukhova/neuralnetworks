/**
 * app.js
 * DenoiserApp — CNN Autoencoder denoiser (Homework Slide 33).
 *
 * Extends the pattern of the original MNISTApp (app-13.js) with:
 *   • Two models: Max Pooling autoencoder vs Avg Pooling autoencoder
 *   • Noise injection (Gaussian, σ configurable via slider)
 *   • Custom mini loss chart (no tfjs-vis needed for training curve)
 *   • onTestFive shows: Original | Noisy | Max denoised | Avg denoised + PSNR
 *   • Separate Save/Load for each model
 *
 * Architecture — CNN Autoencoder:
 *   Encoder: Conv2D(32) → Pool(2×2) → Conv2D(64)
 *   Decoder: Conv2DTranspose(64) → UpSampling2D → Conv2DTranspose(32) → Conv2D(1, sigmoid)
 *   Loss: MSE  |  Optimizer: Adam  |  Input/Output: [N, 28, 28, 1]
 */
class DenoiserApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();
        this.modelMax   = null;   // MaxPooling autoencoder
        this.modelAvg   = null;   // AveragePooling autoencoder
        this.trainData  = null;
        this.testData   = null;
        this.isTraining = false;

        // Loss history for the mini chart: arrays of per-epoch MSE
        this.lossHistory = { max: [], avg: [] };

        this.initializeUI();
        this.drawLossChart();   // draw empty axes on startup
    }

    // ─────────────────────────────────────────────────────────
    //  UI wiring
    // ─────────────────────────────────────────────────────────
    initializeUI() {
        document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
        document.getElementById('trainMaxBtn').addEventListener('click', () => this.onTrain('max'));
        document.getElementById('trainAvgBtn').addEventListener('click', () => this.onTrain('avg'));
        document.getElementById('testFiveBtn').addEventListener('click', () => this.onTestFive());
        document.getElementById('saveMaxBtn').addEventListener('click', () => this.onSaveDownload('max'));
        document.getElementById('saveAvgBtn').addEventListener('click', () => this.onSaveDownload('avg'));
        document.getElementById('loadMaxBtn').addEventListener('click', () => this.onLoadFromFiles('max'));
        document.getElementById('loadAvgBtn').addEventListener('click', () => this.onLoadFromFiles('avg'));
        document.getElementById('resetBtn').addEventListener('click', () => this.onReset());
        document.getElementById('toggleVisorBtn').addEventListener('click', () => this.toggleVisor());

        // Live noise slider label
        const slider = document.getElementById('noiseSlider');
        slider.addEventListener('input', () => {
            document.getElementById('noiseVal').textContent = (+slider.value).toFixed(2);
        });
    }

    // ─────────────────────────────────────────────────────────
    //  Step 1 — Load Data
    // ─────────────────────────────────────────────────────────
    async onLoadData() {
        const trainFile = document.getElementById('trainFile').files[0];
        const testFile  = document.getElementById('testFile').files[0];

        if (!trainFile || !testFile) {
            this.showError('Please select both Train and Test CSV files');
            return;
        }

        try {
            console.log('Loading train file:', trainFile.name);
            this.showStatus('Loading train CSV…');
            this.trainData = await this.dataLoader.loadTrainFromFiles(trainFile);
            console.log('Train data loaded:', this.trainData);

            console.log('Loading test file:', testFile.name);
            this.showStatus('Loading test CSV…');
            this.testData  = await this.dataLoader.loadTestFromFiles(testFile);
            console.log('Test data loaded:', this.testData);

            this.updateDataStatus(this.trainData.count, this.testData.count);
            this.showStatus(`Data loaded — train: ${this.trainData.count}, test: ${this.testData.count}`);

            // Preview first few training images to verify loading
            this.previewFirstImages();

        } catch (err) {
            console.error('Error in onLoadData:', err);
            this.showError(`Failed to load data: ${err.message}`);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Preview first few images to verify data loading
    // ─────────────────────────────────────────────────────────
    previewFirstImages() {
        if (!this.trainData) return;
        
        tf.tidy(() => {
            // Get first 5 images
            const firstImages = this.trainData.xs.slice([0, 0, 0, 0], [5, 28, 28, 1]);
            
            // Create a temporary preview container
            const container = document.getElementById('previewContainer');
            container.innerHTML = '<h3>First 5 training images (verification):</h3>';
            
            const strip = document.createElement('div');
            strip.className = 'img-strip';
            strip.style.marginBottom = '20px';
            
            for (let i = 0; i < 5; i++) {
                const imgTensor = firstImages.slice([i, 0, 0, 0], [1, 28, 28, 1]).reshape([28, 28, 1]);
                const canvas = document.createElement('canvas');
                canvas.style.margin = '2px';
                this.dataLoader.draw28x28ToCanvas(imgTensor, canvas, 2);
                strip.appendChild(canvas);
                imgTensor.dispose();
            }
            
            container.appendChild(strip);
            firstImages.dispose();
        });
    }

    // ─────────────────────────────────────────────────────────
    //  Step 3 — Train one of the two autoencoders
    //  poolType: 'max' | 'avg'
    // ─────────────────────────────────────────────────────────
    async onTrain(poolType) {
        if (!this.trainData) {
            this.showError('Please load training data first');
            return;
        }
        if (this.isTraining) {
            this.showError('Training already in progress');
            return;
        }

        const sigma    = +document.getElementById('noiseSlider').value;
        const epochs   = +document.getElementById('epochSelect').value;
        const label    = poolType === 'max' ? 'MAX' : 'AVG';

        try {
            this.isTraining = true;
            this.showStatus(`Starting ${label} Pooling training (σ=${sigma}, epochs=${epochs})…`);
            this.showProgress(`Epoch 0/${epochs} — ${label} Pooling`);

            // Dispose previous model of same type to free GPU memory
            if (poolType === 'max' && this.modelMax) { 
                this.modelMax.dispose(); 
                this.modelMax = null; 
            }
            if (poolType === 'avg' && this.modelAvg) { 
                this.modelAvg.dispose(); 
                this.modelAvg = null; 
            }

            const model = this.createModel(poolType);

            // Validation split (90 / 10)
            const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
                this.trainData.xs, this.trainData.ys, 0.1
            );

            const nTrain  = trainXs.shape[0];
            const BATCH   = 128;
            const nBatch  = Math.ceil(nTrain / BATCH);
            const t0      = Date.now();

            this.lossHistory[poolType] = [];   // reset loss curve for this model

            // ── Manual epoch loop ──
            for (let epoch = 0; epoch < epochs; epoch++) {
                let epochLoss = 0;

                for (let b = 0; b < nBatch; b++) {
                    const start = b * BATCH;
                    const len   = Math.min(BATCH, nTrain - start);

                    // Autoencoder input: noisy version of the clean image
                    // Autoencoder target: the original clean image
                    const cleanBatch = trainXs.slice([start, 0, 0, 0], [len, 28, 28, 1]);
                    const noisyBatch = this.addNoise(cleanBatch, sigma);

                    // Train on batch
                    const history = await model.fit(noisyBatch, cleanBatch, {
                        batchSize: len,
                        epochs: 1,
                        verbose: 0
                    });
                    
                    const loss = history.history.loss[0];
                    epochLoss += loss;

                    // Free batch tensors
                    cleanBatch.dispose();
                    noisyBatch.dispose();

                    // Update progress bar
                    const pct = ((epoch * nBatch + b + 1) / (epochs * nBatch) * 100).toFixed(0);
                    this.setProgressFill(pct);
                    
                    // Yield control to keep UI responsive
                    await new Promise(r => setTimeout(r, 0));
                }

                const avgLoss = epochLoss / nBatch;
                this.lossHistory[poolType].push(avgLoss);
                this.drawLossChart();   // redraw chart after each epoch

                const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
                this.setProgressLabel(`Epoch ${epoch+1}/${epochs} — ${label} — loss: ${avgLoss.toFixed(5)} — ${elapsed}s`);
                this.showStatus(`${label} epoch ${epoch+1}/${epochs} loss=${avgLoss.toFixed(5)}`);
            }

            // Store trained model
            if (poolType === 'max') {
                this.modelMax = model;
            } else {
                this.modelAvg = model;
            }

            trainXs.dispose();
            trainYs.dispose();
            valXs.dispose();
            valYs.dispose();
            this.hideProgress();

            const totalTime = ((Date.now() - t0) / 1000).toFixed(1);
            this.showStatus(`✅ ${label} Pooling training done in ${totalTime}s`);
            this.updateModelInfo();

        } catch (err) {
            console.error('Training error:', err);
            this.showError(`Training failed: ${err.message}`);
        } finally {
            this.isTraining = false;
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Step 4 — Test 5 Random images
    // ─────────────────────────────────────────────────────────
    async onTestFive() {
        if (!this.testData) {
            this.showError('Please load test data first');
            return;
        }
        if (!this.modelMax && !this.modelAvg) {
            this.showError('Train at least one model first');
            return;
        }

        try {
            const sigma = +document.getElementById('noiseSlider').value;
            this.showStatus(`Testing 5 random samples (σ=${sigma})…`);

            // Sample 5 random clean test images
            const { batchXs: cleanBatch } = this.dataLoader.getRandomTestBatch(
                this.testData.xs, this.testData.ys, 5
            );

            // Add noise once for all 5 images
            const noisyBatch = this.addNoise(cleanBatch, sigma);

            // Run available models
            const maxRecon = this.modelMax ? this.modelMax.predict(noisyBatch) : null;
            const avgRecon = this.modelAvg ? this.modelAvg.predict(noisyBatch) : null;

            // Convert tensors to plain JS arrays
            const cleanArr = await cleanBatch.array();
            const noisyArr = await noisyBatch.array();
            const maxArr   = maxRecon ? await maxRecon.array() : null;
            const avgArr   = avgRecon ? await avgRecon.array() : null;

            // Render result cards
            this.renderPreview(cleanArr, noisyArr, maxArr, avgArr);

            // Cleanup
            cleanBatch.dispose();
            noisyBatch.dispose();
            if (maxRecon) maxRecon.dispose();
            if (avgRecon) avgRecon.dispose();

            this.showStatus('✅ Test 5 Random done.');

        } catch (err) {
            console.error('Test error:', err);
            this.showError(`Test preview failed: ${err.message}`);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Save model to browser download
    // ─────────────────────────────────────────────────────────
    async onSaveDownload(poolType) {
        const model = poolType === 'max' ? this.modelMax : this.modelAvg;
        const label = poolType === 'max' ? 'MAX' : 'AVG';

        if (!model) {
            this.showError(`No ${label} model to save — train it first`);
            return;
        }

        try {
            const filename = `mnist-denoiser-${poolType}`;
            await model.save(`downloads://${filename}`);
            this.showStatus(`${label} model saved as ${filename}.json + .bin`);
        } catch (err) {
            this.showError(`Failed to save ${label} model: ${err.message}`);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Load model from user-selected files
    // ─────────────────────────────────────────────────────────
    async onLoadFromFiles(poolType) {
        const jsonFile    = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];
        const label       = poolType === 'max' ? 'MAX' : 'AVG';

        if (!jsonFile || !weightsFile) {
            this.showError('Please select both model .json and .bin files');
            return;
        }

        try {
            this.showStatus(`Loading ${label} model from files…`);

            // Dispose current model
            if (poolType === 'max' && this.modelMax) { 
                this.modelMax.dispose(); 
                this.modelMax = null; 
            }
            if (poolType === 'avg' && this.modelAvg) { 
                this.modelAvg.dispose(); 
                this.modelAvg = null; 
            }

            const loaded = await tf.loadLayersModel(
                tf.io.browserFiles([jsonFile, weightsFile])
            );
            
            // Re-compile
            loaded.compile({ 
                optimizer: tf.train.adam(1e-3), 
                loss: 'meanSquaredError' 
            });

            if (poolType === 'max') {
                this.modelMax = loaded;
            } else {
                this.modelAvg = loaded;
            }

            this.showStatus(`✅ ${label} model loaded successfully`);
            this.updateModelInfo();

        } catch (err) {
            console.error('Load model error:', err);
            this.showError(`Failed to load ${label} model: ${err.message}`);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Reset application
    // ─────────────────────────────────────────────────────────
    onReset() {
        if (this.modelMax) { 
            this.modelMax.dispose(); 
            this.modelMax = null; 
        }
        if (this.modelAvg) { 
            this.modelAvg.dispose(); 
            this.modelAvg = null; 
        }

        this.dataLoader.dispose();
        this.trainData = this.testData = null;

        this.lossHistory = { max: [], avg: [] };
        this.drawLossChart();

        this.updateDataStatus(0, 0);
        this.updateModelInfo();
        this.clearPreview();
        this.clearMetrics();
        this.showStatus('Reset completed');
    }

    // ─────────────────────────────────────────────────────────
    //  Toggle tfjs-vis visor
    // ─────────────────────────────────────────────────────────
    toggleVisor() {
        if (tfvis && tfvis.visor) {
            tfvis.visor().toggle();
        }
    }

    // ─────────────────────────────────────────────────────────
    //  CNN Autoencoder architecture
    // ─────────────────────────────────────────────────────────
    createModel(poolType = 'max') {
        const model = tf.sequential();

        // ── Encoder ──
        model.add(tf.layers.conv2d({
            filters: 32, 
            kernelSize: 3, 
            activation: 'relu',
            padding: 'same', 
            inputShape: [28, 28, 1]
        }));

        // Pooling layer (key difference)
        if (poolType === 'max') {
            model.add(tf.layers.maxPooling2d({ 
                poolSize: [2, 2], 
                padding: 'same' 
            }));
        } else {
            model.add(tf.layers.averagePooling2d({ 
                poolSize: [2, 2], 
                padding: 'same' 
            }));
        }

        model.add(tf.layers.conv2d({
            filters: 64, 
            kernelSize: 3, 
            activation: 'relu', 
            padding: 'same'
        }));

        // ── Decoder ──
        model.add(tf.layers.conv2dTranspose({
            filters: 64, 
            kernelSize: 3, 
            activation: 'relu', 
            padding: 'same'
        }));
        
        model.add(tf.layers.upSampling2d({ 
            size: [2, 2] 
        }));
        
        model.add(tf.layers.conv2dTranspose({
            filters: 32, 
            kernelSize: 3, 
            activation: 'relu', 
            padding: 'same'
        }));

        // Final layer
        model.add(tf.layers.conv2d({
            filters: 1, 
            kernelSize: 1, 
            activation: 'sigmoid', 
            padding: 'same'
        }));

        // Compile
        model.compile({ 
            optimizer: tf.train.adam(1e-3), 
            loss: 'meanSquaredError' 
        });

        return model;
    }

    // ─────────────────────────────────────────────────────────
    //  Add Gaussian noise to a batch tensor
    // ─────────────────────────────────────────────────────────
    addNoise(cleanBatch, sigma) {
        return tf.tidy(() => {
            const noise = tf.randomNormal(cleanBatch.shape, 0, sigma);
            return cleanBatch.add(noise).clipByValue(0, 1);
        });
    }

    // ─────────────────────────────────────────────────────────
    //  PSNR helper
    // ─────────────────────────────────────────────────────────
    computePSNR(cleanData, reconData) {
        // Flatten arrays
        const flatten = (arr) => {
            if (Array.isArray(arr[0])) {
                return arr.flat(3);
            }
            return arr;
        };
        
        const flat = flatten(cleanData);
        const flatR = flatten(reconData);
        
        let mse = 0;
        for (let i = 0; i < flat.length; i++) {
            const d = flat[i] - flatR[i];
            mse += d * d;
        }
        mse /= flat.length;
        
        return mse > 0 ? 10 * Math.log10(1.0 / mse) : 60;
    }

    // ─────────────────────────────────────────────────────────
    //  Render denoising result cards for 5 images
    // ─────────────────────────────────────────────────────────
    renderPreview(cleanArr, noisyArr, maxArr, avgArr) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '';

        let totalPsnrNoisy = 0, totalPsnrMax = 0, totalPsnrAvg = 0;

        for (let i = 0; i < 5; i++) {
            // Compute PSNR
            const psnrNoisy = this.computePSNR(cleanArr[i], noisyArr[i]);
            const psnrMax   = maxArr ? this.computePSNR(cleanArr[i], maxArr[i]) : null;
            const psnrAvg   = avgArr ? this.computePSNR(cleanArr[i], avgArr[i]) : null;

            totalPsnrNoisy += psnrNoisy;
            if (psnrMax !== null) totalPsnrMax += psnrMax;
            if (psnrAvg !== null) totalPsnrAvg += psnrAvg;

            // Card container
            const item = document.createElement('div');
            item.className = 'preview-item';

            const title = document.createElement('div');
            title.className = 'item-title';
            title.textContent = `Sample ${i + 1}`;
            item.appendChild(title);

            // Image strip
            const strip = document.createElement('div');
            strip.className = 'img-strip';

            const makeCell = (imageData, label) => {
                const cell = document.createElement('div');
                cell.className = 'img-cell';

                const canvas = document.createElement('canvas');
                
                // Create tensor and draw
                tf.tidy(() => {
                    const tensor = tf.tensor(imageData);
                    this.dataLoader.draw28x28ToCanvas(tensor, canvas, 2);
                });

                const lbl = document.createElement('span');
                lbl.textContent = label;

                cell.appendChild(canvas);
                cell.appendChild(lbl);
                return cell;
            };

            strip.appendChild(makeCell(cleanArr[i], 'Original'));
            strip.appendChild(makeCell(noisyArr[i], '+Noise'));
            if (maxArr) strip.appendChild(makeCell(maxArr[i], 'Max'));
            if (avgArr) strip.appendChild(makeCell(avgArr[i], 'Avg'));

            item.appendChild(strip);

            // PSNR badges
            const psnrRow = document.createElement('div');
            psnrRow.className = 'psnr-row';

            const noiseBadge = document.createElement('span');
            noiseBadge.className = 'psnr-badge psnr-noise';
            noiseBadge.textContent = `Noisy: ${psnrNoisy.toFixed(1)} dB`;
            psnrRow.appendChild(noiseBadge);

            if (psnrMax !== null) {
                const b = document.createElement('span');
                b.className = 'psnr-badge psnr-max';
                b.textContent = `Max: ${psnrMax.toFixed(1)} dB`;
                if (psnrAvg === null || psnrMax >= psnrAvg) {
                    b.classList.add('psnr-winner');
                }
                psnrRow.appendChild(b);
            }
            
            if (psnrAvg !== null) {
                const b = document.createElement('span');
                b.className = 'psnr-badge psnr-avg';
                b.textContent = `Avg: ${psnrAvg.toFixed(1)} dB`;
                if (psnrMax === null || psnrAvg > psnrMax) {
                    b.classList.add('psnr-winner');
                }
                psnrRow.appendChild(b);
            }

            item.appendChild(psnrRow);
            container.appendChild(item);
        }

        // Update global metrics
        document.getElementById('metNoise').textContent = (totalPsnrNoisy / 5).toFixed(2) + ' dB';
        document.getElementById('metMax').textContent = maxArr ? (totalPsnrMax / 5).toFixed(2) + ' dB' : '—';
        document.getElementById('metAvg').textContent = avgArr ? (totalPsnrAvg / 5).toFixed(2) + ' dB' : '—';

        if (maxArr && avgArr) {
            document.getElementById('metWinner').textContent =
                totalPsnrMax >= totalPsnrAvg ? 'Max Pool 🔵' : 'Avg Pool 🔴';
        } else if (maxArr) {
            document.getElementById('metWinner').textContent = 'Max Pool 🔵';
        } else {
            document.getElementById('metWinner').textContent = 'Avg Pool 🔴';
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Clear preview
    // ─────────────────────────────────────────────────────────
    clearPreview() {
        document.getElementById('previewContainer').innerHTML =
            '<p style="color:#bbb">Train both models, then click "Test 5 Random"</p>';
    }

    // ─────────────────────────────────────────────────────────
    //  Clear metrics
    // ─────────────────────────────────────────────────────────
    clearMetrics() {
        ['metNoise','metMax','metAvg','metWinner'].forEach(id => {
            document.getElementById(id).textContent = '—';
        });
    }

    // ─────────────────────────────────────────────────────────
    //  Mini loss chart
    // ─────────────────────────────────────────────────────────
    drawLossChart() {
        const canvas = document.getElementById('lossChartCanvas');
        if (!canvas) return;
        
        const W = canvas.width, H = canvas.height;
        const ctx = canvas.getContext('2d');

        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, W, H);

        const allVals = [...this.lossHistory.max, ...this.lossHistory.avg];
        const maxLoss = allVals.length ? Math.max(...allVals, 0.01) : 0.1;
        const pad = { t: 10, b: 22, l: 38, r: 10 };

        const toX = (i, len) => pad.l + (i / Math.max(len - 1, 1)) * (W - pad.l - pad.r);
        const toY = (v) => pad.t + (1 - v / maxLoss) * (H - pad.t - pad.b);

        // Axes
        ctx.strokeStyle = '#ccc'; 
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad.l, pad.t);
        ctx.lineTo(pad.l, H - pad.b);
        ctx.lineTo(W - pad.r, H - pad.b);
        ctx.stroke();

        // Y-axis labels + grid lines
        ctx.fillStyle = '#888'; 
        ctx.font = '9px Arial'; 
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        
        for (let t = 0; t <= 4; t++) {
            const v = maxLoss * t / 4;
            const yy = toY(v);
            ctx.fillText(v.toFixed(3), pad.l - 3, yy);
            
            ctx.strokeStyle = '#f0f0f0';
            ctx.beginPath(); 
            ctx.moveTo(pad.l, yy); 
            ctx.lineTo(W - pad.r, yy); 
            ctx.stroke();
        }

        // X-axis label
        ctx.fillStyle = '#aaa'; 
        ctx.textAlign = 'center'; 
        ctx.font = '8px Arial';
        ctx.fillText('epoch', W / 2, H - 4);

        // Draw loss curves
        const drawLine = (data, color) => {
            if (data.length < 1) return;
            
            ctx.strokeStyle = color; 
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            data.forEach((v, i) => {
                const px = toX(i, data.length);
                const py = toY(v);
                if (i === 0) {
                    ctx.moveTo(px, py);
                } else {
                    ctx.lineTo(px, py);
                }
            });
            ctx.stroke();

            // Dot at last point
            if (data.length > 0) {
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(
                    toX(data.length - 1, data.length), 
                    toY(data[data.length - 1]), 
                    3, 0, Math.PI * 2
                );
                ctx.fill();
            }
        };

        drawLine(this.lossHistory.max, '#1565c0');
        drawLine(this.lossHistory.avg, '#c62828');

        // Legend
        ctx.font = '9px Arial'; 
        ctx.textAlign = 'left';
        ctx.fillStyle = '#1565c0'; 
        ctx.fillText('● Max Pool', pad.l + 4, pad.t + 10);
        ctx.fillStyle = '#c62828'; 
        ctx.fillText('● Avg Pool', pad.l + 68, pad.t + 10);
    }

    // ─────────────────────────────────────────────────────────
    //  Progress bar helpers
    // ─────────────────────────────────────────────────────────
    showProgress(label) {
        const wrap = document.getElementById('progressWrap');
        if (wrap) {
            wrap.style.display = 'block';
        }
        this.setProgressLabel(label);
        this.setProgressFill('0');
    }
    
    setProgressFill(pct) {
        const fill = document.getElementById('progressFill');
        if (fill) {
            fill.style.width = pct + '%';
        }
    }
    
    setProgressLabel(text) {
        const label = document.getElementById('progressLabel');
        if (label) {
            label.textContent = text;
        }
    }
    
    hideProgress() {
        const wrap = document.getElementById('progressWrap');
        if (wrap) {
            wrap.style.display = 'none';
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Status / error messages
    // ─────────────────────────────────────────────────────────
    showStatus(message) {
        const logs = document.getElementById('trainingLogs');
        if (!logs) return;
        
        const entry = document.createElement('div');
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logs.appendChild(entry);
        logs.scrollTop = logs.scrollHeight;
        
        console.log(message);
    }

    showError(message) {
        const logs = document.getElementById('trainingLogs');
        if (!logs) return;
        
        const entry = document.createElement('div');
        entry.className = 'err';
        entry.textContent = `[${new Date().toLocaleTimeString()}] ERROR: ${message}`;
        logs.appendChild(entry);
        logs.scrollTop = logs.scrollHeight;
        
        console.error(message);
    }

    // ─────────────────────────────────────────────────────────
    //  Update UI info panels
    // ─────────────────────────────────────────────────────────
    updateDataStatus(trainCount, testCount) {
        const statusDiv = document.getElementById('dataStatus');
        if (!statusDiv) return;
        
        statusDiv.innerHTML = `
            <h3>Data Status</h3>
            <p>Train samples: ${trainCount || 0}</p>
            <p>Test samples:  ${testCount || 0}</p>
        `;
    }

    updateModelInfo() {
        const paramCount = (model) => {
            if (!model) return 'not trained';
            let total = 0;
            model.layers.forEach(l => {
                l.getWeights().forEach(w => total += w.size);
            });
            return `${total.toLocaleString()} params`;
        };

        const infoDiv = document.getElementById('modelInfo');
        if (!infoDiv) return;
        
        infoDiv.innerHTML = `
            <h3>Model Info</h3>
            <p>Max Pool: ${paramCount(this.modelMax)}</p>
            <p>Avg Pool: ${paramCount(this.modelAvg)}</p>
        `;
    }
}

// Instantiate the app once the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing DenoiserApp...');
    new DenoiserApp();
});
