class MNISTDataLoader {

    // Load CSV: rows = label, px0..px783  (785 cols, no header)
    async loadCSVFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onerror = () => reject(new Error('Cannot read file'));
            reader.onload = (e) => {
                try {
                    const lines = e.target.result
                        .replace(/\r\n/g, '\n').replace(/\r/g, '\n')
                        .split('\n')
                        .filter(l => l.trim() !== '');

                    const labels = [];
                    const pixels = [];

                    for (const line of lines) {
                        const vals = line.trim().split(',');
                        if (vals.length !== 785) continue;
                        const nums = vals.map(Number);
                        if (nums.some(isNaN)) continue;
                        labels.push(nums[0]);
                        pixels.push(nums.slice(1));
                    }

                    if (labels.length === 0) {
                        reject(new Error('No valid rows in CSV')); return;
                    }

                    // Build tensors WITHOUT tf.tidy — they must survive outside this function
                    const pixelTensor = tf.tensor2d(pixels, [labels.length, 784]);
                    const xs = pixelTensor.div(255).reshape([labels.length, 28, 28, 1]);
                    pixelTensor.dispose();

                    const labelTensor = tf.tensor1d(labels, 'int32');
                    const ys = tf.oneHot(labelTensor, 10);
                    labelTensor.dispose();

                    resolve({ xs, ys, count: labels.length });
                } catch (err) { reject(err); }
            };
            reader.readAsText(file);
        });
    }

    async loadTrainFromFiles(file) {
        if (this.trainData) {
            this.trainData.xs.dispose();
            this.trainData.ys.dispose();
        }
        this.trainData = await this.loadCSVFile(file);
        return this.trainData;
    }

    async loadTestFromFiles(file) {
        if (this.testData) {
            this.testData.xs.dispose();
            this.testData.ys.dispose();
        }
        this.testData = await this.loadCSVFile(file);
        return this.testData;
    }

    // Add Gaussian noise — NO tf.tidy, caller disposes result
    addNoise(xs, stddev) {
        stddev = stddev || 0.3;
        const noise  = tf.randomNormal(xs.shape, 0, stddev);
        const noisy  = xs.add(noise).clipByValue(0, 1);
        noise.dispose();
        return noisy;
    }

    // Split train/val — NO tf.tidy, caller disposes all 4 tensors
    splitTrainVal(xs, ys, valRatio) {
        valRatio = valRatio || 0.1;
        const n    = xs.shape[0];
        const nVal = Math.floor(n * valRatio);
        const nTrn = n - nVal;
        return {
            trainXs: xs.slice([0,    0,0,0], [nTrn, 28,28,1]),
            trainYs: ys.slice([0,    0],     [nTrn, 10]),
            valXs:   xs.slice([nTrn, 0,0,0], [nVal, 28,28,1]),
            valYs:   ys.slice([nTrn, 0],     [nVal, 10]),
        };
    }

    // Random batch of k samples — NO tf.tidy, caller disposes
    getRandomTestBatch(xs, ys, k) {
        k = k || 5;
        const idx = Array.from({length: xs.shape[0]}, (_, i) => i);
        for (let i = idx.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [idx[i], idx[j]] = [idx[j], idx[i]];
        }
        const sel = idx.slice(0, k);
        return {
            batchXs: tf.gather(xs, sel),
            batchYs: tf.gather(ys, sel),
            indices: sel,
        };
    }

    // Draw a single image to canvas.
    // imgArray = plain JS nested array shape [28][28][1] (values 0..1)
    drawToCanvas(imgArray, canvas, scale) {
        scale = scale || 4;

        // Flatten [28][28][1] → 784 values
        const flat = [];
        for (let r = 0; r < 28; r++) {
            for (let c = 0; c < 28; c++) {
                // imgArray[r][c] is either [v] (shape [28,28,1]) or v (shape [28,28])
                const v = Array.isArray(imgArray[r][c]) ? imgArray[r][c][0] : imgArray[r][c];
                flat.push(v);
            }
        }

        const imgData = new ImageData(28, 28);
        for (let i = 0; i < 784; i++) {
            const v = Math.min(255, Math.max(0, Math.round(flat[i] * 255)));
            imgData.data[i * 4]     = v;
            imgData.data[i * 4 + 1] = v;
            imgData.data[i * 4 + 2] = v;
            imgData.data[i * 4 + 3] = 255;
        }

        const tmp = document.createElement('canvas');
        tmp.width = 28; tmp.height = 28;
        tmp.getContext('2d').putImageData(imgData, 0, 0);

        canvas.width  = 28 * scale;
        canvas.height = 28 * scale;
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    }

    dispose() {
        if (this.trainData) { this.trainData.xs.dispose(); this.trainData.ys.dispose(); this.trainData = null; }
        if (this.testData)  { this.testData.xs.dispose();  this.testData.ys.dispose();  this.testData  = null; }
    }
}
