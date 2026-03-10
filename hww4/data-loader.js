class MNISTDataLoader {

    // Load CSV file: each row = label, px0..px783  (785 values, no header)
    async loadCSVFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.onload = (e) => {
                try {
                    const lines = e.target.result
                        .replace(/\r\n/g, '\n')
                        .replace(/\r/g, '\n')
                        .split('\n')
                        .filter(l => l.trim().length > 0);

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
                        reject(new Error('No valid rows found in CSV'));
                        return;
                    }

                    const xs = tf.tidy(() =>
                        tf.tensor2d(pixels, [labels.length, 784])
                            .div(255)
                            .reshape([labels.length, 28, 28, 1])
                    );
                    const ys = tf.tidy(() =>
                        tf.oneHot(tf.tensor1d(labels, 'int32'), 10)
                    );

                    resolve({ xs, ys, count: labels.length });
                } catch (err) {
                    reject(err);
                }
            };
            reader.readAsText(file);
        });
    }

    async loadTrainFromFiles(file) {
        this.trainData = await this.loadCSVFile(file);
        return this.trainData;
    }

    async loadTestFromFiles(file) {
        this.testData = await this.loadCSVFile(file);
        return this.testData;
    }

    // Add Gaussian noise, clip to [0,1]  (NO tf.tidy — caller owns result)
    addNoise(xs, stddev) {
        stddev = stddev || 0.3;
        const noise = tf.randomNormal(xs.shape, 0, stddev);
        const result = xs.add(noise).clipByValue(0, 1);
        noise.dispose();
        return result;
    }

    // Split into train / validation  (NO tf.tidy — callers own the tensors)
    splitTrainVal(xs, ys, valRatio) {
        valRatio = valRatio || 0.1;
        const total  = xs.shape[0];
        const numVal = Math.floor(total * valRatio);
        const numTrn = total - numVal;
        return {
            trainXs: xs.slice([0,      0, 0, 0], [numTrn, 28, 28, 1]),
            trainYs: ys.slice([0,      0],        [numTrn, 10]),
            valXs:   xs.slice([numTrn, 0, 0, 0], [numVal, 28, 28, 1]),
            valYs:   ys.slice([numTrn, 0],        [numVal, 10]),
        };
    }

    // Pick k random samples  (NO tf.tidy — caller owns tensors)
    getRandomTestBatch(xs, ys, k) {
        k = k || 5;
        const all = [];
        for (let i = 0; i < xs.shape[0]; i++) all.push(i);
        for (let i = all.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            const tmp = all[i]; all[i] = all[j]; all[j] = tmp;
        }
        const indices = all.slice(0, k);
        return {
            batchXs: tf.gather(xs, indices),
            batchYs: tf.gather(ys, indices),
            indices: indices,
        };
    }

    // Draw image to canvas. img can be a flat/nested JS array or a tf.Tensor [28,28,1] or [28,28]
    drawToCanvas(img, canvas, scale) {
        scale = scale || 4;

        // If a tensor is passed, extract data and dispose it
        let flat;
        if (img instanceof tf.Tensor) {
            flat = Array.from(img.reshape([784]).dataSync());
            img.dispose();
        } else {
            // Flatten nested array (shape [28,28,1] or [28,28])
            flat = img.flat ? img.flat(Infinity) : [].concat(...img.map(row =>
                Array.isArray(row[0]) ? [].concat(...row) : row
            ));
        }

        const imgData = new ImageData(28, 28);
        for (let i = 0; i < 784; i++) {
            const v = Math.round(flat[i] * 255);
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
        ctx.drawImage(tmp, 0, 0, 28 * scale, 28 * scale);
    }

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
