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

    // Add Gaussian noise, clip to [0,1]
    addNoise(xs, stddev) {
        stddev = stddev || 0.3;
        return tf.tidy(() =>
            xs.add(tf.randomNormal(xs.shape, 0, stddev)).clipByValue(0, 1)
        );
    }

    // Split into train / validation
    splitTrainVal(xs, ys, valRatio) {
        valRatio = valRatio || 0.1;
        const total  = xs.shape[0];
        const numVal = Math.floor(total * valRatio);
        const numTrn = total - numVal;
        return tf.tidy(() => ({
            trainXs: xs.slice([0,      0, 0, 0], [numTrn, 28, 28, 1]),
            trainYs: ys.slice([0,      0],        [numTrn, 10]),
            valXs:   xs.slice([numTrn, 0, 0, 0], [numVal, 28, 28, 1]),
            valYs:   ys.slice([numTrn, 0],        [numVal, 10]),
        }));
    }

    // Pick k random samples
    getRandomTestBatch(xs, ys, k) {
        k = k || 5;
        const all = [];
        for (let i = 0; i < xs.shape[0]; i++) all.push(i);
        for (let i = all.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            const tmp = all[i]; all[i] = all[j]; all[j] = tmp;
        }
        const indices = all.slice(0, k);
        return tf.tidy(() => ({
            batchXs: tf.gather(xs, indices),
            batchYs: tf.gather(ys, indices),
            indices: indices,
        }));
    }

    // Draw [28,28,1] tensor to canvas
    drawToCanvas(tensor, canvas, scale) {
        scale = scale || 4;
        const data = tf.tidy(() =>
            tensor.reshape([28, 28]).mul(255).clipByValue(0, 255).dataSync()
        );
        const imgData = new ImageData(28, 28);
        for (let i = 0; i < 784; i++) {
            imgData.data[i * 4]     = data[i];
            imgData.data[i * 4 + 1] = data[i];
            imgData.data[i * 4 + 2] = data[i];
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
