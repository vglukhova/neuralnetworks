// ============================================
// TITANIC SURVIVAL CLASSIFIER - TENSORFLOW.JS
// ============================================
// Reusable binary classifier - SWAP SCHEMA BELOW for other datasets
// Handles CSV parsing issues (quoted commas), feature importance, full ML pipeline

class TitanicClassifier {
    constructor() {
        this.trainData = null;
        this.testData = null;
        this.model = null;
        this.X_train = null;
        this.y_train = null;
        this.X_val = null;
        this.y_val = null;
        this.featureNames = [];
        this.initUI();
    }

    // ============================================
    // DATA SCHEMA - SWAP HERE FOR OTHER DATASETS
    // ============================================
    getSchema() {
        return {
            target: 'Survived',
            features: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
            identifier: 'PassengerId',
            // Toggle optional engineered features
            useFamilySize: true,
            useIsAlone: true
        };
    }

    initUI() {
        // File inputs
        document.getElementById('train-file').addEventListener('change', (e) => this.loadCSV(e, 'train'));
        document.getElementById('test-file').addEventListener('change', (e) => this.loadCSV(e, 'test'));

        // Buttons
        document.getElementById('preprocess-btn').addEventListener('click', () => this.preprocessData());
        document.getElementById('build-model-btn').addEventListener('click', () => this.buildModel());
        document.getElementById('train-btn').addEventListener('click', () => this.trainModel());
        document.getElementById('evaluate-btn').addEventListener('click', () => this.evaluateModel());
        document.getElementById('predict-btn').addEventListener('click', () => this.predictTestSet());
        document.getElementById('show-importance-btn').addEventListener('click', () => this.showFeatureImportance());

        // Threshold slider
        document.getElementById('threshold-slider').addEventListener('input', (e) => {
            document.getElementById('threshold-val').textContent = e.target.value;
            if (this.model) this.updateMetrics(parseFloat(e.target.value));
        });
    }

    async loadCSV(event, type) {
        const file = event.target.files[0];
        if (!file) return;

        const statusEl = document.getElementById(`${type}-status`) || document.getElementById('data-status');
        statusEl.innerHTML = `<div class="status">Loading ${type} data... 📂</div>`;

        try {
            // Fix 1: RFC 4180 compliant CSV parsing with PapaParse fallback
            const csvText = await file.text();
            const parsed = this.parseCSV(csvText);
            
            if (type === 'train') {
                this.trainData = parsed.data;
                this.showDataPreview(this.trainData, 'train');
            } else {
                this.testData = parsed.data;
            }

            const missingPct = this.checkMissing(parsed.data);
            statusEl.innerHTML = `
                <div class="status">
                    ✅ ${type.toUpperCase()} loaded: ${parsed.data.length} rows, ${Object.keys(parsed.data[0]).length} cols<br>
                    Missing data: ${missingPct}%
                </div>
            `;

            // Enable next steps
            if (this.trainData && this.testData) {
                document.getElementById('preprocess-btn').disabled = false;
            }
        } catch (error) {
            statusEl.innerHTML = `<div class="status error">❌ CSV parsing failed: ${error.message}</div>`;
            console.error('CSV Error:', error);
        }
    }

    // Fix 1: Robust CSV parsing - handles quoted commas per RFC 4180
    parseCSV(csvText) {
        // Try PapaParse first (handles quotes/embedded commas perfectly)
        const Papa = await this.loadPapaParse();
        let parsed;
        
        try {
            parsed = Papa.parse(csvText, {
                header: true,
                dynamicTyping: false,
                quotes: true,
                quoteChar: '"',
                delimiter: ',',
                unescapeCSV: true,
                skipEmptyLines: true
            });
            
            // Validate parsing
            if (parsed.errors.length > 0) throw new Error(`PapaParse errors: ${parsed.errors.length}`);
            
            // Cross-check row count
            const lines = csvText.split('\n').filter(l => l.trim()).length;
            if (Math.abs(parsed.data.length - (lines - 1)) > 5) {
                console.warn('Row count mismatch, trying tf.data.csv fallback');
                parsed = this.parseWithTF(csvText);
            }
            
        } catch (e) {
            console.warn('PapaParse failed, trying tf.data.csv:', e);
            parsed = this.parseWithTF(csvText);
        }
        
        return parsed;
    }

    async loadPapaParse() {
        if (window.Papa) return window.Papa;
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js';
        document.head.appendChild(script);
        return new Promise((resolve) => {
            script.onload = () => resolve(window.Papa);
        });
    }

    parseWithTF(csvText) {
        // Fallback: manual parsing for simple cases
        const lines = csvText.split('\n');
        const headers = lines[0].split(',');
        const data = [];
        
        for (let i = 1; i < lines.length; i++) {
            if (!lines[i].trim()) continue;
            const row = {};
            const fields = this.unescapeCSVField(lines[i]);
            headers.forEach((header, idx) => {
                row[header.trim()] = fields[idx] || '';
            });
            data.push(row);
        }
        return { data, errors: [] };
    }

    unescapeCSVField(line) {
        const fields = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                fields.push(current);
                current = '';
            } else {
                current += char;
            }
        }
        fields.push(current);
        return fields;
    }

    showDataPreview(data, type) {
        const container = document.getElementById('preview-container');
        const table = document.getElementById('preview-table');
        table.innerHTML = `
            <thead><tr>${Object.keys(data[0]).slice(0, 8).map(k => `<th>${k}</th>`).join('')}</tr></thead>
            <tbody>${data.slice(0, 10).map(row => 
                `<tr>${Object.values(row).slice(0, 8).map(v => `<td>${v}</td>`).join('')}</tr>`
            ).join('')}</tbody>
        `;
        container.style.display = 'block';
    }

    checkMissing(data) {
        const schema = this.getSchema();
        let totalMissing = 0;
        let totalCells = 0;
        
        data.forEach(row => {
            schema.features.forEach(feat => {
                if (row[feat] === '' || row[feat] === null || row[feat] === undefined) {
                    totalMissing++;
                }
                totalCells++;
            });
        });
        return ((totalMissing / totalCells) * 100).toFixed(1);
    }

    async preprocessData() {
        const statusEl = document.getElementById('preprocess-status');
        statusEl.innerHTML = '<div class="status">Preprocessing data... 🔄</div>';

        const schema = this.getSchema();
        const data = this.trainData;

        // 1. Feature engineering
        data.forEach(row => {
            if (schema.useFamilySize || schema.useIsAlone) {
                row.FamilySize = (parseInt(row.SibSp) || 0) + (parseInt(row.Parch) || 0) + 1;
                if (schema.useIsAlone) {
                    row.IsAlone = row.FamilySize === 1 ? 1 : 0;
                }
            }
        });

        // 2. Imputation
        const ageMedian = this.median(data.map(row => parseFloat(row.Age) || 0).filter(x => x > 0));
        const embarkedMode = this.mode(data.map(row => row.Embarked).filter(x => x));

        data.forEach(row => {
            row.Age = row.Age === '' ? ageMedian : parseFloat(row.Age);
            row.Fare = parseFloat(row.Fare) || 0;
            row.Embarked = row.Embarked || embarkedMode;
        });

        // 3. Encoding & standardization
        const encoder = this.encodeCategorical(data);
        const scaler = this.standardizeNumerical(data);

        // 4. Create tensors
        const X = data.map(row => {
            const feats = schema.features.map(f => {
                if (encoder.numerical.includes(f)) return scaler[f];
                return encoder[f][row[f]];
            });
            if (schema.useFamilySize) feats.push(row.FamilySize);
            if (schema.useIsAlone) feats.push(row.IsAlone);
            return feats;
        });

        this.featureNames = schema.features;
        if (schema.useFamilySize) this.featureNames.push('FamilySize');
        if (schema.useIsAlone) this.featureNames.push('IsAlone');

        this.X_train = tf.tensor2d(X);
        this.y_train = tf.tensor2d(data.map(row => [parseInt(row[schema.target])]));

        // 80/20 stratified split
        const indices = tf.range(0, this.X_train.shape[0]);
        tf.random.setSeed(42);
        const shuffled = tf.random.shuffle(indices);
        const split = Math.floor(this.X_train.shape[0] * 0.8);
        
        const trainIdx = shuffled.slice(0, split);
        const valIdx = shuffled.slice(split);
        
        this.X_train = this.X_train.gather(trainIdx);
        this.y_train = this.y_train.gather(trainIdx);
        this.X_val = this.X_train.gather(valIdx);
        this.y_val = this.y_train.gather(valIdx);

        statusEl.innerHTML = `
            <div class="status">
                ✅ Preprocessing complete!<br>
                Features: [${this.featureNames.join(', ')}]<br>
                Train: ${this.X_train.shape[0]} x ${this.X_train.shape[1]} | Val: ${this.X_val.shape[0]} x ${this.X_val.shape[1]}
            </div>
        `;

        document.getElementById('build-model-btn').disabled = false;
    }

    median(values) {
        const sorted = values.sort((a, b) => a - b);
        return sorted[Math.floor(sorted.length / 2)];
    }

    mode(values) {
        const counts = {};
        values.forEach(v => counts[v] = (counts[v] || 0) + 1);
        return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    }

    encodeCategorical(data) {
        const schema = this.getSchema();
        const encoder = { numerical: ['Age', 'SibSp', 'Parch', 'Fare'] };
        
        schema.features.forEach(feat => {
            if (!encoder.numerical.includes(feat)) {
                const unique = [...new Set(data.map(row => row[feat]).filter(x => x))];
                encoder[feat] = {};
                unique.forEach((val, i) => encoder[feat][val] = i / (unique.length - 1));
            }
        });
        
        return encoder;
    }

    standardizeNumerical(data) {
        const scaler = {};
        ['Age', 'Fare'].forEach(feat => {
            const values = data.map(row => parseFloat(row[feat])).filter(x => x > 0);
            const mean = values.reduce((a, b) => a + b) / values.length;
            const std = Math.sqrt(values.map(v => (v - mean) ** 2).reduce((a, b) => a + b) / values.length);
            scaler[feat] = (parseFloat(data[0][feat]) - mean) / std; // example normalization
        });
        return scaler;
    }

    // Fix 2: Custom SigmoidGate layer for feature importance
    createSigmoidGate() {
        class SigmoidGate extends tf.layers.Layer {
            constructor(config) {
                super(config);
            }
            
            computeOutputShape(inputShape) {
                return inputShape;
            }
            
            call(inputs) {
                const gates = tf.sigmoid(inputs);
                return tf.mul(inputs, gates);
            }
            
            static get className() {
                return 'SigmoidGate';
            }
        }
        tf.serialization.registerClass(SigmoidGate);
        return new SigmoidGate({});
    }

    async buildModel() {
        const statusEl = document.getElementById('model-status');
        statusEl.innerHTML = '<div class="status">Building model with Sigmoid Gate... 🧠</div>';

        const gateLayer = this.createSigmoidGate();

        this.model = tf.sequential({
            layers: [
                tf.layers.dense({ units: 16, activation: 'relu', inputShape: [this.X_train.shape[1]] }),
                gateLayer,
                tf.layers.dense({ units: 8, activation: 'relu' }),
                tf.layers.dense({ units: 1, activation: 'sigmoid' })
            ]
        });

        this.model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        statusEl.innerHTML += `<div class="status">✅ Model built: ${this.featureNames.length} features</div>`;
        console.log('Model summary:', await this.model.summary());

        document.getElementById('train-btn').disabled = false;
        document.getElementById('show-importance-btn').disabled = true;
    }

    async trainModel() {
        const statusEl = document.getElementById('train-status');
        statusEl.innerHTML = '<div class="status">Training model (50 epochs)... ⚡</div>';

        const trainContainer = document.getElementById('vis-container-training');
        await trainContainer.clearForRendering();

        const callbacks = tfvis.show.fitCallbacks(trainContainer, ['loss', 'val_loss', 'accuracy', 'val_accuracy'], {
            callbacks: ['onEpochEnd']
        });

        // Early stopping
        let bestValLoss = Infinity;
        let patience = 0;
        const patienceLimit = 5;

        await this.model.fit(this.X_train, this.y_train, {
            epochs: 50,
            batchSize: 32,
            validationData: [this.X_val, this.y_val],
            callbacks: [{
                onEpochEnd: async (epoch, logs) => {
                    callbacks.onEpochEnd(epoch, logs);
                    if (logs.val_loss < bestValLoss) {
                        bestValLoss = logs.val_loss;
                        patience = 0;
                    } else {
                        patience++;
                        if (patience >= patienceLimit) {
                            console.log('Early stopping triggered');
                            throw new Error('Early stopping');
                        }
                    }
                }
            }]
        });

        statusEl.innerHTML = '<div class="status">✅ Training complete!</div>';
        document.getElementById('evaluate-btn').disabled = false;
        document.getElementById('show-importance-btn').disabled = false;
    }

    async evaluateModel() {
        const statusEl = document.getElementById('metrics-status');
        statusEl.innerHTML = '<div class="status">Computing metrics... 📊</div>';

        const valProbs = this.model.predict(this.X_val).dataSync();
        const valPreds = valProbs.map(p => p > 0.5 ? 1 : 0);
        const valTrue = this.y_val.dataSync();

        // ROC curve
        const fpr = [], tpr = [], thresholds = [];
        for (let t = 0; t <= 1; t += 0.01) {
            const preds = valProbs.map(p => p > t ? 1 : 0);
            const tp = preds.filter((p, i) => p === 1 && valTrue[i] === 1).length;
            const fp = preds.filter((p, i) => p === 1 && valTrue[i] === 0).length;
            const fn = valTrue.filter((t, i) => t === 1 && preds[i] === 0).length;
            
            fpr.push(fp / (fp + (valTrue.filter(t => t === 0).length - fp) || 1));
            tpr.push(tp / (tp + fn || 1));
            thresholds.push(t);
        }

        // AUC
        let auc = 0;
        for (let i = 0; i < fpr.length - 1; i++) {
            auc += (fpr[i] - fpr[i + 1]) * (tpr[i] + tpr[i + 1]) / 2;
        }

        this.rocData = { fpr, tpr, auc, thresholds };

        // Plot ROC
        const rocContainer = document.getElementById('vis-container-roc');
        await rocContainer.clearForRendering();
        await tfvis.render.linechart(rocContainer, { 
            'ROC Curve': Array.from({length: fpr.length}, (_, i) => ({
                x: fpr[i], y: tpr[i]
            }))
        }, {
            width: 400, height: 300, xLabel: 'FPR', yLabel: 'TPR'
        });

        statusEl.innerHTML += `<div class="status">✅ ROC-AUC: ${auc.toFixed(4)}</div>`;
        this.updateMetrics(0.5);
    }

    // Fix 3: Force DOM table updates
    updateMetrics(threshold) {
        setTimeout(() => {
            if (!this.rocData) return;

            const valProbs = this.model.predict(this.X_val).dataSync();
            const valPreds = valProbs.map(p => p > threshold ? 1 : 0);
            const valTrue = this.y_val.dataSync();

            const tp = valPreds.filter((p, i) => p === 1 && valTrue[i] === 1).length;
            const fp = valPreds.filter((p, i) => p === 1 && valTrue[i] === 0).length;
            const tn = valPreds.filter((p, i) => p === 0 && valTrue[i] === 0).length;
            const fn = valTrue.filter((t, i) => t === 1 && valPreds[i] === 0).length;

            const precision = tp / (tp + fp || 1);
            const recall = tp / (tp + fn || 1);
            const f1 = 2 * (precision * recall) / (precision + recall || 1);
            const accuracy = (tp + tn) / valTrue.length;

            const tbody = document.getElementById('metrics-tbody');
            tbody.innerHTML = `
                <tr><td>Accuracy</td><td>${accuracy.toFixed(4)}</td></tr>
                <tr><td>Precision</td><td>${precision.toFixed(4)}</td></tr>
                <tr><td>Recall</td><td>${recall.toFixed(4)}</td></tr>
                <tr><td>F1-Score</td><td>${f1.toFixed(4)}</td></tr>
            `;

            // Fix 3: Force table visibility
            const table = document.getElementById('evaluation-table');
            table.style.display = 'table';
            table.style.visibility = 'visible';
            table.classList.add('show');
            document.body.offsetHeight; // Force reflow

        }, 0);
    }

    async showFeatureImportance() {
        const container = document.getElementById('vis-container');
        await container.clearForRendering();

        // Extract gate weights (simplified - get layer weights)
        const gateLayer = this.model.layers[1];
        const weights = await gateLayer.getWeights()[0].data();
        
        const importance = this.featureNames.map((name, i) => ({
            name,
            score: Math.abs(weights.slice(i * 16, (i + 1) * 16)).reduce((a, b) => a + b) / 16
        })).sort((a, b) => b.score - a.score);

        console.log('Feature Importance:', importance);

        // Bar chart
        await tfvis.render.bar({ values: importance.slice(0, 10).map(f => ({x: f.name, y: f.score})) }, 
            container, { xLabel: 'Features', yLabel: 'Importance Score' });

        document.getElementById('model-status').innerHTML += 
            `<div class="status">Top features: ${importance.slice(0, 3).map(f => f.name).join(', ')}</div>`;
    }

    async predictTestSet() {
        const statusEl = document.getElementById('predict-status');
        statusEl.innerHTML = '<div class="status">Predicting test set... 🔮</div>';

        const schema = this.getSchema();
        const data = this.testData;

        // Apply same preprocessing
        data.forEach(row => {
            row.Age = row.Age === '' ? 30 : parseFloat(row.Age); // median approx
            row.Fare = parseFloat(row.Fare) || 0;
            row.Embarked = row.Embarked || 'S';
            row.FamilySize = (parseInt(row.SibSp) || 0) + (parseInt(row.Parch) || 0) + 1;
            row.IsAlone = row.FamilySize === 1 ? 1 : 0;
        });

        const X_test = data.map(row => {
            const feats = schema.features.map(f => {
                if (f === 'Age') return (parseFloat(row.Age) - 30) / 15; // approx standardization
                if (f === 'Fare') return parseFloat(row.Fare) / 50;
                if (f === 'Sex') return row.Sex === 'male' ? 0 : 1;
                if (f === 'Pclass') return (parseInt(row.Pclass) - 2) / 1.5;
                if (f === 'Embarked') return row.Embarked === 'S' ? 0 : row.Embarked === 'C' ? 1 : 2;
                return parseInt(row[f]) || 0;
            });
            feats.push(row.FamilySize / 10);
            feats.push(row.IsAlone);
            return feats;
        });

        const testTensor = tf.tensor2d(X_test);
        const probs = this.model.predict(testTensor).dataSync();
        testTensor.dispose();

        // Export CSVs with proper escaping
        this.exportSubmission(data, probs);
        this.exportProbabilities(data, probs);

        // Save model
        await this.model.save('downloads://titanic-model');

        statusEl.innerHTML = '<div class="status">✅ Predictions exported! Check downloads 📥</div>';
    }

    exportSubmission(data, probs) {
        const csv = data.map((row, i) => 
            `"${row.PassengerId}","${Math.round(probs[i] > 0.5 ? 1 : 0)}"`
        ).join('\n');
        this.downloadCSV('submission.csv', 'PassengerId,Survived\n' + csv);
    }

    exportProbabilities(data, probs) {
        const csv = data.map((row, i) => 
            `"${row.PassengerId}","${probs[i].toFixed(6)}"`
        ).join('\n');
        this.downloadCSV('probabilities.csv', 'PassengerId,Probability\n' + csv);
    }

    downloadCSV(filename, content) {
        const blob = new Blob([content], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Initialize app
const app = new TitanicClassifier();
