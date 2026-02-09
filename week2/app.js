// ============================================
// Titanic Survival Classifier with Sigmoid Gates
// TensorFlow.js - Browser-based Binary Classifier
// ============================================

// Global variables
let trainData = null;
let testData = null;
let processedTrainData = null;
let processedTestData = null;
let model = null;
let gateWeights = null;
let isTraining = false;
let trainingHistory = null;
let predictions = null;
let featureImportance = null;
let evaluationResults = null;

// Feature columns (excluding PassengerId and Survived)
const FEATURE_COLS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];
const TARGET_COL = 'Survived';
const ID_COL = 'PassengerId';

// DOM Elements
const elements = {
    trainFile: document.getElementById('trainFile'),
    testFile: document.getElementById('testFile'),
    loadDataBtn: document.getElementById('loadDataBtn'),
    inspectBtn: document.getElementById('inspectBtn'),
    preprocessBtn: document.getElementById('preprocessBtn'),
    createModelBtn: document.getElementById('createModelBtn'),
    trainBtn: document.getElementById('trainBtn'),
    stopTrainBtn: document.getElementById('stopTrainBtn'),
    evaluateBtn: document.getElementById('evaluateBtn'),
    predictBtn: document.getElementById('predictBtn'),
    exportModelBtn: document.getElementById('exportModelBtn'),
    exportPredBtn: document.getElementById('exportPredBtn'),
    exportImportanceBtn: document.getElementById('exportImportanceBtn'),
    showImportanceBtn: document.getElementById('showImportanceBtn'),
    useGateToggle: document.getElementById('useGateToggle'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue'),
    loadStatus: document.getElementById('loadStatus'),
    preprocessStatus: document.getElementById('preprocessStatus'),
    modelStatus: document.getElementById('modelStatus'),
    trainingStatus: document.getElementById('trainingStatus'),
    predictionStatus: document.getElementById('predictionStatus'),
    exportStatus: document.getElementById('exportStatus'),
    evaluationTable: document.getElementById('evaluation-table'),
    dataPreview: document.getElementById('data-preview'),
    inspectionCharts: document.getElementById('inspection-charts'),
    featureImportanceDiv: document.getElementById('feature-importance')
};

// ============================================
// 1. CSV Loading Functions - FIXED
// ============================================

/**
 * Simple CSV parser to handle escaped commas properly
 */
function parseCSV(text) {
    const lines = text.split('\n').filter(line => line.trim() !== '');
    if (lines.length === 0) return { headers: [], rows: [] };
    
    // Parse headers
    const headers = parseCSVLine(lines[0]);
    
    // Parse rows
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        if (values.length === headers.length) {
            const row = {};
            headers.forEach((header, index) => {
                row[header.trim()] = values[index] !== undefined ? values[index].trim() : '';
            });
            rows.push(row);
        }
    }
    
    return { headers, rows };
}

/**
 * Parse a single CSV line, handling quotes and escaped commas
 */
function parseCSVLine(line) {
    const values = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = i < line.length - 1 ? line[i + 1] : '';
        
        if (char === '"') {
            if (inQuotes && nextChar === '"') {
                // Escaped quote
                current += '"';
                i++; // Skip next char
            } else {
                // Start or end of quoted field
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            // End of field
            values.push(current);
            current = '';
        } else {
            current += char;
        }
    }
    
    // Add the last field
    values.push(current);
    return values;
}

/**
 * Load CSV file with manual parsing
 */
async function loadCSV(file, isTest = false) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            try {
                const csvText = e.target.result;
                const { headers, rows } = parseCSV(csvText);
                
                // Validate required columns
                const requiredCols = isTest ? 
                    ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'] :
                    ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'];
                
                const missingCols = requiredCols.filter(col => !headers.includes(col));
                if (missingCols.length > 0) {
                    throw new Error(`Missing columns: ${missingCols.join(', ')}`);
                }
                
                // Convert to the format expected by the rest of the app
                const data = {
                    raw: rows,
                    features: rows.map(row => {
                        const features = {};
                        FEATURE_COLS.forEach(col => {
                            features[col] = row[col];
                        });
                        return features;
                    }),
                    ids: rows.map(row => row[ID_COL])
                };
                
                if (!isTest) {
                    data.labels = rows.map(row => parseInt(row[TARGET_COL]) || 0);
                }
                
                showStatus('success', `${isTest ? 'Test' : 'Train'} data loaded: ${rows.length} rows`);
                resolve(data);
                
            } catch (error) {
                showStatus('error', `CSV parsing error: ${error.message}`);
                reject(error);
            }
        };
        
        reader.onerror = () => {
            showStatus('error', 'Error reading file');
            reject(new Error('File read error'));
        };
        
        reader.readAsText(file);
    });
}

// ============================================
// 2. Data Inspection & Preview
// ============================================

/**
 * Show data preview table
 */
function showDataPreview(data, title = 'Train Data Preview') {
    if (!data || !data.raw || data.raw.length === 0) return;
    
    const previewRows = data.raw.slice(0, 10);
    let html = `<h3>${title} (10 rows)</h3>`;
    html += '<table><thead><tr>';
    
    // Headers - use first row keys
    const headers = Object.keys(previewRows[0]);
    headers.forEach(header => {
        html += `<th>${header}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // Rows
    previewRows.forEach(row => {
        html += '<tr>';
        headers.forEach(header => {
            let value = row[header];
            // Truncate long values
            if (value && value.toString().length > 20) {
                value = value.toString().substring(0, 20) + '...';
            }
            html += `<td>${value !== undefined ? value : ''}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    elements.dataPreview.innerHTML = html;
}

/**
 * Inspect data statistics
 */
async function inspectData() {
    if (!trainData) return;
    
    try {
        // Calculate basic statistics
        const stats = {
            totalRows: trainData.raw.length,
            features: FEATURE_COLS,
            survivalRate: null,
            missingValues: {}
        };
        
        // Calculate survival rate
        if (trainData.labels) {
            const survived = trainData.labels.filter(l => l === 1).length;
            stats.survivalRate = (survived / trainData.labels.length * 100).toFixed(1);
        }
        
        // Calculate missing values percentage
        FEATURE_COLS.forEach(feature => {
            const missing = trainData.raw.filter(row => 
                row[feature] === undefined || row[feature] === null || row[feature] === '' || row[feature] === 'NaN'
            ).length;
            stats.missingValues[feature] = ((missing / trainData.raw.length) * 100).toFixed(1);
        });
        
        // Show statistics
        let statsHTML = `<h3>Data Statistics</h3>`;
        statsHTML += `<div class="metric-display">
            <div class="metric-box">
                <div class="metric-value">${stats.totalRows}</div>
                <div class="metric-label">Total Rows</div>
            </div>`;
        
        if (stats.survivalRate) {
            statsHTML += `<div class="metric-box">
                <div class="metric-value">${stats.survivalRate}%</div>
                <div class="metric-label">Survival Rate</div>
            </div>`;
        }
        
        statsHTML += `</div>`;
        
        statsHTML += `<h4>Missing Values (%)</h4><ul>`;
        Object.entries(stats.missingValues).forEach(([feature, percent]) => {
            statsHTML += `<li>${feature}: ${percent}%</li>`;
        });
        statsHTML += `</ul>`;
        
        elements.inspectionCharts.innerHTML = statsHTML;
        
        // Show visualizations if tfvis is available
        if (typeof tfvis !== 'undefined') {
            // Survival by Sex chart
            const sexData = {};
            trainData.raw.forEach(row => {
                if (row.Sex && row.Survived !== undefined) {
                    const sex = row.Sex;
                    if (!sexData[sex]) {
                        sexData[sex] = { survived: 0, total: 0 };
                    }
                    sexData[sex].total++;
                    if (parseInt(row.Survived) === 1) {
                        sexData[sex].survived++;
                    }
                }
            });
            
            const sexChartData = Object.entries(sexData).map(([sex, data]) => ({
                x: sex,
                y: (data.survived / data.total * 100).toFixed(1)
            }));
            
            tfvis.render.barchart(
                { name: 'Survival Rate by Sex (%)', tab: 'Data Inspection' },
                { values: sexChartData },
                { xLabel: 'Sex', yLabel: 'Survival %', height: 300 }
            );
            
            // Survival by Pclass chart
            const pclassData = {};
            trainData.raw.forEach(row => {
                if (row.Pclass && row.Survived !== undefined) {
                    const pclass = `Class ${row.Pclass}`;
                    if (!pclassData[pclass]) {
                        pclassData[pclass] = { survived: 0, total: 0 };
                    }
                    pclassData[pclass].total++;
                    if (parseInt(row.Survived) === 1) {
                        pclassData[pclass].survived++;
                    }
                }
            });
            
            const pclassChartData = Object.entries(pclassData).map(([pclass, data]) => ({
                x: pclass,
                y: (data.survived / data.total * 100).toFixed(1)
            }));
            
            tfvis.render.barchart(
                { name: 'Survival Rate by Passenger Class (%)', tab: 'Data Inspection' },
                { values: pclassChartData },
                { xLabel: 'Passenger Class', yLabel: 'Survival %', height: 300 }
            );
        }
        
    } catch (error) {
        showStatus('error', `Inspection error: ${error.message}`);
    }
}

// ============================================
// 3. Preprocessing Functions
// ============================================

/**
 * Preprocess data: imputation, standardization, one-hot encoding
 */
function preprocessData(data, isTest = false) {
    return tf.tidy(() => {
        const rawData = data.raw;
        
        // Calculate statistics from training data
        const ageValues = rawData
            .map(row => parseFloat(row.Age))
            .filter(val => !isNaN(val) && val !== null);
        
        const fareValues = rawData
            .map(row => parseFloat(row.Fare))
            .filter(val => !isNaN(val) && val !== null);
        
        const ageMedian = ageValues.length > 0 ? 
            ageValues.sort((a, b) => a - b)[Math.floor(ageValues.length / 2)] : 30;
        
        const fareMedian = fareValues.length > 0 ? 
            fareValues.sort((a, b) => a - b)[Math.floor(fareValues.length / 2)] : 32.2;
        
        // Calculate fare mean and std for standardization
        const fareMean = fareValues.reduce((sum, val) => sum + val, 0) / fareValues.length || 0;
        const fareStd = Math.sqrt(
            fareValues.reduce((sum, val) => sum + Math.pow(val - fareMean, 2), 0) / fareValues.length
        ) || 1;
        
        // Calculate age mean and std for standardization
        const ageMean = ageValues.reduce((sum, val) => sum + val, 0) / ageValues.length || 0;
        const ageStd = Math.sqrt(
            ageValues.reduce((sum, val) => sum + Math.pow(val - ageMean, 2), 0) / ageValues.length
        ) || 1;
        
        // Find mode for Embarked
        const embarkedCounts = {};
        rawData.forEach(row => {
            if (row.Embarked && row.Embarked.trim() !== '' && row.Embarked !== 'NaN') {
                embarkedCounts[row.Embarked] = (embarkedCounts[row.Embarked] || 0) + 1;
            }
        });
        const embarkedMode = Object.keys(embarkedCounts).length > 0 ?
            Object.keys(embarkedCounts).reduce((a, b) => embarkedCounts[a] > embarkedCounts[b] ? a : b) : 'S';
        
        // Preprocess each row
        const processedFeatures = [];
        const processedLabels = !isTest ? [] : null;
        
        rawData.forEach(row => {
            // Handle missing values
            let age = parseFloat(row.Age);
            if (isNaN(age) || age === null) age = ageMedian;
            
            let fare = parseFloat(row.Fare);
            if (isNaN(fare) || fare === null) fare = fareMedian;
            
            let embarked = row.Embarked;
            if (!embarked || embarked.trim() === '' || embarked === 'NaN') {
                embarked = embarkedMode;
            }
            
            const sex = row.Sex || 'male';
            const pclass = row.Pclass || '3';
            
            // Standardize Age and Fare
            const ageStdized = (age - ageMean) / ageStd;
            const fareStdized = (fare - fareMean) / fareStd;
            
            // Convert categorical variables to numeric
            const sexEncoded = sex === 'female' ? 1 : 0;
            const pclassEncoded = parseInt(pclass);
            
            // Encode Embarked as numeric (C=0, Q=1, S=2)
            let embarkedEncoded = 2; // Default to S
            if (embarked === 'C') embarkedEncoded = 0;
            else if (embarked === 'Q') embarkedEncoded = 1;
            
            // Create feature vector (7 original features as specified)
            const featureVector = [
                pclassEncoded,      // Pclass (1,2,3)
                sexEncoded,         // Sex (0=male, 1=female)
                ageStdized,         // Standardized Age
                parseInt(row.SibSp || 0), // SibSp
                parseInt(row.Parch || 0), // Parch
                fareStdized,        // Standardized Fare
                embarkedEncoded     // Embarked (0=C, 1=Q, 2=S)
            ];
            
            processedFeatures.push(featureVector);
            
            if (!isTest && row.Survived !== undefined) {
                processedLabels.push(parseInt(row.Survived) || 0);
            }
        });
        
        // Convert to tensors
        const featuresTensor = tf.tensor2d(processedFeatures);
        const labelsTensor = !isTest ? tf.tensor1d(processedLabels) : null;
        
        return {
            features: featuresTensor,
            labels: labelsTensor,
            ids: data.ids,
            preprocessingInfo: {
                ageMean, ageStd, ageMedian,
                fareMean, fareStd, fareMedian,
                embarkedMode
            }
        };
    });
}

// ============================================
// 4. Model Creation with Sigmoid Gates
// ============================================

/**
 * Create model with Sigmoid Feature Gates
 */
function createModel() {
    // Clear previous model
    if (model) {
        model.dispose();
    }
    
    const useGate = elements.useGateToggle.checked;
    
    if (useGate) {
        // Create Sigmoid gate weights variable
        gateWeights = tf.variable(tf.randomUniform([7, 1], -0.1, 0.1));
        
        // Custom layer for sigmoid gate
        class SigmoidGateLayer extends tf.layers.Layer {
            constructor(config) {
                super(config);
                this.gateWeights = gateWeights;
            }
            
            call(inputs) {
                return tf.tidy(() => {
                    const input = inputs;
                    const gateOutput = tf.sigmoid(this.gateWeights);
                    return tf.mul(input, gateOutput);
                });
            }
            
            static get className() {
                return 'SigmoidGateLayer';
            }
        }
        tf.serialization.registerClass(SigmoidGateLayer);
        
        // Build model
        model = tf.sequential();
        model.add(new SigmoidGateLayer({}));
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            kernelInitializer: 'glorotNormal'
        }));
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
        
    } else {
        // Create standard model without gate
        model = tf.sequential();
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [7],
            kernelInitializer: 'glorotNormal'
        }));
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
    }
    
    // Compile model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Show model summary
    console.log('Model Summary:');
    console.log(model.summary());
    
    showStatus('success', `Model created ${useGate ? 'with' : 'without'} Sigmoid Feature Gates`);
    elements.trainBtn.disabled = false;
    elements.exportModelBtn.disabled = false;
    
    return model;
}

// ============================================
// 5. Training Functions
// ============================================

/**
 * Train the model with early stopping
 */
async function trainModel() {
    if (!processedTrainData || !model) {
        showStatus('error', 'No data or model to train');
        return;
    }
    
    isTraining = true;
    elements.trainBtn.disabled = true;
    elements.stopTrainBtn.disabled = false;
    elements.trainingStatus.style.display = 'block';
    elements.trainingStatus.innerHTML = 'Starting training...';
    
    try {
        // Create training and validation sets (80/20 split)
        const dataSize = processedTrainData.features.shape[0];
        const trainSize = Math.floor(dataSize * 0.8);
        
        // Create shuffled indices
        const indices = tf.range(0, dataSize);
        const shuffledIndices = tf.util.createShuffledIndices(dataSize);
        
        const trainIndices = shuffledIndices.slice(0, trainSize);
        const valIndices = shuffledIndices.slice(trainSize);
        
        const trainFeatures = processedTrainData.features.gather(trainIndices);
        const trainLabels = processedTrainData.labels.gather(trainIndices);
        const valFeatures = processedTrainData.features.gather(valIndices);
        const valLabels = processedTrainData.labels.gather(valIndices);
        
        // Training parameters
        const epochs = 50;
        const batchSize = Math.min(32, trainSize);
        
        // Early stopping variables
        let bestValLoss = Infinity;
        let patience = 10;
        let patienceCounter = 0;
        
        trainingHistory = {
            loss: [],
            val_loss: [],
            accuracy: [],
            val_accuracy: []
        };
        
        // Create tfvis surface for training
        const surface = { name: 'Training History', tab: 'Training' };
        
        // Training loop
        for (let epoch = 0; epoch < epochs && isTraining; epoch++) {
            const history = await model.fit(trainFeatures, trainLabels, {
                epochs: 1,
                batchSize: batchSize,
                validationData: [valFeatures, valLabels],
                shuffle: true,
                verbose: 0
            });
            
            const loss = history.history.loss[0];
            const acc = history.history.acc ? history.history.acc[0] : history.history.accuracy[0];
            const valLoss = history.history.val_loss[0];
            const valAcc = history.history.val_acc ? history.history.val_acc[0] : history.history.val_accuracy[0];
            
            trainingHistory.loss.push(loss);
            trainingHistory.accuracy.push(acc);
            trainingHistory.val_loss.push(valLoss);
            trainingHistory.val_accuracy.push(valAcc);
            
            // Update status
            elements.trainingStatus.innerHTML = 
                `Epoch ${epoch + 1}/${epochs} - Loss: ${loss.toFixed(4)}, Acc: ${acc.toFixed(4)}, ` +
                `Val Loss: ${valLoss.toFixed(4)}, Val Acc: ${valAcc.toFixed(4)}`;
            
            // Visualize training progress
            if (typeof tfvis !== 'undefined') {
                const metrics = [
                    { label: 'loss', values: trainingHistory.loss },
                    { label: 'val_loss', values: trainingHistory.val_loss },
                    { label: 'accuracy', values: trainingHistory.accuracy },
                    { label: 'val_accuracy', values: trainingHistory.val_accuracy }
                ];
                
                tfvis.show.history(surface, metrics, ['loss', 'val_loss', 'accuracy', 'val_accuracy']);
            }
            
            // Early stopping check
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                patienceCounter = 0;
            } else {
                patienceCounter++;
                if (patienceCounter >= patience) {
                    showStatus('success', `Early stopping at epoch ${epoch + 1}`);
                    break;
                }
            }
            
            // Allow UI updates
            await tf.nextFrame();
        }
        
        if (isTraining) {
            showStatus('success', 'Training completed successfully');
            elements.evaluateBtn.disabled = false;
            elements.predictBtn.disabled = false;
            elements.showImportanceBtn.disabled = false;
        } else {
            showStatus('info', 'Training stopped by user');
        }
        
    } catch (error) {
        showStatus('error', `Training error: ${error.message}`);
        console.error('Training error:', error);
    } finally {
        isTraining = false;
        elements.trainBtn.disabled = false;
        elements.stopTrainBtn.disabled = true;
        
        // Clean up
        tf.disposeVariables();
    }
}

// ============================================
// 6. Evaluation & Metrics
// ============================================

/**
 * Evaluate model and update metrics table
 */
async function evaluateModel() {
    if (!model || !processedTrainData) {
        showStatus('error', 'No model or data for evaluation');
        return;
    }
    
    try {
        elements.evaluateBtn.disabled = true;
        elements.evaluationTable.innerHTML = '<p>Evaluating model...</p>';
        
        // Create validation set (20% of data)
        const dataSize = processedTrainData.features.shape[0];
        const valSize = Math.floor(dataSize * 0.2);
        const valIndices = tf.range(dataSize - valSize, dataSize);
        
        const valFeatures = processedTrainData.features.gather(valIndices);
        const valLabels = processedTrainData.labels.gather(valIndices);
        
        // Get predictions
        const predictions = model.predict(valFeatures);
        const predValues = await predictions.array();
        const actualValues = await valLabels.array();
        
        predictions.dispose();
        valFeatures.dispose();
        valLabels.dispose();
        
        // Calculate metrics at current threshold
        const threshold = parseFloat(elements.thresholdSlider.value);
        updateEvaluationMetrics(predValues, actualValues, threshold);
        
        // Generate ROC curve data
        const rocData = await calculateROCCurve(predValues, actualValues);
        
        // Plot ROC curve
        if (typeof tfvis !== 'undefined') {
            tfvis.render.linechart(
                { name: 'ROC Curve', tab: 'Evaluation' },
                { values: rocData },
                { xLabel: 'False Positive Rate', yLabel: 'True Positive Rate', height: 400 }
            );
        }
        
        // Show evaluation table
        showEvaluationTable();
        elements.evaluateBtn.disabled = false;
        
    } catch (error) {
        showStatus('error', `Evaluation error: ${error.message}`);
        console.error('Evaluation error:', error);
        elements.evaluateBtn.disabled = false;
    }
}

/**
 * Calculate ROC curve data
 */
async function calculateROCCurve(predictions, labels) {
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    const rocData = [];
    
    for (const threshold of thresholds) {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            const pred = predictions[i][0] >= threshold ? 1 : 0;
            const actual = labels[i];
            
            if (actual === 1 && pred === 1) tp++;
            if (actual === 0 && pred === 1) fp++;
            if (actual === 0 && pred === 0) tn++;
            if (actual === 1 && pred === 0) fn++;
        }
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        rocData.push({ x: fpr, y: tpr });
    }
    
    // Calculate AUC (trapezoidal rule)
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        auc += (rocData[i].x - rocData[i-1].x) * (rocData[i].y + rocData[i-1].y) / 2;
    }
    
    evaluationResults = { rocData, auc };
    return rocData;
}

/**
 * Update evaluation metrics based on threshold
 */
function updateEvaluationMetrics(predictions, labels, threshold) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < predictions.length; i++) {
        const pred = predictions[i][0] >= threshold ? 1 : 0;
        const actual = labels[i];
        
        if (actual === 1 && pred === 1) tp++;
        if (actual === 0 && pred === 1) fp++;
        if (actual === 0 && pred === 0) tn++;
        if (actual === 1 && pred === 0) fn++;
    }
    
    const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    evaluationResults = {
        ...evaluationResults,
        confusionMatrix: { tp, fp, tn, fn },
        metrics: { accuracy, precision, recall, f1, threshold }
    };
}

/**
 * Show evaluation table with metrics
 */
function showEvaluationTable() {
    if (!evaluationResults || !evaluationResults.confusionMatrix) return;
    
    const { confusionMatrix, metrics, auc } = evaluationResults;
    const { tp, fp, tn, fn } = confusionMatrix;
    
    let html = `<h3>Evaluation Metrics</h3>`;
    
    // Metrics display
    html += `<div class="metric-display">
        <div class="metric-box">
            <div class="metric-value">${(metrics.accuracy * 100).toFixed(1)}%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">${(metrics.precision * 100).toFixed(1)}%</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">${(metrics.recall * 100).toFixed(1)}%</div>
            <div class="metric-label">Recall</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">${(metrics.f1 * 100).toFixed(1)}%</div>
            <div class="metric-label">F1-Score</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">${auc ? auc.toFixed(3) : 'N/A'}</div>
            <div class="metric-label">AUC</div>
        </div>
    </div>`;
    
    // Confusion Matrix
    html += `<h3>Confusion Matrix (Threshold: ${metrics.threshold.toFixed(2)})</h3>`;
    html += `<table>
        <thead>
            <tr>
                <th></th>
                <th>Predicted Negative</th>
                <th>Predicted Positive</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Actual Negative</strong></td>
                <td>${tn} (TN)</td>
                <td>${fp} (FP)</td>
            </tr>
            <tr>
                <td><strong>Actual Positive</strong></td>
                <td>${fn} (FN)</td>
                <td>${tp} (TP)</td>
            </tr>
        </tbody>
    </table>`;
    
    // Force DOM update
    setTimeout(() => {
        elements.evaluationTable.innerHTML = html;
    }, 0);
}

// ============================================
// 7. Feature Importance Visualization
// ============================================

/**
 * Extract and display feature importance from Sigmoid gates
 */
async function showFeatureImportance() {
    if (!gateWeights || !elements.useGateToggle.checked) {
        showStatus('error', 'No gate weights available. Enable Sigmoid Gates and train the model.');
        return;
    }
    
    try {
        elements.showImportanceBtn.disabled = true;
        elements.featureImportanceDiv.innerHTML = '<p>Calculating feature importance...</p>';
        
        // Read gate weights
        const weights = await gateWeights.read().array();
        const importanceValues = weights.map(w => w[0]);
        
        // Apply sigmoid to get [0,1] importance scores
        const sigmoidImportance = importanceValues.map(w => 1 / (1 + Math.exp(-w)));
        
        // Create feature importance data
        featureImportance = FEATURE_COLS.map((feature, i) => ({
            feature,
            importance: sigmoidImportance[i],
            rawWeight: importanceValues[i]
        })).sort((a, b) => b.importance - a.importance);
        
        // Display as table
        let html = `<h3>Feature Importance (Sigmoid Gate Weights)</h3>`;
        html += `<p>Gate weights are passed through sigmoid to get importance scores in range [0,1]</p>`;
        html += `<table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Importance [0,1]</th>
                    <th>Raw Weight</th>
                    <th>Rank</th>
                </tr>
            </thead>
            <tbody>`;
        
        featureImportance.forEach((item, index) => {
            const importancePercent = (item.importance * 100).toFixed(1);
            html += `<tr>
                <td><strong>${item.feature}</strong></td>
                <td>${item.importance.toFixed(4)} (${importancePercent}%)</td>
                <td>${item.rawWeight.toFixed(4)}</td>
                <td>${index + 1}</td>
            </tr>`;
        });
        
        html += `</tbody></table>`;
        elements.featureImportanceDiv.innerHTML = html;
        
        // Show bar chart
        if (typeof tfvis !== 'undefined') {
            const values = featureImportance.map(item => ({
                x: item.feature,
                y: item.importance
            }));
            
            tfvis.render.barchart(
                { name: 'Feature Importance', tab: 'Feature Importance' },
                { values },
                { 
                    xLabel: 'Feature', 
                    yLabel: 'Importance Score',
                    yAxisDomain: [0, 1],
                    height: 400 
                }
            );
        }
        
        elements.exportImportanceBtn.disabled = false;
        elements.showImportanceBtn.disabled = false;
        
    } catch (error) {
        showStatus('error', `Feature importance error: ${error.message}`);
        console.error('Feature importance error:', error);
        elements.showImportanceBtn.disabled = false;
    }
}

// ============================================
// 8. Prediction Functions
// ============================================

/**
 * Make predictions on test data
 */
async function makePredictions() {
    if (!model || !processedTestData) {
        showStatus('error', 'No model or test data available');
        return;
    }
    
    try {
        elements.predictionStatus.style.display = 'block';
        elements.predictionStatus.innerHTML = '<p>Making predictions...</p>';
        elements.predictBtn.disabled = true;
        
        const features = processedTestData.features;
        const predictionsTensor = model.predict(features);
        const predValues = await predictionsTensor.array();
        predictionsTensor.dispose();
        
        // Store predictions
        predictions = {
            ids: processedTestData.ids,
            probabilities: predValues.map(p => p[0]),
            predictions: predValues.map(p => p[0] >= 0.5 ? 1 : 0),
            threshold: 0.5
        };
        
        // Show sample predictions
        let html = `<h3>Prediction Results (Sample)</h3>`;
        html += `<p><strong>Total predictions:</strong> ${predictions.ids.length}</p>`;
        html += `<p><strong>Survival rate predicted:</strong> ${(predictions.predictions.filter(p => p === 1).length / predictions.predictions.length * 100).toFixed(1)}%</p>`;
        html += `<table>
            <thead>
                <tr>
                    <th>PassengerId</th>
                    <th>Probability</th>
                    <th>Prediction (≥0.5)</th>
                </tr>
            </thead>
            <tbody>`;
        
        // Show first 10 predictions
        for (let i = 0; i < Math.min(10, predictions.ids.length); i++) {
            html += `<tr>
                <td>${predictions.ids[i]}</td>
                <td>${predictions.probabilities[i].toFixed(4)}</td>
                <td>${predictions.predictions[i]}</td>
            </tr>`;
        }
        
        html += `</tbody></table>`;
        
        if (predictions.ids.length > 10) {
            html += `<p>... and ${predictions.ids.length - 10} more predictions</p>`;
        }
        
        elements.predictionStatus.innerHTML = html;
        elements.exportPredBtn.disabled = false;
        elements.predictBtn.disabled = false;
        
    } catch (error) {
        showStatus('error', `Prediction error: ${error.message}`);
        console.error('Prediction error:', error);
        elements.predictBtn.disabled = false;
    }
}

// ============================================
// 9. Export Functions
// ============================================

/**
 * Export model for download
 */
async function exportModel() {
    if (!model) {
        showStatus('error', 'No model to export');
        return;
    }
    
    try {
        elements.exportStatus.style.display = 'block';
        elements.exportStatus.innerHTML = '<p>Exporting model...</p>';
        
        // Save model locally
        await model.save('downloads://titanic-tfjs-model');
        
        showStatus('success', 'Model exported successfully. Check your downloads folder.');
        
    } catch (error) {
        showStatus('error', `Model export error: ${error.message}`);
        console.error('Model export error:', error);
    }
}

/**
 * Export predictions as CSV
 */
function exportPredictions() {
    if (!predictions) {
        showStatus('error', 'No predictions to export');
        return;
    }
    
    try {
        elements.exportStatus.style.display = 'block';
        elements.exportStatus.innerHTML = '<p>Exporting predictions...</p>';
        
        // Create submission CSV
        let csvContent = 'PassengerId,Survived\n';
        predictions.ids.forEach((id, i) => {
            csvContent += `${id},${predictions.predictions[i]}\n`;
        });
        
        // Create probabilities CSV
        let probContent = 'PassengerId,Probability\n';
        predictions.ids.forEach((id, i) => {
            probContent += `${id},${predictions.probabilities[i].toFixed(6)}\n`;
        });
        
        // Trigger downloads
        downloadFile(csvContent, 'titanic_submission.csv', 'text/csv');
        downloadFile(probContent, 'titanic_probabilities.csv', 'text/csv');
        
        showStatus('success', 'Predictions exported as titanic_submission.csv and titanic_probabilities.csv');
        
    } catch (error) {
        showStatus('error', `Export error: ${error.message}`);
        console.error('Export error:', error);
    }
}

/**
 * Export feature importance as JSON
 */
function exportFeatureImportance() {
    if (!featureImportance) {
        showStatus('error', 'No feature importance data to export');
        return;
    }
    
    try {
        elements.exportStatus.style.display = 'block';
        elements.exportStatus.innerHTML = '<p>Exporting feature importance...</p>';
        
        const exportData = {
            model: 'Titanic Survival Classifier with Sigmoid Gates',
            timestamp: new Date().toISOString(),
            features: featureImportance,
            summary: featureImportance.reduce((acc, item) => {
                acc[item.feature] = item.importance;
                return acc;
            }, {})
        };
        
        const jsonContent = JSON.stringify(exportData, null, 2);
        downloadFile(jsonContent, 'feature_importance.json', 'application/json');
        
        showStatus('success', 'Feature importance exported as feature_importance.json');
        
    } catch (error) {
        showStatus('error', `Export error: ${error.message}`);
        console.error('Export error:', error);
    }
}

/**
 * Utility function to trigger file download
 */
function downloadFile(content, fileName, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ============================================
// 10. UI Utility Functions
// ============================================

/**
 * Show status messages
 */
function showStatus(type, message) {
    const statusDiv = elements.loadStatus; // Use load status for general messages
    
    statusDiv.innerHTML = message;
    statusDiv.className = `status ${type}`;
    statusDiv.style.display = 'block';
    
    // Auto-hide success messages after 5 seconds
    if (type === 'success') {
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 5000);
    }
    
    // Also log to console for debugging
    console.log(`${type.toUpperCase()}: ${message}`);
}

/**
 * Show specific status in designated element
 */
function showSpecificStatus(elementId, type, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = message;
        element.className = `status ${type}`;
        element.style.display = 'block';
    }
}

/**
 * Initialize event listeners
 */
function initEventListeners() {
    // Load data button
    elements.loadDataBtn.addEventListener('click', async () => {
        const trainFile = elements.trainFile.files[0];
        const testFile = elements.testFile.files[0];
        
        if (!trainFile) {
            showStatus('error', 'Please select train.csv file');
            return;
        }
        
        try {
            showStatus('info', 'Loading CSV files...');
            elements.loadDataBtn.disabled = true;
            
            // Load train data
            trainData = await loadCSV(trainFile, false);
            showDataPreview(trainData, 'Train Data Preview');
            
            // Load test data if provided
            if (testFile) {
                testData = await loadCSV(testFile, true);
                showDataPreview(testData, 'Test Data Preview');
            } else {
                showStatus('info', 'No test file provided. You can still train and evaluate the model.');
            }
            
            elements.inspectBtn.disabled = false;
            elements.preprocessBtn.disabled = false;
            elements.loadDataBtn.disabled = false;
            
        } catch (error) {
            showStatus('error', `Load error: ${error.message}`);
            console.error('Load error:', error);
            elements.loadDataBtn.disabled = false;
        }
    });
    
    // Inspect data button
    elements.inspectBtn.addEventListener('click', () => {
        elements.inspectBtn.disabled = true;
        inspectData();
        setTimeout(() => {
            elements.inspectBtn.disabled = false;
        }, 1000);
    });
    
    // Preprocess button
    elements.preprocessBtn.addEventListener('click', () => {
        if (!trainData) {
            showStatus('error', 'No data to preprocess');
            return;
        }
        
        try {
            showStatus('info', 'Preprocessing data...');
            elements.preprocessBtn.disabled = true;
            
            // Preprocess train data
            processedTrainData = preprocessData(trainData, false);
            
            // Preprocess test data if available
            if (testData) {
                processedTestData = preprocessData(testData, true);
            }
            
            const shape = processedTrainData.features.shape;
            showStatus('success', 
                `Preprocessing complete. Features shape: [${shape[0]}, ${shape[1]}]`
            );
            
            elements.createModelBtn.disabled = false;
            elements.preprocessBtn.disabled = false;
            
        } catch (error) {
            showStatus('error', `Preprocessing error: ${error.message}`);
            console.error('Preprocessing error:', error);
            elements.preprocessBtn.disabled = false;
        }
    });
    
    // Create model button
    elements.createModelBtn.addEventListener('click', () => {
        elements.createModelBtn.disabled = true;
        createModel();
        setTimeout(() => {
            elements.createModelBtn.disabled = false;
        }, 500);
    });
    
    // Train model button
    elements.trainBtn.addEventListener('click', () => {
        trainModel();
    });
    
    // Stop training button
    elements.stopTrainBtn.addEventListener('click', () => {
        isTraining = false;
        elements.stopTrainBtn.disabled = true;
        showStatus('info', 'Stopping training...');
    });
    
    // Evaluate button
    elements.evaluateBtn.addEventListener('click', () => {
        evaluateModel();
    });
    
    // Show feature importance button
    elements.showImportanceBtn.addEventListener('click', () => {
        showFeatureImportance();
    });
    
    // Predict button
    elements.predictBtn.addEventListener('click', () => {
        makePredictions();
    });
    
    // Export buttons
    elements.exportModelBtn.addEventListener('click', exportModel);
    elements.exportPredBtn.addEventListener('click', exportPredictions);
    elements.exportImportanceBtn.addEventListener('click', exportFeatureImportance);
    
    // Threshold slider
    elements.thresholdSlider.addEventListener('input', () => {
        const value = parseFloat(elements.thresholdSlider.value);
        elements.thresholdValue.textContent = value.toFixed(2);
        
        // Update evaluation if results exist
        if (evaluationResults && evaluationResults.confusionMatrix) {
            // Recalculate metrics with new threshold
            const { rocData } = evaluationResults;
            if (rocData) {
                // Find the closest point in ROC data for visualization
                const thresholdIndex = Math.round(value * 100);
                if (rocData[thresholdIndex]) {
                    updateEvaluationMetrics(
                        rocData.map(p => [p.y]),
                        Array(rocData.length).fill(0).map((_, i) => i < rocData.length * 0.5 ? 1 : 0),
                        value
                    );
                    showEvaluationTable();
                }
            }
        }
    });
    
    // Use gate toggle
    elements.useGateToggle.addEventListener('change', () => {
        if (model) {
            showStatus('info', 'Sigmoid gates ' + (elements.useGateToggle.checked ? 'enabled' : 'disabled') + 
                '. Create a new model for changes to take effect.');
        }
    });
}

// ============================================
// 11. Memory Management
// ============================================

/**
 * Clean up tensors and memory
 */
function cleanup() {
    if (model) {
        try {
            model.dispose();
        } catch (e) {
            console.warn('Model disposal error:', e);
        }
    }
    if (gateWeights) {
        try {
            gateWeights.dispose();
        } catch (e) {
            console.warn('Gate weights disposal error:', e);
        }
    }
    if (processedTrainData) {
        try {
            processedTrainData.features.dispose();
            processedTrainData.labels.dispose();
        } catch (e) {
            console.warn('Train data disposal error:', e);
        }
    }
    if (processedTestData && processedTestData.features) {
        try {
            processedTestData.features.dispose();
        } catch (e) {
            console.warn('Test data disposal error:', e);
        }
    }
    
    try {
        tf.disposeVariables();
    } catch (e) {
        console.warn('TensorFlow disposal error:', e);
    }
}

// ============================================
// 12. Initialize Application
// ============================================

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Titanic Survival Classifier...');
    
    // Check if TensorFlow.js is loaded
    if (typeof tf === 'undefined') {
        showStatus('error', 'TensorFlow.js failed to load. Please check your internet connection.');
        return;
    }
    
    if (typeof tfvis === 'undefined') {
        showStatus('error', 'TensorFlow.js Vis failed to load. Visualizations will not be available.');
    }
    
    initEventListeners();
    
    // Set up memory cleanup
    window.addEventListener('beforeunload', cleanup);
    window.addEventListener('pagehide', cleanup);
    
    // Initial status
    showStatus('info', 'Ready to load Titanic CSV files. Please select train.csv and optionally test.csv.');
    
    // Enable test file input
    elements.testFile.disabled = false;
    
    console.log('Application initialized successfully.');
});

// ============================================
// 13. Helper Functions
// ============================================

/**
 * Format number with commas
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
