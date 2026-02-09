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
// 1. CSV Loading Functions
// ============================================

/**
 * Load CSV file with proper comma escaping
 */
async function loadCSV(file, isTest = false) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = async function(e) {
            try {
                const csvText = e.target.result;
                
                // Handle CSV parsing with proper escaping
                const lines = csvText.split('\n');
                const headers = lines[0].split(',');
                
                // Validate required columns
                const requiredCols = isTest ? 
                    ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'] :
                    ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'];
                
                for (const col of requiredCols) {
                    if (!headers.includes(col)) {
                        throw new Error(`Missing column: ${col}`);
                    }
                }
                
                // Parse CSV using tf.data.csv with proper configuration
                const columnConfigs = {};
                requiredCols.forEach(col => {
                    columnConfigs[col] = {
                        required: true,
                        dtype: col === 'Survived' || col === 'Pclass' || col === 'Sex' || col === 'Embarked' ? 
                               'string' : 'float32'
                    };
                });
                
                const dataset = tf.data.csv(csvText, {
                    columnConfigs,
                    configuredColumnsOnly: false,
                    hasHeader: true,
                    delimiter: ','
                });
                
                // Convert to array
                const dataArray = await dataset.toArray();
                
                // Extract features and labels
                const data = {
                    raw: dataArray,
                    features: dataArray.map(row => {
                        const features = {};
                        FEATURE_COLS.forEach(col => {
                            features[col] = row[col];
                        });
                        return features;
                    }),
                    ids: dataArray.map(row => row[ID_COL])
                };
                
                if (!isTest) {
                    data.labels = dataArray.map(row => row[TARGET_COL]);
                }
                
                showStatus('success', `${isTest ? 'Test' : 'Train'} data loaded: ${dataArray.length} rows`);
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
    
    // Headers
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
                row[feature] === undefined || row[feature] === null || row[feature] === ''
            ).length;
            stats.missingValues[feature] = ((missing / trainData.raw.length) * 100).toFixed(1);
        });
        
        // Show statistics
        let statsHTML = `<h3>Data Statistics</h3>`;
        statsHTML += `<p><strong>Total Rows:</strong> ${stats.totalRows}</p>`;
        statsHTML += `<p><strong>Features:</strong> ${stats.features.join(', ')}</p>`;
        if (stats.survivalRate) {
            statsHTML += `<p><strong>Survival Rate:</strong> ${stats.survivalRate}%</p>`;
        }
        
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
                    if (row.Survived == 1) {
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
                    if (row.Survived == 1) {
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
        const ageValues = rawData.map(row => parseFloat(row.Age)).filter(val => !isNaN(val));
        const fareValues = rawData.map(row => parseFloat(row.Fare)).filter(val => !isNaN(val));
        
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
            if (row.Embarked && row.Embarked.trim() !== '') {
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
            const age = !isNaN(parseFloat(row.Age)) ? parseFloat(row.Age) : ageMedian;
            const fare = !isNaN(parseFloat(row.Fare)) ? parseFloat(row.Fare) : fareMedian;
            const embarked = row.Embarked && row.Embarked.trim() !== '' ? row.Embarked : embarkedMode;
            const sex = row.Sex || 'male';
            const pclass = row.Pclass || '3';
            
            // Standardize Age and Fare
            const ageStdized = (age - ageMean) / ageStd;
            const fareStdized = (fare - fareMean) / fareStd;
            
            // One-hot encoding for categorical variables
            const sexMale = sex === 'male' ? 1 : 0;
            const sexFemale = sex === 'female' ? 1 : 0;
            
            const pclass1 = pclass === '1' ? 1 : 0;
            const pclass2 = pclass === '2' ? 1 : 0;
            const pclass3 = pclass === '3' ? 1 : 0;
            
            const embarkedC = embarked === 'C' ? 1 : 0;
            const embarkedQ = embarked === 'Q' ? 1 : 0;
            const embarkedS = embarked === 'S' ? 1 : 0;
            
            // Create feature vector (7 original features)
            const featureVector = [
                parseInt(pclass),      // Pclass (1,2,3)
                sex === 'female' ? 1 : 0, // Sex (0=male, 1=female)
                ageStdized,            // Standardized Age
                parseInt(row.SibSp || 0), // SibSp
                parseInt(row.Parch || 0), // Parch
                fareStdized,           // Standardized Fare
                ['C', 'Q', 'S'].indexOf(embarked) // Embarked (0=C, 1=Q, 2=S)
            ];
            
            processedFeatures.push(featureVector);
            
            if (!isTest && row.Survived !== undefined) {
                processedLabels.push(parseInt(row.Survived));
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
        
        // Create Lambda layer for sigmoid gate
        const sigmoidGateLayer = tf.layers.lambda({
            function: (x, gate) => {
                return tf.mul(x, tf.sigmoid(gate));
            },
            functionArgs: { gate: gateWeights }
        });
        
        // Build sequential model with sigmoid gate
        model = tf.sequential();
        model.add(sigmoidGateLayer);
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
    model.summary();
    
    showStatus('success', `Model created ${useGate ? 'with' : 'without'} Sigmoid Feature Gates`);
    elements.trainBtn.disabled = false;
    
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
    
    try {
        // Create training and validation sets (80/20 split)
        const dataSize = processedTrainData.features.shape[0];
        const trainSize = Math.floor(dataSize * 0.8);
        
        const indices = tf.range(0, dataSize);
        const shuffledIndices = indices.gather(tf.util.createShuffledIndices(dataSize));
        
        const trainIndices = shuffledIndices.slice([0], [trainSize]);
        const valIndices = shuffledIndices.slice([trainSize], [dataSize - trainSize]);
        
        const trainFeatures = processedTrainData.features.gather(trainIndices);
        const trainLabels = processedTrainData.labels.gather(trainIndices);
        const valFeatures = processedTrainData.features.gather(valIndices);
        const valLabels = processedTrainData.labels.gather(valIndices);
        
        // Training parameters
        const epochs = 50;
        const batchSize = 32;
        
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
            
            // Visualize training progress
            if (typeof tfvis !== 'undefined') {
                tfvis.show.history(
                    { name: 'Training History', tab: 'Training' },
                    trainingHistory,
                    ['loss', 'val_loss', 'accuracy', 'val_accuracy']
                );
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
    } finally {
        isTraining = false;
        elements.trainBtn.disabled = false;
        elements.stopTrainBtn.disabled = true;
        elements.trainingStatus.style.display = 'none';
        
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
        // Create validation set
        const dataSize = processedTrainData.features.shape[0];
        const valSize = Math.floor(dataSize * 0.2);
        const valIndices = tf.range(dataSize - valSize, dataSize);
        
        const valFeatures = processedTrainData.features.gather(valIndices);
        const valLabels = processedTrainData.labels.gather(valIndices);
        
        // Get predictions
        const predictions = model.predict(valFeatures);
        const predValues = await predictions.array();
        predictions.dispose();
        
        // Calculate metrics at current threshold
        const threshold = parseFloat(elements.thresholdSlider.value);
        updateEvaluationMetrics(predValues, await valLabels.array(), threshold);
        
        // Generate ROC curve data
        const rocData = await calculateROCCurve(predValues, await valLabels.array());
        
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
        
    } catch (error) {
        showStatus('error', `Evaluation error: ${error.message}`);
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
            html += `<tr>
                <td><strong>${item.feature}</strong></td>
                <td>${item.importance.toFixed(4)}</td>
                <td>${item.rawWeight.toFixed(4)}</td>
                <td>${index + 1}</td>
            </tr>`;
        });
        
        html += `</tbody></table>`;
        elements.featureImportanceDiv.innerHTML = html;
        
        // Show bar chart
        if (typeof tfvis !== 'undefined') {
            tfvis.render.barchart(
                { name: 'Feature Importance', tab: 'Feature Importance' },
                { 
                    values: featureImportance.map(item => ({
                        x: item.feature,
                        y: item.importance
                    }))
                },
                { xLabel: 'Feature', yLabel: 'Importance', height: 400 }
            );
        }
        
        elements.exportImportanceBtn.disabled = false;
        
    } catch (error) {
        showStatus('error', `Feature importance error: ${error.message}`);
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
        elements.predictionStatus.innerHTML = 'Making predictions...';
        
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
        
    } catch (error) {
        showStatus('error', `Prediction error: ${error.message}`);
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
        await model.save('downloads://titanic-tfjs-model');
        showStatus('success', 'Model exported successfully. Check your downloads folder.');
    } catch (error) {
        showStatus('error', `Model export error: ${error.message}`);
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
        downloadFile(csvContent, 'submission.csv', 'text/csv');
        downloadFile(probContent, 'probabilities.csv', 'text/csv');
        
        showStatus('success', 'Predictions exported as submission.csv and probabilities.csv');
        
    } catch (error) {
        showStatus('error', `Export error: ${error.message}`);
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
    const statusDivs = [
        elements.loadStatus,
        elements.preprocessStatus,
        elements.modelStatus,
        elements.trainingStatus,
        elements.predictionStatus,
        elements.exportStatus
    ];
    
    statusDivs.forEach(div => {
        if (div.innerHTML.includes(message)) return;
        
        div.innerHTML = message;
        div.className = `status ${type}`;
        div.style.display = 'block';
        
        // Auto-hide success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                div.style.display = 'none';
            }, 5000);
        }
    });
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
            
            // Load train data
            trainData = await loadCSV(trainFile, false);
            showDataPreview(trainData, 'Train Data Preview');
            
            // Load test data if provided
            if (testFile) {
                testData = await loadCSV(testFile, true);
                showDataPreview(testData, 'Test Data Preview');
            }
            
            elements.inspectBtn.disabled = false;
            elements.preprocessBtn.disabled = false;
            
        } catch (error) {
            showStatus('error', `Load error: ${error.message}`);
        }
    });
    
    // Inspect data button
    elements.inspectBtn.addEventListener('click', inspectData);
    
    // Preprocess button
    elements.preprocessBtn.addEventListener('click', () => {
        if (!trainData) {
            showStatus('error', 'No data to preprocess');
            return;
        }
        
        try {
            showStatus('info', 'Preprocessing data...');
            
            // Preprocess train data
            processedTrainData = preprocessData(trainData, false);
            
            // Preprocess test data if available
            if (testData) {
                processedTestData = preprocessData(testData, true);
            }
            
            showStatus('success', 
                `Preprocessing complete. Features shape: ${processedTrainData.features.shape}`
            );
            
            elements.createModelBtn.disabled = false;
            
        } catch (error) {
            showStatus('error', `Preprocessing error: ${error.message}`);
        }
    });
    
    // Create model button
    elements.createModelBtn.addEventListener('click', createModel);
    
    // Train model button
    elements.trainBtn.addEventListener('click', trainModel);
    
    // Stop training button
    elements.stopTrainBtn.addEventListener('click', () => {
        isTraining = false;
        elements.stopTrainBtn.disabled = true;
    });
    
    // Evaluate button
    elements.evaluateBtn.addEventListener('click', evaluateModel);
    
    // Show feature importance button
    elements.showImportanceBtn.addEventListener('click', showFeatureImportance);
    
    // Predict button
    elements.predictBtn.addEventListener('click', makePredictions);
    
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
            updateEvaluationMetrics(
                evaluationResults.rocData ? 
                    evaluationResults.rocData.map(p => [p.y]) : 
                    Array(evaluationResults.confusionMatrix.tp + evaluationResults.confusionMatrix.fn + 
                          evaluationResults.confusionMatrix.tn + evaluationResults.confusionMatrix.fp).fill([0.5]),
                Array(evaluationResults.confusionMatrix.tp + evaluationResults.confusionMatrix.fn).fill(1)
                    .concat(Array(evaluationResults.confusionMatrix.tn + evaluationResults.confusionMatrix.fp).fill(0)),
                value
            );
            showEvaluationTable();
        }
    });
    
    // Use gate toggle
    elements.useGateToggle.addEventListener('change', () => {
        if (model) {
            showStatus('info', 'Model will use sigmoid gates: ' + elements.useGateToggle.checked);
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
    if (model) model.dispose();
    if (gateWeights) gateWeights.dispose();
    if (processedTrainData) {
        processedTrainData.features.dispose();
        processedTrainData.labels.dispose();
    }
    if (processedTestData && processedTestData.features) {
        processedTestData.features.dispose();
    }
    
    tf.disposeVariables();
}

// ============================================
// 12. Initialize Application
// ============================================

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    
    // Set up memory cleanup
    window.addEventListener('beforeunload', cleanup);
    window.addEventListener('pagehide', cleanup);
    
    // Initial status
    showStatus('info', 'Ready to load Titanic CSV files. Please select train.csv and optionally test.csv.');
});

// ============================================
// 13. Helper Functions
// ============================================

/**
 * Validate CSV structure
 */
function validateCSVStructure(headers, isTest) {
    const required = isTest ? 
        ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'] :
        ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'];
    
    return required.every(col => headers.includes(col));
}

/**
 * Format number with commas
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
