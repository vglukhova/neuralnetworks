// Titanic Binary Classifier - TensorFlow.js
// Complete browser-based implementation

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

let rawTrainData = null;
let rawTestData = null;
let processedTrainData = null;
let processedTestData = null;
let trainTensors = null;
let testTensors = null;
let model = null;
let trainingHistory = null;
let featureInfo = null;
let featureNames = [];
let featureImportance = null;
let firstLayerWeights = null;
let validationProbs = null;
let validationLabels = null;
let testPredictions = null;
let testProbabilities = null;
let testPassengerIds = null;

// Feature flags
let includeFamilySize = true;
let includeIsAlone = true;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Simple CSV parser
 */
function parseCSV(text) {
    const rows = [];
    let currentRow = [];
    let currentCell = '';
    let insideQuotes = false;
    
    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        
        if (char === '"') {
            insideQuotes = !insideQuotes;
        } else if (char === ',' && !insideQuotes) {
            currentRow.push(currentCell);
            currentCell = '';
        } else if (char === '\n' && !insideQuotes) {
            currentRow.push(currentCell);
            rows.push(currentRow);
            currentRow = [];
            currentCell = '';
        } else {
            currentCell += char;
        }
    }
    
    // Last row
    if (currentCell !== '' || currentRow.length > 0) {
        currentRow.push(currentCell);
        rows.push(currentRow);
    }
    
    return rows;
}

/**
 * Update status message
 */
function updateStatus(elementId, message, type = 'normal') {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.textContent = message;
    element.className = 'status';
    
    if (type === 'success') {
        element.classList.add('success');
    } else if (type === 'error') {
        element.classList.add('error');
    } else if (type === 'warning') {
        element.classList.add('warning');
    }
}

/**
 * Create HTML table
 */
function createTable(headers, data, maxRows = 10) {
    let html = '<table class="evaluation-table">';
    
    // Headers
    html += '<thead><tr>';
    headers.forEach(header => {
        html += `<th>${header}</th>`;
    });
    html += '</tr></thead>';
    
    // Body
    html += '<tbody>';
    data.slice(0, maxRows).forEach(row => {
        html += '<tr>';
        row.forEach(cell => {
            html += `<td>${cell}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody>';
    
    if (data.length > maxRows) {
        html += `<tfoot><tr><td colspan="${headers.length}" style="text-align: center;">Showing ${maxRows} of ${data.length} rows</td></tr></tfoot>`;
    }
    
    html += '</table>';
    return html;
}

/**
 * Download CSV file
 */
function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Export CSV with proper quoting
 */
function exportCSV(headers, data, filename) {
    let csv = headers.map(h => `"${h}"`).join(',') + '\n';
    
    data.forEach(row => {
        const escapedRow = row.map(val => {
            const strVal = String(val);
            return `"${strVal.replace(/"/g, '""')}"`;
        });
        csv += escapedRow.join(',') + '\n';
    });
    
    downloadCSV(csv, filename);
}

/**
 * Calculate metrics
 */
function calculateMetrics(tp, fp, fn, tn) {
    const accuracy = (tp + tn) / (tp + fp + fn + tn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    return {
        accuracy: accuracy.toFixed(4),
        precision: precision.toFixed(4),
        recall: recall.toFixed(4),
        f1: f1.toFixed(4)
    };
}

// ============================================================================
// DATA LOADING & INSPECTION
// ============================================================================

/**
 * Load CSV file
 */
async function loadCSVFile(fileInput) {
    return new Promise((resolve, reject) => {
        if (!fileInput.files || fileInput.files.length === 0) {
            reject(new Error('No file selected'));
            return;
        }
        
        const file = fileInput.files[0];
        const reader = new FileReader();
        
        reader.onload = function(event) {
            try {
                const text = event.target.result;
                const rows = parseCSV(text);
                
                if (rows.length < 2) {
                    reject(new Error('CSV file is empty'));
                    return;
                }
                
                const headers = rows[0];
                const data = rows.slice(1);
                resolve({ headers, data });
            } catch (error) {
                reject(new Error('Failed to parse CSV'));
            }
        };
        
        reader.onerror = function() {
            reject(new Error('Failed to read file'));
        };
        
        reader.readAsText(file);
    });
}

/**
 * Load and inspect data
 */
async function loadAndInspectData() {
    try {
        updateStatus('dataStatus', 'Loading CSV files...', 'normal');
        
        // Load train data
        const trainFileInput = document.getElementById('trainFile');
        const trainResult = await loadCSVFile(trainFileInput);
        rawTrainData = trainResult;
        
        // Load test data
        const testFileInput = document.getElementById('testFile');
        const testResult = await loadCSVFile(testFileInput);
        rawTestData = testResult;
        
        // Validate
        if (!rawTrainData.headers.includes('Survived')) {
            throw new Error('Train CSV must contain "Survived" column');
        }
        
        // Show preview
        document.getElementById('dataPreview').style.display = 'block';
        
        // Train preview
        const trainPreviewRows = rawTrainData.data.slice(0, 5).map(row => {
            const passengerIdIdx = rawTrainData.headers.indexOf('PassengerId');
            const survivedIdx = rawTrainData.headers.indexOf('Survived');
            const pclassIdx = rawTrainData.headers.indexOf('Pclass');
            const nameIdx = rawTrainData.headers.indexOf('Name');
            const sexIdx = rawTrainData.headers.indexOf('Sex');
            const ageIdx = rawTrainData.headers.indexOf('Age');
            
            return [
                row[passengerIdIdx] || '',
                row[survivedIdx] || '',
                row[pclassIdx] || '',
                (row[nameIdx] || '').substring(0, 15) + '...',
                row[sexIdx] || '',
                row[ageIdx] || ''
            ];
        });
        
        document.getElementById('trainPreview').innerHTML = createTable(
            ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age'],
            trainPreviewRows,
            5
        );
        
        // Test preview
        const testPreviewRows = rawTestData.data.slice(0, 5).map(row => {
            const passengerIdIdx = rawTestData.headers.indexOf('PassengerId');
            const pclassIdx = rawTestData.headers.indexOf('Pclass');
            const nameIdx = rawTestData.headers.indexOf('Name');
            const sexIdx = rawTestData.headers.indexOf('Sex');
            const ageIdx = rawTestData.headers.indexOf('Age');
            
            return [
                row[passengerIdIdx] || '',
                row[pclassIdx] || '',
                (row[nameIdx] || '').substring(0, 15) + '...',
                row[sexIdx] || '',
                row[ageIdx] || ''
            ];
        });
        
        document.getElementById('testPreview').innerHTML = createTable(
            ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age'],
            testPreviewRows,
            5
        );
        
        // Stats
        const trainRows = rawTrainData.data.length;
        const testRows = rawTestData.data.length;
        const trainCols = rawTrainData.headers.length;
        const testCols = rawTestData.headers.length;
        
        const survivedIdx = rawTrainData.headers.indexOf('Survived');
        const totalSurvived = rawTrainData.data.filter(row => row[survivedIdx] === '1').length;
        const survivalRate = (totalSurvived / trainRows * 100).toFixed(1);
        
        updateStatus('dataStatus', 
            `✅ Data loaded! Train: ${trainRows} rows, Test: ${testRows} rows. Survival rate: ${survivalRate}%`,
            'success'
        );
        
        // Enable preprocessing
        document.getElementById('preprocessBtn').disabled = false;
        
    } catch (error) {
        updateStatus('dataStatus', `❌ Error: ${error.message}`, 'error');
        console.error('Load error:', error);
    }
}

// ============================================================================
// DATA PREPROCESSING
// ============================================================================

/**
 * Extract features
 */
function extractFeatures(data, headers, isTraining = true) {
    // Get indices
    const passengerIdIdx = headers.indexOf('PassengerId');
    const survivedIdx = isTraining ? headers.indexOf('Survived') : -1;
    const pclassIdx = headers.indexOf('Pclass');
    const sexIdx = headers.indexOf('Sex');
    const ageIdx = headers.indexOf('Age');
    const sibSpIdx = headers.indexOf('SibSp');
    const parchIdx = headers.indexOf('Parch');
    const fareIdx = headers.indexOf('Fare');
    const embarkedIdx = headers.indexOf('Embarked');
    
    const features = [];
    const targets = [];
    const passengerIds = [];
    
    // Collect for stats
    const ages = [];
    const fares = [];
    const embarkedValues = [];
    
    for (const row of data) {
        passengerIds.push(row[passengerIdIdx]);
        
        if (isTraining && survivedIdx !== -1) {
            targets.push(parseInt(row[survivedIdx]) || 0);
        }
        
        const age = parseFloat(row[ageIdx]);
        if (!isNaN(age)) ages.push(age);
        
        const fare = parseFloat(row[fareIdx]);
        if (!isNaN(fare)) fares.push(fare);
        
        const embarked = row[embarkedIdx];
        if (embarked && embarked.trim() !== '') {
            embarkedValues.push(embarked.trim());
        }
    }
    
    // Calculate stats
    const medianAge = ages.length > 0 ? 
        ages.sort((a, b) => a - b)[Math.floor(ages.length / 2)] : 30;
    
    const medianFare = fares.length > 0 ? 
        fares.sort((a, b) => a - b)[Math.floor(fares.length / 2)] : 32;
    
    const embarkedCounts = {};
    embarkedValues.forEach(val => {
        embarkedCounts[val] = (embarkedCounts[val] || 0) + 1;
    });
    const modeEmbarked = Object.keys(embarkedCounts).length > 0 ?
        Object.keys(embarkedCounts).reduce((a, b) => embarkedCounts[a] > embarkedCounts[b] ? a : b) : 'S';
    
    const meanAge = ages.reduce((a, b) => a + b, 0) / ages.length || medianAge;
    const stdAge = Math.sqrt(ages.reduce((sq, n) => sq + Math.pow(n - meanAge, 2), 0) / ages.length) || 10;
    
    const meanFare = fares.reduce((a, b) => a + b, 0) / fares.length || medianFare;
    const stdFare = Math.sqrt(fares.reduce((sq, n) => sq + Math.pow(n - meanFare, 2), 0) / fares.length) || 20;
    
    // Process rows
    for (const row of data) {
        const featureRow = [];
        
        // Pclass (one-hot)
        const pclass = parseInt(row[pclassIdx]) || 1;
        featureRow.push(pclass === 1 ? 1 : 0);
        featureRow.push(pclass === 2 ? 1 : 0);
        featureRow.push(pclass === 3 ? 1 : 0);
        
        // Sex
        const sex = (row[sexIdx] || '').toLowerCase();
        featureRow.push(sex === 'female' ? 1 : 0);
        
        // Age (standardized)
        let age = parseFloat(row[ageIdx]);
        if (isNaN(age)) age = medianAge;
        featureRow.push((age - meanAge) / stdAge);
        
        // SibSp and Parch
        const sibSp = parseInt(row[sibSpIdx]) || 0;
        const parch = parseInt(row[parchIdx]) || 0;
        featureRow.push(sibSp);
        featureRow.push(parch);
        
        // Engineered features
        if (includeFamilySize) {
            featureRow.push(sibSp + parch + 1);
        }
        
        if (includeIsAlone) {
            featureRow.push((sibSp === 0 && parch === 0) ? 1 : 0);
        }
        
        // Fare (standardized)
        let fare = parseFloat(row[fareIdx]);
        if (isNaN(fare)) fare = medianFare;
        featureRow.push((fare - meanFare) / stdFare);
        
        // Embarked (one-hot)
        let embarked = (row[embarkedIdx] || '').trim();
        if (embarked === '') embarked = modeEmbarked;
        featureRow.push(embarked === 'C' ? 1 : 0);
        featureRow.push(embarked === 'Q' ? 1 : 0);
        featureRow.push(embarked === 'S' ? 1 : 0);
        
        features.push(featureRow);
    }
    
    // Feature names
    const baseFeatureNames = [
        'Pclass_1', 'Pclass_2', 'Pclass_3',
        'Sex_female',
        'Age_std',
        'SibSp', 'Parch'
    ];
    
    if (includeFamilySize) baseFeatureNames.push('FamilySize');
    if (includeIsAlone) baseFeatureNames.push('IsAlone');
    
    baseFeatureNames.push('Fare_std', 'Embarked_C', 'Embarked_Q', 'Embarked_S');
    
    return {
        features,
        targets: isTraining ? targets : null,
        passengerIds,
        stats: {
            meanAge, stdAge, meanFare, stdFare,
            medianAge, medianFare, modeEmbarked
        },
        featureNames: baseFeatureNames
    };
}

/**
 * Preprocess data
 */
function preprocessData() {
    try {
        updateStatus('preprocessStatus', 'Preprocessing data...', 'normal');
        
        if (!rawTrainData || !rawTestData) {
            throw new Error('Please load data first');
        }
        
        // Process train data
        const trainResult = extractFeatures(rawTrainData.data, rawTrainData.headers, true);
        processedTrainData = trainResult;
        
        // Process test data
        const testResult = extractFeatures(rawTestData.data, rawTestData.headers, false);
        processedTestData = testResult;
        
        // Store info
        featureInfo = {
            names: trainResult.featureNames,
            stats: trainResult.stats,
            trainShape: [trainResult.features.length, trainResult.features[0].length],
            testShape: [testResult.features.length, testResult.features[0].length]
        };
        
        featureNames = trainResult.featureNames;
        
        // Convert to tensors
        trainTensors = {
            features: tf.tensor2d(trainResult.features),
            targets: tf.tensor1d(trainResult.targets),
            passengerIds: trainResult.passengerIds
        };
        
        testTensors = {
            features: tf.tensor2d(testResult.features),
            passengerIds: testResult.passengerIds
        };
        
        // Show info
        document.getElementById('preprocessInfo').style.display = 'block';
        
        const featureInfoHTML = `
            <p><strong>Features (${featureInfo.trainShape[1]}):</strong> ${featureInfo.names.join(', ')}</p>
            <p><strong>Training:</strong> ${featureInfo.trainShape[0]} samples</p>
            <p><strong>Test:</strong> ${featureInfo.testShape[0]} samples</p>
        `;
        
        document.getElementById('featureInfo').innerHTML = featureInfoHTML;
        
        updateStatus('preprocessStatus', 
            `✅ Preprocessing complete! ${featureInfo.trainShape[1]} features created.`,
            'success'
        );
        
        // Enable model creation
        document.getElementById('createModelBtn').disabled = false;
        document.getElementById('toggleFeatureBtn').disabled = false;
        
    } catch (error) {
        updateStatus('preprocessStatus', `❌ Error: ${error.message}`, 'error');
    }
}

/**
 * Toggle features
 */
function toggleFeatures() {
    includeFamilySize = !includeFamilySize;
    includeIsAlone = !includeIsAlone;
    preprocessData();
}

// ============================================================================
// MODEL CREATION
// ============================================================================

/**
 * Create model
 */
function createModel() {
    try {
        updateStatus('modelStatus', 'Creating model...', 'normal');
        
        if (!processedTrainData) {
            throw new Error('Please preprocess data first');
        }
        
        const numFeatures = processedTrainData.features[0].length;
        
        // Create model
        model = tf.sequential();
        
        // Hidden layer
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [numFeatures],
            name: 'hidden_layer'
        }));
        
        // Output layer
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            name: 'output_layer'
        }));
        
        // Compile
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Show summary
        document.getElementById('modelSummary').style.display = 'block';
        
        let summaryHTML = '<p><strong>Model:</strong></p><ul>';
        model.layers.forEach((layer, i) => {
            summaryHTML += `<li>Layer ${i+1}: ${layer.name} - ${layer.outputShape}</li>`;
        });
        summaryHTML += '</ul>';
        
        document.getElementById('modelLayers').innerHTML = summaryHTML;
        
        updateStatus('modelStatus', 
            `✅ Model created! ${numFeatures} features → 16 hidden → 1 output`,
            'success'
        );
        
        // Enable training
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('summaryBtn').disabled = false;
        
    } catch (error) {
        updateStatus('modelStatus', `❌ Error: ${error.message}`, 'error');
    }
}

/**
 * Show model summary
 */
function showModelSummary() {
    if (!model) {
        alert('Please create the model first');
        return;
    }
    
    tfvis.show.modelSummary({name: 'Model Summary', tab: 'Model'}, model);
}

// ============================================================================
// TRAINING
// ============================================================================

let isTraining = false;
let stopTrainingRequested = false;

/**
 * Train model
 */
async function trainModel() {
    try {
        if (!model || !trainTensors) {
            throw new Error('Please create model first');
        }
        
        updateStatus('trainingStatus', 'Starting training...', 'normal');
        document.getElementById('trainingCharts').style.display = 'block';
        
        isTraining = true;
        stopTrainingRequested = false;
        
        document.getElementById('trainBtn').disabled = true;
        document.getElementById('stopTrainingBtn').disabled = false;
        
        // Create validation split
        const indices = tf.range(0, trainTensors.features.shape[0]);
        const shuffled = tf.util.createShuffledIndices(indices.size);
        
        const splitIdx = Math.floor(indices.size * 0.8);
        const trainIndices = shuffled.slice(0, splitIdx);
        const valIndices = shuffled.slice(splitIdx);
        
        // Extract data
        const trainFeatures = tf.gather(trainTensors.features, trainIndices);
        const trainTargets = tf.gather(trainTensors.targets, trainIndices);
        const valFeatures = tf.gather(trainTensors.features, valIndices);
        const valTargets = tf.gather(trainTensors.targets, valIndices);
        
        // Store validation data
        validationLabels = await valTargets.array();
        
        // Callbacks
        const metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy'];
        const container = {
            name: 'Training Metrics',
            tab: 'Training',
            styles: { height: '300px' }
        };
        
        const fitCallbacks = tfvis.show.fitCallbacks(container, metrics, {
            callbacks: ['onBatchEnd', 'onEpochEnd']
        });
        
        // Progress callback
        const progressCallback = {
            onEpochEnd: (epoch, logs) => {
                const progress = (epoch + 1) / 50 * 100;
                document.getElementById('trainingProgress').style.width = `${progress}%`;
                
                if (stopTrainingRequested) {
                    model.stopTraining = true;
                }
            }
        };
        
        // Train
        const history = await model.fit(trainFeatures, trainTargets, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valTargets],
            callbacks: [fitCallbacks, progressCallback],
            verbose: 0
        });
        
        trainingHistory = history;
        
        // Get validation predictions
        validationProbs = await model.predict(valFeatures).array();
        
        // Get first layer weights
        const hiddenLayer = model.getLayer('hidden_layer');
        if (hiddenLayer) {
            const weights = hiddenLayer.getWeights();
            if (weights.length > 0) {
                firstLayerWeights = weights[0];
            }
        }
        
        // Clean up
        tf.dispose([trainFeatures, trainTargets, valFeatures, valTargets, indices]);
        
        if (stopTrainingRequested) {
            updateStatus('trainingStatus', 'Training stopped early', 'warning');
        } else {
            const finalAcc = (history.history.acc[history.history.acc.length-1] * 100).toFixed(1);
            const valAcc = (history.history.val_acc[history.history.val_acc.length-1] * 100).toFixed(1);
            updateStatus('trainingStatus', 
                `✅ Training complete! Accuracy: ${finalAcc}%, Val: ${valAcc}%`,
                'success'
            );
        }
        
        // Enable evaluation
        document.getElementById('evaluateBtn').disabled = false;
        document.getElementById('rocBtn').disabled = false;
        document.getElementById('thresholdSlider').disabled = false;
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('saveModelBtn').disabled = false;
        
    } catch (error) {
        updateStatus('trainingStatus', `❌ Error: ${error.message}`, 'error');
    } finally {
        isTraining = false;
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('stopTrainingBtn').disabled = true;
        document.getElementById('trainingProgress').style.width = '0%';
    }
}

/**
 * Stop training
 */
function stopTraining() {
    if (isTraining) {
        stopTrainingRequested = true;
        updateStatus('trainingStatus', 'Stopping training...', 'warning');
    }
}

// ============================================================================
// EVALUATION & FEATURE IMPORTANCE
// ============================================================================

/**
 * Calculate ROC
 */
function calculateROC() {
    if (!validationProbs || !validationLabels) return;
    
    const paired = validationProbs.map((prob, i) => ({
        prob: prob[0],
        label: validationLabels[i]
    })).sort((a, b) => b.prob - a.prob);
    
    const thresholds = Array.from({length: 101}, (_, i) => i / 100);
    const tprs = [];
    const fprs = [];
    
    let area = 0;
    let prevFPR = 0;
    let prevTPR = 0;
    
    thresholds.forEach(threshold => {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        paired.forEach(item => {
            const prediction = item.prob >= threshold ? 1 : 0;
            if (item.label === 1) {
                if (prediction === 1) tp++;
                else fn++;
            } else {
                if (prediction === 1) fp++;
                else tn++;
            }
        });
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        tprs.push(tpr);
        fprs.push(fpr);
        
        if (fpr > prevFPR) {
            area += (fpr - prevFPR) * (tpr + prevTPR) / 2;
            prevFPR = fpr;
            prevTPR = tpr;
        }
    });
    
    return {
        fpr: fprs,
        tpr: tprs,
        auc: area.toFixed(4)
    };
}

/**
 * Show ROC curve
 */
function showROCCurve() {
    try {
        if (!validationProbs || !validationLabels) {
            throw new Error('Please train model first');
        }
        
        const rocData = calculateROC();
        
        const rocSeries = [{
            values: rocData.fpr.map((fpr, i) => ({x: fpr, y: rocData.tpr[i]})),
            series: 'ROC Curve'
        }];
        
        const container = {
            name: 'ROC Curve',
            tab: 'Evaluation',
            styles: { height: '300px' }
        };
        
        tfvis.render.linechart(container, rocSeries, {
            xLabel: 'False Positive Rate',
            yLabel: 'True Positive Rate',
            width: 400,
            height: 300
        });
        
        updateStatus('evaluationStatus', `ROC Curve displayed. AUC: ${rocData.auc}`, 'success');
        
    } catch (error) {
        updateStatus('evaluationStatus', `❌ Error: ${error.message}`, 'error');
    }
}

/**
 * Evaluate model
 */
function evaluateModel(threshold = 0.5) {
    try {
        if (!validationProbs || !validationLabels) {
            throw new Error('Please train model first');
        }
        
        // Update threshold
        document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
        
        // Calculate confusion matrix
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (let i = 0; i < validationProbs.length; i++) {
            const prediction = validationProbs[i][0] >= threshold ? 1 : 0;
            const actual = validationLabels[i];
            
            if (actual === 1) {
                if (prediction === 1) tp++;
                else fn++;
            } else {
                if (prediction === 1) fp++;
                else tn++;
            }
        }
        
        // Calculate metrics
        const metrics = calculateMetrics(tp, fp, fn, tn);
        
        // Display confusion matrix
        const confusionMatrixHTML = `
            <table class="evaluation-table">
                <tr>
                    <th></th>
                    <th>Predicted Negative</th>
                    <th>Predicted Positive</th>
                </tr>
                <tr>
                    <th>Actual Negative</th>
                    <td>${tn}</td>
                    <td>${fp}</td>
                </tr>
                <tr>
                    <th>Actual Positive</th>
                    <td>${fn}</td>
                    <td>${tp}</td>
                </tr>
            </table>
        `;
        
        document.getElementById('confusionMatrix').innerHTML = `
            <h4>Confusion Matrix (Threshold: ${threshold.toFixed(2)})</h4>
            ${confusionMatrixHTML}
        `;
        
        // Display metrics
        const metricsTableHTML = `
            <table class="evaluation-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td>${metrics.accuracy}</td>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td>${metrics.precision}</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td>${metrics.recall}</td>
                </tr>
                <tr>
                    <td>F1-Score</td>
                    <td>${metrics.f1}</td>
                </tr>
            </table>
        `;
        
        document.getElementById('metricsTable').innerHTML = `
            <h4>Performance Metrics</h4>
            ${metricsTableHTML}
        `;
        
        // Calculate feature importance
        if (firstLayerWeights && featureInfo) {
            calculateFeatureImportance();
        }
        
    } catch (error) {
        updateStatus('evaluationStatus', `❌ Error: ${error.message}`, 'error');
    }
}

/**
 * Calculate feature importance
 */
async function calculateFeatureImportance() {
    try {
        if (!firstLayerWeights || !featureInfo || !trainTensors) {
            return;
        }
        
        // Get feature data
        const featureData = await trainTensors.features.array();
        const numFeatures = featureData[0].length;
        
        // Calculate ranges
        const featureRanges = [];
        for (let i = 0; i < numFeatures; i++) {
            const values = featureData.map(row => row[i]);
            const min = Math.min(...values);
            const max = Math.max(...values);
            featureRanges.push(max - min);
        }
        
        // Get weights
        const weights = await firstLayerWeights.array();
        
        // Calculate importance
        const importances = [];
        for (let i = 0; i < numFeatures; i++) {
            let weightedSum = 0;
            
            for (let j = 0; j < weights[i].length; j++) {
                weightedSum += Math.abs(weights[i][j]) * (featureRanges[i] > 1 ? 1 : featureRanges[i]);
            }
            
            const importance = 1 / (1 + Math.exp(-weightedSum));
            importances.push({
                feature: featureInfo.names[i] || `Feature_${i}`,
                importance: importance
            });
        }
        
        // Sort
        importances.sort((a, b) => b.importance - a.importance);
        featureImportance = importances;
        
        // Display top 5
        const topFeatures = importances.slice(0, 5);
        
        let importanceHTML = '<table class="evaluation-table">';
        importanceHTML += '<tr><th>Rank</th><th>Feature</th><th>Importance</th><th>Bar</th></tr>';
        
        topFeatures.forEach((item, index) => {
            const percentage = (item.importance * 100).toFixed(1);
            importanceHTML += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${item.feature}</td>
                    <td>${item.importance.toFixed(4)}</td>
                    <td>
                        <div style="display: flex; align-items: center;">
                            <div style="width: 150px; background: #eee; height: 20px; border-radius: 3px; margin-right: 10px;">
                                <div class="feature-importance-bar" style="width: ${percentage}%"></div>
                            </div>
                            <span>${percentage}%</span>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        importanceHTML += '</table>';
        document.getElementById('featureImportance').innerHTML = importanceHTML;
        
    } catch (error) {
        console.error('Feature importance error:', error);
        document.getElementById('featureImportance').innerHTML = 
            `<p>Error calculating feature importance</p>`;
    }
}

// ============================================================================
// PREDICTION & EXPORT
// ============================================================================

/**
 * Predict test data
 */
async function predictTestData() {
    try {
        updateStatus('predictionStatus', 'Generating predictions...', 'normal');
        
        if (!model || !testTensors) {
            throw new Error('Please train model first');
        }
        
        // Generate predictions
        const probsTensor = model.predict(testTensors.features);
        testProbabilities = await probsTensor.array();
        
        // Convert to binary
        const threshold = parseFloat(document.getElementById('thresholdValue').textContent);
        testPredictions = testProbabilities.map(prob => prob[0] >= threshold ? 1 : 0);
        testPassengerIds = testTensors.passengerIds;
        
        // Clean up
        tf.dispose(probsTensor);
        
        // Show preview
        document.getElementById('predictionPreview').style.display = 'block';
        
        const previewRows = [];
        for (let i = 0; i < Math.min(10, testPassengerIds.length); i++) {
            previewRows.push([
                testPassengerIds[i],
                testPredictions[i],
                testProbabilities[i][0].toFixed(4)
            ]);
        }
        
        document.getElementById('submissionPreview').innerHTML = createTable(
            ['PassengerId', 'Survived', 'Probability'],
            previewRows,
            10
        );
        
        const survivalRate = (testPredictions.filter(p => p === 1).length / testPredictions.length * 100).toFixed(1);
        updateStatus('predictionStatus', 
            `✅ Predictions generated! ${testPredictions.length} samples. Survival rate: ${survivalRate}%`,
            'success'
        );
        
        // Enable export
        document.getElementById('exportBtn').disabled = false;
        document.getElementById('exportProbsBtn').disabled = false;
        
    } catch (error) {
        updateStatus('predictionStatus', `❌ Error: ${error.message}`, 'error');
    }
}

/**
 * Export submission
 */
function exportSubmission() {
    try {
        if (!testPredictions || !testPassengerIds) {
            throw new Error('Please generate predictions first');
        }
        
        const data = testPassengerIds.map((id, i) => [id, testPredictions[i]]);
        exportCSV(['PassengerId', 'Survived'], data, 'submission.csv');
        
        updateStatus('predictionStatus', '✅ Submission CSV exported!', 'success');
        
    } catch (error) {
        updateStatus('predictionStatus', `❌ Error: ${error.message}`, 'error');
    }
}

/**
 * Export probabilities
 */
function exportProbabilities() {
    try {
        if (!testProbabilities || !testPassengerIds) {
            throw new Error('Please generate predictions first');
        }
        
        const data = testPassengerIds.map((id, i) => [id, testProbabilities[i][0].toFixed(6)]);
        exportCSV(['PassengerId', 'Probability'], data, 'probabilities.csv');
        
        updateStatus('predictionStatus', '✅ Probabilities CSV exported!', 'success');
        
    } catch (error) {
        updateStatus('predictionStatus', `❌ Error: ${error.message}`, 'error');
    }
}

/**
 * Save model
 */
async function saveModel() {
    try {
        if (!model) {
            throw new Error('No model to save');
        }
        
        updateStatus('predictionStatus', 'Saving model...', 'normal');
        await model.save('downloads://titanic-tfjs-model');
        updateStatus('predictionStatus', '✅ Model saved!', 'success');
        
    } catch (error) {
        updateStatus('predictionStatus', `❌ Error: ${error.message}`, 'error');
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize app
 */
function initializeApp() {
    console.log('App initializing...');
    
    // Set up event listeners
    document.getElementById('loadDataBtn').addEventListener('click', loadAndInspectData);
    document.getElementById('preprocessBtn').addEventListener('click', preprocessData);
    document.getElementById('toggleFeatureBtn').addEventListener('click', toggleFeatures);
    document.getElementById('createModelBtn').addEventListener('click', createModel);
    document.getElementById('summaryBtn').addEventListener('click', showModelSummary);
    document.getElementById('trainBtn').addEventListener('click', trainModel);
    document.getElementById('stopTrainingBtn').addEventListener('click', stopTraining);
    document.getElementById('evaluateBtn').addEventListener('click', () => evaluateModel(0.5));
    document.getElementById('rocBtn').addEventListener('click', showROCCurve);
    document.getElementById('predictBtn').addEventListener('click', predictTestData);
    document.getElementById('exportBtn').addEventListener('click', exportSubmission);
    document.getElementById('exportProbsBtn').addEventListener('click', exportProbabilities);
    document.getElementById('saveModelBtn').addEventListener('click', saveModel);
    
    // Threshold slider
    const thresholdSlider = document.getElementById('thresholdSlider');
    thresholdSlider.addEventListener('input', function() {
        evaluateModel(parseFloat(this.value));
    });
    
    // Initial status
    updateStatus('dataStatus', 'Upload train.csv and test.csv files', 'normal');
    updateStatus('preprocessStatus', 'Load data first', 'normal');
    updateStatus('modelStatus', 'Preprocess data first', 'normal');
    updateStatus('trainingStatus', 'Create model first', 'normal');
    updateStatus('evaluationStatus', 'Train model first', 'normal');
    updateStatus('predictionStatus', 'Train model first', 'normal');
    
    console.log('App initialized');
}

// Start app
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}
