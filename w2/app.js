// Titanic Binary Classifier - TensorFlow.js
// Complete browser-based implementation with all required features

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;
let testProbabilities = null;
let testPassengerIds = null;
let firstLayerWeights = null;
let featureNames = [];
let featureImportance = null;
let validationProbs = null;

// Schema configuration
const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];

// Feature flags
let includeFamilyFeatures = true;
let isTraining = false;
let stopTrainingRequested = false;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Robust CSV parser with quote handling
 */
function parseCSV(csvText) {
    console.log('Parsing CSV...');
    
    // Normalize line endings
    csvText = csvText.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    
    const rows = [];
    let currentRow = [];
    let currentField = '';
    let insideQuotes = false;
    
    for (let i = 0; i < csvText.length; i++) {
        const char = csvText[i];
        const nextChar = csvText[i + 1] || '';
        
        if (char === '"') {
            if (insideQuotes && nextChar === '"') {
                // Escaped quote inside quoted field
                currentField += '"';
                i++; // Skip next character
            } else {
                // Start or end of quoted field
                insideQuotes = !insideQuotes;
            }
        } else if (char === ',' && !insideQuotes) {
            // End of field
            currentRow.push(currentField);
            currentField = '';
        } else if (char === '\n' && !insideQuotes) {
            // End of row
            currentRow.push(currentField);
            rows.push(currentRow);
            currentRow = [];
            currentField = '';
        } else {
            // Regular character
            currentField += char;
        }
    }
    
    // Add last row if exists
    if (currentField !== '' || currentRow.length > 0) {
        currentRow.push(currentField);
        rows.push(currentRow);
    }
    
    console.log(`Parsed ${rows.length} rows`);
    return rows;
}

/**
 * Read file as text
 */
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

/**
 * Update status message
 */
function updateStatus(elementId, message, type = 'normal') {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.textContent = message;
    element.className = 'status-box';
    
    if (type === 'success') {
        element.classList.add('success');
    } else if (type === 'error') {
        element.classList.add('error');
    }
}

/**
 * Create HTML table with proper styling
 */
function createTable(headers, data, maxRows = 10) {
    let html = '<table class="evaluation-table">';
    
    // Header
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
    
    // Footer if truncated
    if (data.length > maxRows) {
        html += `<tfoot><tr><td colspan="${headers.length}" style="text-align: center; font-style: italic;">Showing ${maxRows} of ${data.length} rows</td></tr></tfoot>`;
    }
    
    html += '</table>';
    return html;
}

/**
 * CSV export with proper quoting
 */
function exportCSV(headers, data, filename) {
    console.log(`Exporting ${data.length} rows to ${filename}`);
    
    // Create CSV content with proper quoting
    let csv = headers.map(h => `"${h}"`).join(',') + '\n';
    
    data.forEach(row => {
        const escapedRow = row.map(val => {
            const strVal = String(val);
            // Escape double quotes by doubling them
            return `"${strVal.replace(/"/g, '""')}"`;
        });
        csv += escapedRow.join(',') + '\n';
    });
    
    // Download file
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    console.log(`Exported ${filename} successfully`);
}

/**
 * Calculate metrics from confusion matrix
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
 * Load and inspect data
 */
async function loadAndInspectData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }
    
    updateStatus('data-status', 'Loading CSV files...', 'normal');
    
    try {
        // Load training data
        const trainText = await readFile(trainFile);
        console.log('Train file loaded, size:', trainText.length);
        
        const trainRows = parseCSV(trainText);
        if (trainRows.length < 2) {
            throw new Error('Training CSV file is empty or invalid');
        }
        
        // Extract headers and data
        const trainHeaders = trainRows[0];
        const trainDataRows = trainRows.slice(1);
        
        // Convert to array of objects
        trainData = trainDataRows.map(row => {
            const obj = {};
            trainHeaders.forEach((header, i) => {
                let value = row[i] || '';
                // Convert to number if possible
                if (value !== '' && !isNaN(value) && value.trim() !== '') {
                    value = parseFloat(value);
                }
                obj[header] = value === '' ? null : value;
            });
            return obj;
        });
        
        // Load test data
        const testText = await readFile(testFile);
        console.log('Test file loaded, size:', testText.length);
        
        const testRows = parseCSV(testText);
        if (testRows.length < 2) {
            throw new Error('Test CSV file is empty or invalid');
        }
        
        const testHeaders = testRows[0];
        const testDataRows = testRows.slice(1);
        
        testData = testDataRows.map(row => {
            const obj = {};
            testHeaders.forEach((header, i) => {
                let value = row[i] || '';
                if (value !== '' && !isNaN(value) && value.trim() !== '') {
                    value = parseFloat(value);
                }
                obj[header] = value === '' ? null : value;
            });
            return obj;
        });
        
        // Validate required columns
        if (!trainHeaders.includes(TARGET_FEATURE)) {
            throw new Error(`Training CSV must contain "${TARGET_FEATURE}" column`);
        }
        
        // Show preview
        document.getElementById('data-preview-container').style.display = 'block';
        
        // Create preview table
        const previewHeaders = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Fare'];
        const previewData = trainData.slice(0, 10).map(row => [
            row[ID_FEATURE] || '',
            row[TARGET_FEATURE] || '',
            row['Pclass'] || '',
            (row['Name'] || '').substring(0, 20) + '...',
            row['Sex'] || '',
            row['Age'] || '',
            row['Fare'] || ''
        ]);
        
        document.getElementById('data-preview').innerHTML = createTable(previewHeaders, previewData, 10);
        
        // Show statistics
        const statsDiv = document.getElementById('data-stats');
        const totalRows = trainData.length;
        const survivedCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
        const survivalRate = ((survivedCount / totalRows) * 100).toFixed(1);
        
        let statsHTML = '<h3>Data Statistics</h3>';
        statsHTML += `<p><strong>Training Samples:</strong> ${totalRows}</p>`;
        statsHTML += `<p><strong>Test Samples:</strong> ${testData.length}</p>`;
        statsHTML += `<p><strong>Survival Rate:</strong> ${survivalRate}% (${survivedCount}/${totalRows})</p>`;
        
        // Missing values
        statsHTML += '<h4>Missing Values:</h4><ul>';
        ['Age', 'Fare', 'Embarked'].forEach(field => {
            const missing = trainData.filter(row => row[field] === null || row[field] === undefined).length;
            const percent = ((missing / totalRows) * 100).toFixed(1);
            statsHTML += `<li>${field}: ${missing} (${percent}%)</li>`;
        });
        statsHTML += '</ul>';
        
        statsDiv.innerHTML = statsHTML;
        
        updateStatus('data-status', 
            `✅ Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples. Survival rate: ${survivalRate}%`,
            'success'
        );
        
        // Enable preprocessing button
        document.getElementById('preprocess-btn').disabled = false;
        
    } catch (error) {
        updateStatus('data-status', `❌ Error loading data: ${error.message}`, 'error');
        console.error('Load error:', error);
    }
}

// ============================================================================
// PREPROCESSING
// ============================================================================

/**
 * Preprocess data
 */
function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }
    
    updateStatus('preprocessing-output', 'Preprocessing data...', 'normal');
    
    try {
        // Get toggle state
        includeFamilyFeatures = document.getElementById('family-features-toggle').checked;
        
        // Calculate statistics from training data
        const ageValues = trainData.map(row => row['Age']).filter(val => val !== null);
        const fareValues = trainData.map(row => row['Fare']).filter(val => val !== null);
        const embarkedValues = trainData.map(row => row['Embarked']).filter(val => val !== null);
        
        const ageMedian = calculateMedian(ageValues);
        const fareMedian = calculateMedian(fareValues);
        const embarkedMode = calculateMode(embarkedValues) || 'S';
        
        const ageStd = calculateStdDev(ageValues) || 1;
        const fareStd = calculateStdDev(fareValues) || 1;
        
        // Build feature names
        featureNames = [];
        
        // Pclass one-hot
        featureNames.push('Pclass_1', 'Pclass_2', 'Pclass_3');
        
        // Sex one-hot
        featureNames.push('Sex_female');
        
        // Numerical features
        featureNames.push('Age_std', 'Fare_std', 'SibSp', 'Parch');
        
        // Family features if enabled
        if (includeFamilyFeatures) {
            featureNames.push('FamilySize', 'IsAlone');
        }
        
        // Embarked one-hot
        featureNames.push('Embarked_C', 'Embarked_Q', 'Embarked_S');
        
        console.log('Feature names:', featureNames);
        
        // Preprocess training data
        const trainFeatures = [];
        const trainLabels = [];
        
        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode, ageStd, fareStd);
            trainFeatures.push(features);
            trainLabels.push(row[TARGET_FEATURE] || 0);
        });
        
        // Preprocess test data
        const testFeatures = [];
        testPassengerIds = [];
        
        testData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode, ageStd, fareStd);
            testFeatures.push(features);
            testPassengerIds.push(row[ID_FEATURE] || '');
        });
        
        // Store preprocessed data
        preprocessedTrainData = {
            features: tf.tensor2d(trainFeatures),
            labels: tf.tensor1d(trainLabels)
        };
        
        preprocessedTestData = {
            features: testFeatures,
            passengerIds: testPassengerIds
        };
        
        updateStatus('preprocessing-output', 
            `✅ Preprocessing complete! Created ${featureNames.length} features. Training shape: [${trainFeatures.length}, ${trainFeatures[0]?.length || 0}]`,
            'success'
        );
        
        // Enable model creation
        document.getElementById('create-model-btn').disabled = false;
        
    } catch (error) {
        updateStatus('preprocessing-output', `❌ Preprocessing error: ${error.message}`, 'error');
        console.error('Preprocessing error:', error);
    }
}

/**
 * Extract features from a row
 */
function extractFeatures(row, ageMedian, fareMedian, embarkedMode, ageStd, fareStd) {
    const features = [];
    
    // Pclass one-hot encoding
    const pclass = row['Pclass'] || 1;
    features.push(pclass === 1 ? 1 : 0);
    features.push(pclass === 2 ? 1 : 0);
    features.push(pclass === 3 ? 1 : 0);
    
    // Sex (female=1, male=0)
    const sex = (row['Sex'] || '').toLowerCase();
    features.push(sex === 'female' ? 1 : 0);
    
    // Age (standardized)
    let age = row['Age'];
    if (age === null || age === undefined) age = ageMedian;
    features.push((age - ageMedian) / ageStd);
    
    // Fare (standardized)
    let fare = row['Fare'];
    if (fare === null || fare === undefined) fare = fareMedian;
    features.push((fare - fareMedian) / fareStd);
    
    // SibSp and Parch
    features.push(row['SibSp'] || 0);
    features.push(row['Parch'] || 0);
    
    // Family features if enabled
    if (includeFamilyFeatures) {
        const familySize = (row['SibSp'] || 0) + (row['Parch'] || 0) + 1;
        const isAlone = familySize === 1 ? 1 : 0;
        features.push(familySize);
        features.push(isAlone);
    }
    
    // Embarked one-hot encoding
    let embarked = row['Embarked'];
    if (!embarked || !['C', 'Q', 'S'].includes(embarked)) {
        embarked = embarkedMode;
    }
    features.push(embarked === 'C' ? 1 : 0);
    features.push(embarked === 'Q' ? 1 : 0);
    features.push(embarked === 'S' ? 1 : 0);
    
    return features;
}

/**
 * Calculate median
 */
function calculateMedian(values) {
    if (!values || values.length === 0) return 0;
    
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    
    if (sorted.length % 2 === 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    
    return sorted[mid];
}

/**
 * Calculate mode
 */
function calculateMode(values) {
    if (!values || values.length === 0) return null;
    
    const counts = {};
    let maxCount = 0;
    let mode = null;
    
    values.forEach(value => {
        counts[value] = (counts[value] || 0) + 1;
        if (counts[value] > maxCount) {
            maxCount = counts[value];
            mode = value;
        }
    });
    
    return mode;
}

/**
 * Calculate standard deviation
 */
function calculateStdDev(values) {
    if (!values || values.length === 0) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    return Math.sqrt(variance);
}

// ============================================================================
// MODEL CREATION
// ============================================================================

/**
 * Create the model
 */
function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }
    
    updateStatus('training-status', 'Creating model...', 'normal');
    
    try {
        const inputShape = preprocessedTrainData.features.shape[1];
        
        // Create sequential model
        model = tf.sequential();
        
        // Hidden layer (single hidden layer as required)
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [inputShape],
            name: 'hidden_layer'
        }));
        
        // Output layer
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            name: 'output_layer'
        }));
        
        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Show model summary
        document.getElementById('model-summary').style.display = 'block';
        
        let summaryHTML = '<h4>Model Architecture:</h4><ul>';
        let totalParams = 0;
        
        model.layers.forEach((layer, i) => {
            const layerParams = layer.countParams();
            totalParams += layerParams;
            
            summaryHTML += `<li>Layer ${i + 1}: ${layer.name} - ${JSON.stringify(layer.outputShape)} (${layerParams} parameters)</li>`;
        });
        
        summaryHTML += `</ul><p><strong>Total Parameters:</strong> ${totalParams}</p>`;
        summaryHTML += `<p><strong>Input Shape:</strong> [${inputShape}]</p>`;
        
        document.getElementById('model-details').innerHTML = summaryHTML;
        
        updateStatus('training-status', 
            `✅ Model created! Input: ${inputShape} features → Hidden: 16 units → Output: 1 unit`,
            'success'
        );
        
        // Enable training button
        document.getElementById('train-btn').disabled = false;
        
    } catch (error) {
        updateStatus('training-status', `❌ Model creation error: ${error.message}`, 'error');
        console.error('Model creation error:', error);
    }
}

// ============================================================================
// TRAINING
// ============================================================================

/**
 * Train the model
 */
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }
    
    updateStatus('training-status', 'Starting training...', 'normal');
    document.getElementById('training-charts').style.display = 'block';
    
    isTraining = true;
    stopTrainingRequested = false;
    
    document.getElementById('train-btn').disabled = true;
    document.getElementById('stop-train-btn').disabled = false;
    document.getElementById('training-progress').style.width = '0%';
    
    try {
        // Split data (80/20)
        const totalSamples = preprocessedTrainData.features.shape[0];
        const splitIndex = Math.floor(totalSamples * 0.8);
        
        const trainFeatures = preprocessedTrainData.features.slice(0, splitIndex);
        const trainLabels = preprocessedTrainData.labels.slice(0, splitIndex);
        
        const valFeatures = preprocessedTrainData.features.slice(splitIndex);
        const valLabels = preprocessedTrainData.labels.slice(splitIndex);
        
        // Store validation data
        validationData = valFeatures;
        validationLabels = await valLabels.array();
        
        // Prepare callbacks
        const metrics = ['loss', 'acc', 'val_loss', 'val_acc'];
        const container = {
            name: 'Training Metrics',
            tab: 'Training',
            styles: { height: '300px' }
        };
        
        const fitCallbacks = tfvis.show.fitCallbacks(container, metrics, {
            callbacks: ['onBatchEnd', 'onEpochEnd']
        });
        
        // Custom callback for progress
        const progressCallback = {
            onEpochEnd: async (epoch, logs) => {
                const progress = ((epoch + 1) / 50) * 100;
                document.getElementById('training-progress').style.width = `${progress}%`;
                
                updateStatus('training-status', 
                    `Epoch ${epoch + 1}/50 - Loss: ${logs.loss.toFixed(4)}, Acc: ${logs.acc.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}, Val Acc: ${logs.val_acc.toFixed(4)}`,
                    'normal'
                );
                
                if (stopTrainingRequested) {
                    model.stopTraining = true;
                }
                
                // Get first layer weights for feature importance
                if (epoch === 49 || stopTrainingRequested) {
                    const hiddenLayer = model.getLayer('hidden_layer');
                    if (hiddenLayer) {
                        const weights = hiddenLayer.getWeights();
                        if (weights.length > 0) {
                            firstLayerWeights = weights[0];
                            console.log('First layer weights shape:', firstLayerWeights.shape);
                        }
                    }
                }
            }
        };
        
        // Train model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: [fitCallbacks, progressCallback],
            verbose: 0
        });
        
        // Get validation predictions
        validationPredictions = model.predict(valFeatures);
        validationProbs = await validationPredictions.array();
        
        updateStatus('training-status', 
            `✅ Training complete! Final accuracy: ${(trainingHistory.history.acc[trainingHistory.history.acc.length - 1] * 100).toFixed(1)}%`,
            'success'
        );
        
        // Enable evaluation and prediction
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('show-roc-btn').disabled = false;
        document.getElementById('predict-btn').disabled = false;
        document.getElementById('save-model-btn').disabled = false;
        
        // Calculate initial metrics
        updateMetrics(0.5);
        
    } catch (error) {
        updateStatus('training-status', `❌ Training error: ${error.message}`, 'error');
        console.error('Training error:', error);
    } finally {
        isTraining = false;
        document.getElementById('train-btn').disabled = false;
        document.getElementById('stop-train-btn').disabled = true;
    }
}

/**
 * Stop training
 */
function stopTraining() {
    if (isTraining) {
        stopTrainingRequested = true;
        updateStatus('training-status', 'Stopping training...', 'normal');
    }
}

// ============================================================================
// EVALUATION & FEATURE IMPORTANCE
// ============================================================================

/**
 * Update metrics based on threshold
 */
async function updateMetrics(threshold) {
    if (!validationProbs || !validationLabels) return;
    
    threshold = threshold || parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    try {
        // Calculate confusion matrix
        let tp = 0, tn = 0, fp = 0, fn = 0;
        
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
        
        // Update confusion matrix display
        const cmDiv = document.getElementById('confusion-matrix');
        cmDiv.innerHTML = `
            <table class="evaluation-table">
                <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
                <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
                <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
            </table>
            <p><strong>Total:</strong> ${tp + tn + fp + fn} samples</p>
        `;
        
        // Calculate metrics
        const metrics = calculateMetrics(tp, fp, fn, tn);
        
        // Update performance metrics display
        const metricsDiv = document.getElementById('performance-metrics');
        metricsDiv.innerHTML = `
            <table class="evaluation-table">
                <tr><th>Metric</th><th>Value</th><th>Formula</th></tr>
                <tr><td>Accuracy</td><td>${metrics.accuracy}</td><td>(TP+TN)/Total</td></tr>
                <tr><td>Precision</td><td>${metrics.precision}</td><td>TP/(TP+FP)</td></tr>
                <tr><td>Recall</td><td>${metrics.recall}</td><td>TP/(TP+FN)</td></tr>
                <tr><td>F1-Score</td><td>${metrics.f1}</td><td>2×(P×R)/(P+R)</td></tr>
            </table>
        `;
        
        // Calculate ROC/AUC
        await calculateROC();
        
        // Calculate feature importance
        if (firstLayerWeights && featureNames.length > 0) {
            await calculateFeatureImportance();
        }
        
    } catch (error) {
        console.error('Metrics update error:', error);
    }
}

/**
 * Calculate ROC curve and AUC
 */
async function calculateROC() {
    if (!validationProbs || !validationLabels) return;
    
    try {
        // Sort by probability
        const paired = validationProbs.map((prob, i) => ({
            prob: prob[0],
            label: validationLabels[i]
        })).sort((a, b) => b.prob - a.prob);
        
        const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
        const tprs = [];
        const fprs = [];
        
        let auc = 0;
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
            
            // Calculate AUC using trapezoidal rule
            if (fpr > prevFPR) {
                auc += (fpr - prevFPR) * (tpr + prevTPR) / 2;
                prevFPR = fpr;
                prevTPR = tpr;
            }
        });
        
        // Update AUC display
        const aucDiv = document.getElementById('auc-metrics');
        aucDiv.innerHTML = `
            <p><strong>AUC:</strong> ${auc.toFixed(4)}</p>
            <p><strong>Interpretation:</strong></p>
            <ul>
                <li>0.9-1.0: Excellent</li>
                <li>0.8-0.9: Good</li>
                <li>0.7-0.8: Fair</li>
                <li>0.6-0.7: Poor</li>
                <li>0.5-0.6: Fail</li>
            </ul>
        `;
        
    } catch (error) {
        console.error('ROC calculation error:', error);
    }
}

/**
 * Show ROC curve visualization
 */
async function showROCCurve() {
    if (!validationProbs || !validationLabels) {
        alert('Please train model first.');
        return;
    }
    
    try {
        // Calculate ROC data
        const paired = validationProbs.map((prob, i) => ({
            prob: prob[0],
            label: validationLabels[i]
        })).sort((a, b) => b.prob - a.prob);
        
        const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
        const rocPoints = [];
        
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
            
            rocPoints.push({ x: fpr, y: tpr });
        });
        
        // Plot ROC curve
        tfvis.render.linechart(
            { name: 'ROC Curve', tab: 'Evaluation' },
            { values: rocPoints, series: ['ROC Curve'] },
            {
                xLabel: 'False Positive Rate',
                yLabel: 'True Positive Rate',
                width: 400,
                height: 400
            }
        );
        
    } catch (error) {
        console.error('ROC visualization error:', error);
    }
}

/**
 * Calculate feature importance using Sigmoid Gate
 */
async function calculateFeatureImportance() {
    if (!firstLayerWeights || !featureNames || featureNames.length === 0 || !preprocessedTrainData) {
        console.log('Missing data for feature importance calculation');
        return;
    }
    
    try {
        // Get feature data
        const featureData = await preprocessedTrainData.features.array();
        const numFeatures = featureData[0].length;
        
        // Calculate feature ranges
        const featureRanges = [];
        for (let i = 0; i < numFeatures; i++) {
            const values = featureData.map(row => row[i]);
            const min = Math.min(...values);
            const max = Math.max(...values);
            const range = max - min;
            // Standardize range to [0,1] for binary features
            featureRanges.push(range > 1 ? 1 : range);
        }
        
        // Get weights
        const weights = await firstLayerWeights.array();
        
        // Calculate importance: sigmoid(W * standardized_range)
        const importances = [];
        for (let i = 0; i < numFeatures; i++) {
            let weightedSum = 0;
            
            // Sum absolute weights across all hidden units
            for (let j = 0; j < weights[i].length; j++) {
                weightedSum += Math.abs(weights[i][j]) * featureRanges[i];
            }
            
            // Apply sigmoid
            const importance = 1 / (1 + Math.exp(-weightedSum));
            
            importances.push({
                feature: featureNames[i] || `Feature_${i}`,
                importance: importance,
                weightedSum: weightedSum
            });
        }
        
        // Sort by importance
        importances.sort((a, b) => b.importance - a.importance);
        featureImportance = importances;
        
        // Display top 5 features
        const topFeatures = importances.slice(0, 5);
        const importanceDiv = document.getElementById('feature-importance');
        
        let importanceHTML = '<table class="evaluation-table">';
        importanceHTML += '<tr><th>Rank</th><th>Feature</th><th>Importance</th><th>Visualization</th></tr>';
        
        topFeatures.forEach((item, index) => {
            const percentage = (item.importance * 100).toFixed(1);
            importanceHTML += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${item.feature}</td>
                    <td>${item.importance.toFixed(4)}</td>
                    <td>
                        <div class="importance-container">
                            <div style="flex: 1; background: #eee; height: 20px; border-radius: 3px;">
                                <div class="importance-bar" style="width: ${percentage}%"></div>
                            </div>
                            <span>${percentage}%</span>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        importanceHTML += '</table>';
        importanceDiv.innerHTML = importanceHTML;
        
        console.log('Feature importance calculated:', topFeatures);
        
    } catch (error) {
        console.error('Feature importance calculation error:', error);
        document.getElementById('feature-importance').innerHTML = 
            `<p>Error calculating feature importance: ${error.message}</p>`;
    }
}

// ============================================================================
// PREDICTION & EXPORT
// ============================================================================

/**
 * Predict on test data
 */
async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }
    
    updateStatus('prediction-output', 'Making predictions...', 'normal');
    
    try {
        // Convert test features to tensor
        const testFeatures = tf.tensor2d(preprocessedTestData.features);
        
        // Make predictions
        const predictions = model.predict(testFeatures);
        const probabilities = await predictions.array();
        
        // Apply threshold
        const threshold = parseFloat(document.getElementById('threshold-value').textContent);
        testPredictions = probabilities.map(prob => prob[0] >= threshold ? 1 : 0);
        testProbabilities = probabilities;
        
        // Show preview
        document.getElementById('prediction-preview').style.display = 'block';
        
        const previewData = [];
        for (let i = 0; i < Math.min(10, testPassengerIds.length); i++) {
            previewData.push([
                testPassengerIds[i],
                testPredictions[i],
                testProbabilities[i][0].toFixed(4)
            ]);
        }
        
        document.getElementById('prediction-table').innerHTML = createTable(
            ['PassengerId', 'Survived', 'Probability'],
            previewData,
            10
        );
        
        const survivalCount = testPredictions.filter(p => p === 1).length;
        const survivalRate = ((survivalCount / testPredictions.length) * 100).toFixed(1);
        
        updateStatus('prediction-output', 
            `✅ Predictions complete! ${testPredictions.length} samples predicted. Survival rate: ${survivalRate}%`,
            'success'
        );
        
        // Enable export buttons
        document.getElementById('export-submission-btn').disabled = false;
        document.getElementById('export-probabilities-btn').disabled = false;
        
    } catch (error) {
        updateStatus('prediction-output', `❌ Prediction error: ${error.message}`, 'error');
        console.error('Prediction error:', error);
    }
}

/**
 * Export submission CSV
 */
function exportSubmission() {
    if (!testPredictions || !testPassengerIds) {
        alert('Please make predictions first.');
        return;
    }
    
    try {
        // Prepare data with proper quoting
        const data = testPassengerIds.map((id, i) => [`"${id}"`, `"${testPredictions[i]}"`]);
        
        // Use proper CSV export function
        exportCSV(
            ['PassengerId', 'Survived'],
            testPassengerIds.map((id, i) => [id, testPredictions[i]]),
            'submission.csv'
        );
        
        updateStatus('export-status', '✅ Submission CSV exported! Check your downloads folder.', 'success');
        
    } catch (error) {
        updateStatus('export-status', `❌ Export error: ${error.message}`, 'error');
        console.error('Export error:', error);
    }
}

/**
 * Export probabilities CSV
 */
function exportProbabilities() {
    if (!testProbabilities || !testPassengerIds) {
        alert('Please make predictions first.');
        return;
    }
    
    try {
        exportCSV(
            ['PassengerId', 'Probability'],
            testPassengerIds.map((id, i) => [id, testProbabilities[i][0].toFixed(6)]),
            'probabilities.csv'
        );
        
        updateStatus('export-status', '✅ Probabilities CSV exported! Check your downloads folder.', 'success');
        
    } catch (error) {
        updateStatus('export-status', `❌ Export error: ${error.message}`, 'error');
        console.error('Export error:', error);
    }
}

/**
 * Save model
 */
async function saveModel() {
    if (!model) {
        alert('No model to save.');
        return;
    }
    
    try {
        await model.save('downloads://titanic-tfjs-model');
        updateStatus('export-status', '✅ Model saved! Check your downloads folder.', 'success');
    } catch (error) {
        updateStatus('export-status', `❌ Model save error: ${error.message}`, 'error');
        console.error('Model save error:', error);
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize application
 */
function initializeApp() {
    console.log('Initializing Titanic Classifier...');
    
    // Set up event listeners
    document.getElementById('load-data-btn').addEventListener('click', loadAndInspectData);
    document.getElementById('preprocess-btn').addEventListener('click', preprocessData);
    document.getElementById('create-model-btn').addEventListener('click', createModel);
    document.getElementById('train-btn').addEventListener('click', trainModel);
    document.getElementById('stop-train-btn').addEventListener('click', stopTraining);
    document.getElementById('predict-btn').addEventListener('click', predict);
    document.getElementById('export-submission-btn').addEventListener('click', exportSubmission);
    document.getElementById('export-probabilities-btn').addEventListener('click', exportProbabilities);
    document.getElementById('save-model-btn').addEventListener('click', saveModel);
    document.getElementById('show-roc-btn').addEventListener('click', showROCCurve);
    
    // Threshold slider
    const thresholdSlider = document.getElementById('threshold-slider');
    thresholdSlider.addEventListener('input', function() {
        updateMetrics(parseFloat(this.value));
    });
    
    // Initialize status messages
    updateStatus('data-status', 'Please upload train.csv and test.csv files', 'normal');
    updateStatus('preprocessing-output', 'Load data first to enable preprocessing', 'normal');
    updateStatus('training-status', 'Create model first to enable training', 'normal');
    updateStatus('prediction-output', 'Train model first to make predictions', 'normal');
    updateStatus('export-status', 'Make predictions first to export results', 'normal');
    
    console.log('Application initialized and ready!');
}

// Start application when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}
