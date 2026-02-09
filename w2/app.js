// Titanic Binary Classifier - TensorFlow.js
// Complete browser-based implementation for Kaggle Titanic dataset

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

// Feature flags for toggling engineered features
let includeFamilySize = true;
let includeIsAlone = true;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Parse CSV data with proper quote handling
 * @param {string} text - CSV text content
 * @returns {Array} Array of parsed rows
 */
function parseCSV(text) {
    const rows = [];
    let currentRow = [];
    let currentCell = "";
    let insideQuotes = false;
    
    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        const nextChar = text[i + 1] || "";
        
        if (char === '"') {
            if (insideQuotes && nextChar === '"') {
                // Escaped quote inside quoted field
                currentCell += '"';
                i++; // Skip next character
            } else {
                // Start or end of quoted field
                insideQuotes = !insideQuotes;
            }
        } else if (char === ',' && !insideQuotes) {
            // End of cell
            currentRow.push(currentCell);
            currentCell = "";
        } else if (char === '\n' && !insideQuotes) {
            // End of row (handle both \n and \r\n)
            currentRow.push(currentCell);
            rows.push(currentRow);
            currentRow = [];
            currentCell = "";
            
            // Skip \r if present
            if (nextChar === '\r') i++;
        } else if (char === '\r' && !insideQuotes) {
            // Handle \r as line break (Mac style)
            currentRow.push(currentCell);
            rows.push(currentRow);
            currentRow = [];
            currentCell = "";
        } else {
            // Regular character
            currentCell += char;
        }
    }
    
    // Add last row if exists
    if (currentCell !== "" || currentRow.length > 0) {
        currentRow.push(currentCell);
        rows.push(currentRow);
    }
    
    return rows;
}

/**
 * Update status message with styling
 * @param {string} elementId - ID of status element
 * @param {string} message - Status message
 * @param {string} type - 'normal', 'success', 'error', or 'warning'
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
 * Create a simple HTML table from data
 * @param {Array} headers - Table headers
 * @param {Array} data - Table rows
 * @param {number} maxRows - Maximum rows to display
 * @returns {string} HTML table string
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
    
    // Show row count if truncated
    if (data.length > maxRows) {
        html += `<tfoot><tr><td colspan="${headers.length}" style="text-align: center; font-style: italic;">Showing ${maxRows} of ${data.length} rows</td></tr></tfoot>`;
    }
    
    html += '</table>';
    return html;
}

/**
 * Download data as CSV file
 * @param {string} content - CSV content
 * @param {string} filename - Filename for download
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
 * Export CSV with proper quoting and escaping
 * @param {Array} headers - Column headers
 * @param {Array} data - Array of data rows
 * @param {string} filename - Output filename
 */
function exportCSV(headers, data, filename) {
    // Quote headers
    let csv = headers.map(h => `"${h}"`).join(',') + '\n';
    
    // Quote and escape data rows
    data.forEach(row => {
        const escapedRow = row.map(val => {
            // Convert to string and escape double quotes
            const strVal = String(val);
            return `"${strVal.replace(/"/g, '""')}"`;
        });
        csv += escapedRow.join(',') + '\n';
    });
    
    downloadCSV(csv, filename);
}

/**
 * Calculate metrics from confusion matrix
 * @param {number} tp - True positives
 * @param {number} fp - False positives
 * @param {number} fn - False negatives
 * @param {number} tn - True negatives
 * @returns {Object} Metrics object
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
 * Load CSV file from file input
 * @param {HTMLInputElement} fileInput - File input element
 * @returns {Promise} Promise resolving to parsed data
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
                    reject(new Error('CSV file is empty or has no data rows'));
                    return;
                }
                
                // Extract headers and data
                const headers = rows[0];
                const data = rows.slice(1);
                
                resolve({ headers, data });
            } catch (error) {
                reject(new Error(`Failed to parse CSV: ${error.message}`));
            }
        };
        
        reader.onerror = function() {
            reject(new Error('Failed to read file'));
        };
        
        reader.readAsText(file);
    });
}

/**
 * Load and inspect both train and test datasets
 */
async function loadAndInspectData() {
    try {
        updateStatus('dataStatus', 'Loading CSV files...', 'normal');
        
        // Reset previous data
        rawTrainData = null;
        rawTestData = null;
        
        // Load train data
        const trainFileInput = document.getElementById('trainFile');
        const trainResult = await loadCSVFile(trainFileInput);
        rawTrainData = trainResult;
        
        // Load test data
        const testFileInput = document.getElementById('testFile');
        const testResult = await loadCSVFile(testFileInput);
        rawTestData = testResult;
        
        // Validate data structure
        if (!rawTrainData.headers.includes('Survived')) {
            throw new Error('Train CSV must contain "Survived" column');
        }
        
        if (!rawTrainData.headers.includes('PassengerId') || !rawTestData.headers.includes('PassengerId')) {
            throw new Error('Both CSVs must contain "PassengerId" column');
        }
        
        // Show preview tables
        document.getElementById('dataPreview').style.display = 'block';
        
        // Train preview
        const trainPreviewRows = rawTrainData.data.slice(0, 5).map(row => {
            const previewRow = [];
            // Show selected columns for preview
            const passengerIdIdx = rawTrainData.headers.indexOf('PassengerId');
            const survivedIdx = rawTrainData.headers.indexOf('Survived');
            const pclassIdx = rawTrainData.headers.indexOf('Pclass');
            const nameIdx = rawTrainData.headers.indexOf('Name');
            const sexIdx = rawTrainData.headers.indexOf('Sex');
            const ageIdx = rawTrainData.headers.indexOf('Age');
            
            previewRow.push(row[passengerIdIdx] || '');
            previewRow.push(row[survivedIdx] || '');
            previewRow.push(row[pclassIdx] || '');
            previewRow.push((row[nameIdx] || '').substring(0, 20) + '...');
            previewRow.push(row[sexIdx] || '');
            previewRow.push(row[ageIdx] || '');
            
            return previewRow;
        });
        
        document.getElementById('trainPreview').innerHTML = createTable(
            ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age'],
            trainPreviewRows,
            5
        );
        
        // Test preview
        const testPreviewRows = rawTestData.data.slice(0, 5).map(row => {
            const previewRow = [];
            const passengerIdIdx = rawTestData.headers.indexOf('PassengerId');
            const pclassIdx = rawTestData.headers.indexOf('Pclass');
            const nameIdx = rawTestData.headers.indexOf('Name');
            const sexIdx = rawTestData.headers.indexOf('Sex');
            const ageIdx = rawTestData.headers.indexOf('Age');
            
            previewRow.push(row[passengerIdIdx] || '');
            previewRow.push(row[pclassIdx] || '');
            previewRow.push((row[nameIdx] || '').substring(0, 20) + '...');
            previewRow.push(row[sexIdx] || '');
            previewRow.push(row[ageIdx] || '');
            
            return previewRow;
        });
        
        document.getElementById('testPreview').innerHTML = createTable(
            ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age'],
            testPreviewRows,
            5
        );
        
        // Calculate and display data stats
        const trainRows = rawTrainData.data.length;
        const testRows = rawTestData.data.length;
        const trainCols = rawTrainData.headers.length;
        const testCols = rawTestData.headers.length;
        
        // Count missing values in key columns
        const ageIdx = rawTrainData.headers.indexOf('Age');
        const embarkedIdx = rawTrainData.headers.indexOf('Embarked');
        
        const missingAge = rawTrainData.data.filter(row => !row[ageIdx] || row[ageIdx].trim() === '').length;
        const missingEmbarked = rawTrainData.data.filter(row => !row[embarkedIdx] || row[embarkedIdx].trim() === '').length;
        
        // Survival stats
        const survivedIdx = rawTrainData.headers.indexOf('Survived');
        const totalSurvived = rawTrainData.data.filter(row => row[survivedIdx] === '1').length;
        const survivalRate = (totalSurvived / trainRows * 100).toFixed(1);
        
        updateStatus('dataStatus', 
            `✅ Data loaded successfully! Train: ${trainRows} rows × ${trainCols} cols, Test: ${testRows} rows × ${testCols} cols. ` +
            `Survival rate: ${survivalRate}%. Missing values: Age (${missingAge}), Embarked (${missingEmbarked}).`,
            'success'
        );
        
        // Enable preprocessing button
        document.getElementById('preprocessBtn').disabled = false;
        
        // Log for debugging
        console.log('Train data shape:', [trainRows, trainCols]);
        console.log('Test data shape:', [testRows, testCols]);
        console.log('Train headers:', rawTrainData.headers);
        
    } catch (error) {
        updateStatus('dataStatus', `❌ Error loading data: ${error.message}`, 'error');
        console.error('Data loading error:', error);
    }
}

// ============================================================================
// DATA PREPROCESSING
// ============================================================================

/**
 * Extract features from raw data
 * @param {Array} data - Raw data rows
 * @param {Array} headers - Column headers
 * @param {boolean} isTraining - Whether this is training data (has Survived)
 * @returns {Object} Processed features and targets
 */
function extractFeatures(data, headers, isTraining = true) {
    // Index key columns
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
    
    // Collect all values for imputation calculations
    const ages = [];
    const fares = [];
    const embarkedValues = [];
    
    for (const row of data) {
        // PassengerId
        passengerIds.push(row[passengerIdIdx]);
        
        // Target (if training)
        if (isTraining && survivedIdx !== -1) {
            targets.push(parseInt(row[survivedIdx]) || 0);
        }
        
        // Collect values for imputation
        const age = parseFloat(row[ageIdx]);
        if (!isNaN(age)) ages.push(age);
        
        const fare = parseFloat(row[fareIdx]);
        if (!isNaN(fare)) fares.push(fare);
        
        const embarked = row[embarkedIdx];
        if (embarked && embarked.trim() !== '') {
            embarkedValues.push(embarked.trim());
        }
    }
    
    // Calculate imputation values
    const medianAge = ages.length > 0 ? 
        ages.sort((a, b) => a - b)[Math.floor(ages.length / 2)] : 30;
    
    const medianFare = fares.length > 0 ? 
        fares.sort((a, b) => a - b)[Math.floor(fares.length / 2)] : 32;
    
    // Find mode of Embarked
    const embarkedCounts = {};
    embarkedValues.forEach(val => {
        embarkedCounts[val] = (embarkedCounts[val] || 0) + 1;
    });
    const modeEmbarked = Object.keys(embarkedCounts).length > 0 ?
        Object.keys(embarkedCounts).reduce((a, b) => embarkedCounts[a] > embarkedCounts[b] ? a : b) : 'S';
    
    // Calculate standardization stats
    const meanAge = ages.reduce((a, b) => a + b, 0) / ages.length || medianAge;
    const stdAge = Math.sqrt(ages.reduce((sq, n) => sq + Math.pow(n - meanAge, 2), 0) / ages.length) || 10;
    
    const meanFare = fares.reduce((a, b) => a + b, 0) / fares.length || medianFare;
    const stdFare = Math.sqrt(fares.reduce((sq, n) => sq + Math.pow(n - meanFare, 2), 0) / fares.length) || 20;
    
    // Process each row
    for (const row of data) {
        const featureRow = [];
        
        // Pclass (one-hot encoded)
        const pclass = parseInt(row[pclassIdx]) || 1;
        featureRow.push(pclass === 1 ? 1 : 0);
        featureRow.push(pclass === 2 ? 1 : 0);
        featureRow.push(pclass === 3 ? 1 : 0);
        
        // Sex (one-hot: female=1, male=0)
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
        
        // FamilySize and IsAlone (engineered features)
        if (includeFamilySize) {
            const familySize = sibSp + parch + 1;
            featureRow.push(familySize);
        }
        
        if (includeIsAlone) {
            const isAlone = (sibSp === 0 && parch === 0) ? 1 : 0;
            featureRow.push(isAlone);
        }
        
        // Fare (standardized)
        let fare = parseFloat(row[fareIdx]);
        if (isNaN(fare)) fare = medianFare;
        featureRow.push((fare - meanFare) / stdFare);
        
        // Embarked (one-hot encoded)
        let embarked = (row[embarkedIdx] || '').trim();
        if (embarked === '') embarked = modeEmbarked;
        featureRow.push(embarked === 'C' ? 1 : 0);
        featureRow.push(embarked === 'Q' ? 1 : 0);
        featureRow.push(embarked === 'S' ? 1 : 0);
        
        features.push(featureRow);
    }
    
    // Create feature names
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
 * Preprocess the loaded data
 */
function preprocessData() {
    try {
        updateStatus('preprocessStatus', 'Preprocessing data...', 'normal');
        
        if (!rawTrainData || !rawTestData) {
            throw new Error('Please load data first');
        }
        
        // Process training data
        const trainResult = extractFeatures(rawTrainData.data, rawTrainData.headers, true);
        processedTrainData = trainResult;
        
        // Process test data (using training stats for consistent standardization)
        const testResult = extractFeatures(rawTestData.data, rawTestData.headers, false);
        processedTestData = testResult;
        
        // Store feature info for later use
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
        
        // Show preprocessing info
        document.getElementById('preprocessInfo').style.display = 'block';
        
        const featureInfoHTML = `
            <p><strong>Processed Features (${featureInfo.trainShape[1]} total):</strong> ${featureInfo.names.join(', ')}</p>
            <p><strong>Training Data Shape:</strong> ${featureInfo.trainShape[0]} samples × ${featureInfo.trainShape[1]} features</p>
            <p><strong>Test Data Shape:</strong> ${featureInfo.testShape[0]} samples × ${featureInfo.testShape[1]} features</p>
            <p><strong>Preprocessing Stats:</strong></p>
            <ul>
                <li>Age: mean=${featureInfo.stats.meanAge.toFixed(2)}, std=${featureInfo.stats.stdAge.toFixed(2)}</li>
                <li>Fare: mean=${featureInfo.stats.meanFare.toFixed(2)}, std=${featureInfo.stats.stdFare.toFixed(2)}</li>
                <li>Embarked mode: ${featureInfo.stats.modeEmbarked}</li>
                <li>Family features: ${includeFamilySize ? 'FamilySize included' : 'excluded'}, ${includeIsAlone ? 'IsAlone included' : 'excluded'}</li>
            </ul>
        `;
        
        document.getElementById('featureInfo').innerHTML = featureInfoHTML;
        
        updateStatus('preprocessStatus', 
            `✅ Preprocessing complete! ${featureInfo.trainShape[1]} features created. Ready for model creation.`,
            'success'
        );
        
        // Enable model creation button
        document.getElementById('createModelBtn').disabled = false;
        document.getElementById('toggleFeatureBtn').disabled = false;
        
        console.log('Preprocessing complete. Feature names:', featureInfo.names);
        console.log('Training features shape:', trainTensors.features.shape);
        console.log('Test features shape:', testTensors.features.shape);
        
    } catch (error) {
        updateStatus('preprocessStatus', `❌ Preprocessing error: ${error.message}`, 'error');
        console.error('Preprocessing error:', error);
    }
}

/**
 * Toggle engineered features
 */
function toggleFeatures() {
    includeFamilySize = !includeFamilySize;
    includeIsAlone = !includeIsAlone;
    
    // Re-preprocess data
    preprocessData();
}

// ============================================================================
// MODEL CREATION
// ============================================================================

/**
 * Create the neural network model
 */
function createModel() {
    try {
        updateStatus('modelStatus', 'Creating model...', 'normal');
        
        if (!processedTrainData) {
            throw new Error('Please preprocess data first');
        }
        
        const numFeatures = processedTrainData.features[0].length;
        
        // Create sequential model
        model = tf.sequential();
        
        // Hidden layer (single hidden layer as required)
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [numFeatures],
            name: 'hidden_layer'
        }));
        
        // Output layer (binary classification)
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
        document.getElementById('modelSummary').style.display = 'block';
        
        let summaryHTML = '<p><strong>Model Architecture:</strong></p>';
        summaryHTML += '<ul>';
        model.layers.forEach((layer, i) => {
            summaryHTML += `<li>Layer ${i+1}: ${layer.name} - ${layer.outputShape}</li>`;
        });
        summaryHTML += '</ul>';
        
        summaryHTML += '<p><strong>Model Summary:</strong></p>';
        summaryHTML += '<pre style="background: #f5f5f5; padding: 10px; overflow: auto;">';
        
        // Count parameters
        let totalParams = 0;
        model.layers.forEach(layer => {
            const weights = layer.getWeights();
            let layerParams = 0;
            weights.forEach(w => {
                layerParams += w.size;
            });
            totalParams += layerParams;
            
            summaryHTML += `${layer.name.padEnd(15)} ${layer.outputShape.toString().padEnd(20)} ${layerParams} params\n`;
        });
        
        summaryHTML += `\nTotal params: ${totalParams}`;
        summaryHTML += '</pre>';
        
        document.getElementById('modelLayers').innerHTML = summaryHTML;
        
        updateStatus('modelStatus', 
            `✅ Model created successfully! ${numFeatures} input features → 16 hidden units → 1 output. Total parameters: ${totalParams}`,
            'success'
        );
        
        // Enable training and evaluation buttons
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('summaryBtn').disabled = false;
        
        console.log('Model created:', model);
        console.log('Input shape:', [null, numFeatures]);
        
    } catch (error) {
        updateStatus('modelStatus', `❌ Model creation error: ${error.message}`, 'error');
        console.error('Model creation error:', error);
    }
}

/**
 * Display model summary in tfjs-vis
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
 * Train the model
 */
async function trainModel() {
    try {
        if (!model || !trainTensors) {
            throw new Error('Please create model and preprocess data first');
        }
        
        updateStatus('trainingStatus', 'Starting training...', 'normal');
        document.getElementById('trainingCharts').style.display = 'block';
        
        isTraining = true;
        stopTrainingRequested = false;
        
        // Enable/disable buttons
        document.getElementById('trainBtn').disabled = true;
        document.getElementById('stopTrainingBtn').disabled = false;
        
        // Create validation split (80/20 stratified)
        const indices = tf.range(0, trainTensors.features.shape[0]);
        const shuffled = tf.util.createShuffledIndices(indices.size);
        
        const splitIdx = Math.floor(indices.size * 0.8);
        const trainIndices = shuffled.slice(0, splitIdx);
        const valIndices = shuffled.slice(splitIdx);
        
        // Extract training data
        const trainFeatures = tf.gather(trainTensors.features, trainIndices);
        const trainTargets = tf.gather(trainTensors.targets, trainIndices);
        
        // Extract validation data
        const valFeatures = tf.gather(trainTensors.features, valIndices);
        const valTargets = tf.gather(trainTensors.targets, valIndices);
        
        // Store validation data for later evaluation
        validationLabels = await valTargets.array();
        
        // Prepare tfjs-vis callbacks
        const metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy'];
        const container = {
            name: 'Training Metrics',
            tab: 'Training',
            styles: { height: '300px' }
        };
        
        const fitCallbacks = tfvis.show.fitCallbacks(container, metrics, {
            callbacks: ['onBatchEnd', 'onEpochEnd'],
            speed: 100
        });
        
        // Add progress callback
        const progressCallback = {
            onEpochEnd: (epoch, logs) => {
                // Update progress bar
                const progress = (epoch + 1) / 50 * 100;
                document.getElementById('trainingProgress').style.width = `${progress}%`;
                
                // Check for early stopping
                if (stopTrainingRequested) {
                    model.stopTraining = true;
                    updateStatus('trainingStatus', 'Training stopped by user', 'warning');
                }
            }
        };
        
        // Train model
        const history = await model.fit(trainFeatures, trainTargets, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valTargets],
            callbacks: [fitCallbacks, progressCallback],
            verbose: 0
        });
        
        trainingHistory = history;
        
        // Get validation predictions for ROC curve
        validationProbs = await model.predict(valFeatures).array();
        
        // Extract first layer weights for feature importance
        const hiddenLayer = model.getLayer('hidden_layer');
        if (hiddenLayer) {
            const weights = hiddenLayer.getWeights();
            if (weights.length > 0) {
                firstLayerWeights = weights[0]; // Weight matrix: [input_features, hidden_units]
                console.log('First layer weights shape:', firstLayerWeights.shape);
            }
        }
        
        // Clean up tensors
        tf.dispose([trainFeatures, trainTargets, valFeatures, valTargets, indices]);
        
        if (stopTrainingRequested) {
            updateStatus('trainingStatus', 'Training stopped early by user.', 'warning');
        } else {
            updateStatus('trainingStatus', 
                `✅ Training completed! Final accuracy: ${(history.history.acc[history.history.acc.length-1] * 100).toFixed(1)}%, ` +
                `Validation accuracy: ${(history.history.val_acc[history.history.val_acc.length-1] * 100).toFixed(1)}%`,
                'success'
            );
        }
        
        // Enable evaluation buttons
        document.getElementById('evaluateBtn').disabled = false;
        document.getElementById('rocBtn').disabled = false;
        document.getElementById('thresholdSlider').disabled = false;
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('saveModelBtn').disabled = false;
        
        console.log('Training completed. History:', history);
        
    } catch (error) {
        updateStatus('trainingStatus', `❌ Training error: ${error.message}`, 'error');
        console.error('Training error:', error);
    } finally {
        isTraining = false;
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('stopTrainingBtn').disabled = true;
        document.getElementById('trainingProgress').style.width = '0%';
    }
}

/**
 * Stop training early
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
 * Calculate ROC curve and AUC
 * @returns {Object} ROC curve data and AUC
 */
function calculateROC() {
    if (!validationProbs || !validationLabels) {
        throw new Error('No validation data available');
    }
    
    // Sort by predicted probability
    const paired = validationProbs.map((prob, i) => ({
        prob: prob[0],
        label: validationLabels[i]
    })).sort((a, b) => b.prob - a.prob);
    
    const thresholds = Array.from({length: 101}, (_, i) => i / 100);
    const tprs = []; // True Positive Rate (Recall)
    const fprs = []; // False Positive Rate
    
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
        
        // Calculate AUC using trapezoidal rule
        if (fpr > prevFPR) {
            area += (fpr - prevFPR) * (tpr + prevTPR) / 2;
            prevFPR = fpr;
            prevTPR = tpr;
        }
    });
    
    return {
        fpr: fprs,
        tpr: tprs,
        thresholds: thresholds,
        auc: area.toFixed(4)
    };
}

/**
 * Show ROC curve using tfjs-vis
 */
function showROCCurve() {
    try {
        if (!validationProbs || !validationLabels) {
            throw new Error('Please train the model first');
        }
        
        const rocData = calculateROC();
        
        // Format data for tfjs-vis
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
            height: 300,
            seriesColors: ['#3498db']
        });
        
        // Display AUC
        updateStatus('evaluationStatus', `ROC Curve displayed. AUC: ${rocData.auc}`, 'success');
        
    } catch (error) {
        updateStatus('evaluationStatus', `❌ ROC curve error: ${error.message}`, 'error');
        console.error('ROC curve error:', error);
    }
}

/**
 * Calculate confusion matrix and metrics for given threshold
 * @param {number} threshold - Classification threshold
 */
function evaluateModel(threshold = 0.5) {
    try {
        if (!validationProbs || !validationLabels) {
            throw new Error('Please train the model first');
        }
        
        // Update threshold display
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
        
        // Display metrics table
        const metricsTableHTML = `
            <table class="evaluation-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td>${metrics.accuracy}</td>
                    <td>(TP+TN) / Total</td>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td>${metrics.precision}</td>
                    <td>TP / (TP+FP)</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td>${metrics.recall}</td>
                    <td>TP / (TP+FN)</td>
                </tr>
                <tr>
                    <td>F1-Score</td>
                    <td>${metrics.f1}</td>
                    <td>2 × (Precision×Recall) / (Precision+Recall)</td>
                </tr>
            </table>
        `;
        
        document.getElementById('metricsTable').innerHTML = `
            <h4>Performance Metrics</h4>
            ${metricsTableHTML}
        `;
        
        // Calculate and display feature importance if weights are available
        if (firstLayerWeights && featureInfo) {
            calculateFeatureImportance();
        }
        
    } catch (error) {
        updateStatus('evaluationStatus', `❌ Evaluation error: ${error.message}`, 'error');
        console.error('Evaluation error:', error);
    }
}

/**
 * Calculate feature importance using Sigmoid Gate
 */
async function calculateFeatureImportance() {
    try {
        if (!firstLayerWeights || !featureInfo || !trainTensors) {
            throw new Error('Missing data for feature importance calculation');
        }
        
        // Get feature ranges from training data
        const featureData = await trainTensors.features.array();
        const numFeatures = featureData[0].length;
        
        // Calculate min and max for each feature
        const featureMins = [];
        const featureMaxs = [];
        
        for (let i = 0; i < numFeatures; i++) {
            const values = featureData.map(row => row[i]);
            featureMins.push(Math.min(...values));
            featureMaxs.push(Math.max(...values));
        }
        
        // Calculate standardized feature range
        const standardizedRanges = [];
        for (let i = 0; i < numFeatures; i++) {
            // Standardize the range difference
            const range = featureMaxs[i] - featureMins[i];
            // For binary features (0 or 1), range is 1
            standardizedRanges.push(range > 1 ? 1 : range);
        }
        
        // Get first layer weights
        const weights = await firstLayerWeights.array();
        
        // Calculate importance: sigmoid(W * standardized_range)
        const importances = [];
        
        for (let i = 0; i < numFeatures; i++) {
            let weightedSum = 0;
            
            // Sum across all hidden units
            for (let j = 0; j < weights[i].length; j++) {
                weightedSum += Math.abs(weights[i][j]) * standardizedRanges[i];
            }
            
            // Apply sigmoid
            const importance = 1 / (1 + Math.exp(-weightedSum));
            importances.push({
                feature: featureInfo.names[i] || `Feature_${i}`,
                importance: importance,
                rawScore: weightedSum
            });
        }
        
        // Sort by importance
        importances.sort((a, b) => b.importance - a.importance);
        
        // Store for later use
        featureImportance = importances;
        
        // Display top 5 features
        const topFeatures = importances.slice(0, 5);
        
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
        
        console.log('Top feature importances:', topFeatures);
        
    } catch (error) {
        console.error('Feature importance calculation error:', error);
        document.getElementById('featureImportance').innerHTML = 
            `<p style="color: #e74c3c;">Error calculating feature importance: ${error.message}</p>`;
    }
}

// ============================================================================
// PREDICTION & EXPORT
// ============================================================================

/**
 * Generate predictions on test data
 */
async function predictTestData() {
    try {
        updateStatus('predictionStatus', 'Generating predictions...', 'normal');
        
        if (!model || !testTensors) {
            throw new Error('Please train the model and load test data first');
        }
        
        // Generate predictions
        const probsTensor = model.predict(testTensors.features);
        testProbabilities = await probsTensor.array();
        
        // Convert to binary predictions with default threshold of 0.5
        const threshold = parseFloat(document.getElementById('thresholdValue').textContent);
        testPredictions = testProbabilities.map(prob => prob[0] >= threshold ? 1 : 0);
        testPassengerIds = testTensors.passengerIds;
        
        // Clean up tensor
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
        
        updateStatus('predictionStatus', 
            `✅ Predictions generated! ${testPredictions.length} samples predicted. ` +
            `Survival rate: ${(testPredictions.filter(p => p === 1).length / testPredictions.length * 100).toFixed(1)}%`,
            'success'
        );
        
        // Enable export buttons
        document.getElementById('exportBtn').disabled = false;
        document.getElementById('exportProbsBtn').disabled = false;
        
        console.log('Predictions generated. Sample:', {
            passengerId: testPassengerIds[0],
            prediction: testPredictions[0],
            probability: testProbabilities[0][0]
        });
        
    } catch (error) {
        updateStatus('predictionStatus', `❌ Prediction error: ${error.message}`, 'error');
        console.error('Prediction error:', error);
    }
}

/**
 * Export submission CSV
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
        updateStatus('predictionStatus', `❌ Export error: ${error.message}`, 'error');
        console.error('Export error:', error);
    }
}

/**
 * Export probabilities CSV
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
        updateStatus('predictionStatus', `❌ Export error: ${error.message}`, 'error');
        console.error('Export error:', error);
    }
}

/**
 * Save model to local downloads
 */
async function saveModel() {
    try {
        if (!model) {
            throw new Error('No model to save');
        }
        
        updateStatus('predictionStatus', 'Saving model...', 'normal');
        
        await model.save('downloads://titanic-tfjs-model');
        
        updateStatus('predictionStatus', '✅ Model saved! Check your downloads folder.', 'success');
        
    } catch (error) {
        updateStatus('predictionStatus', `❌ Model save error: ${error.message}`, 'error');
        console.error('Model save error:', error);
    }
}

// ============================================================================
// EVENT LISTENERS & INITIALIZATION
// ============================================================================

/**
 * Initialize the application
 */
function initializeApp() {
    console.log('Titanic Binary Classifier initializing...');
    
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
        const threshold = parseFloat(this.value);
        evaluateModel(threshold);
    });
    
    // Initial status
    updateStatus('dataStatus', 'Please upload train.csv and test.csv files', 'normal');
    updateStatus('preprocessStatus', 'Load data first to enable preprocessing', 'normal');
    updateStatus('modelStatus', 'Preprocess data first to create model', 'normal');
    updateStatus('trainingStatus', 'Create model first to enable training', 'normal');
    updateStatus('evaluationStatus', 'Train model first to enable evaluation', 'normal');
    updateStatus('predictionStatus', 'Train model first to make predictions', 'normal');
    
    console.log('App initialized. Ready to load Titanic data.');
}

// Initialize when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}
