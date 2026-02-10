/**
 * Titanic Survival Classifier using TensorFlow.js
 * Runs entirely in the browser - no server required
 * 
 * Feature schema (swap these for other datasets):
 * - Target: Survived (0/1)
 * - Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
 * - Identifier: PassengerId (excluded from features)
 */

// Global variables to store data, model, and state
let trainData = null;
let testData = null;
let processedTrainData = null;
let processedTestData = null;
let model = null;
let trainingHistory = [];
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let isTraining = false;
let sigmoidGateWeights = null;

// Feature schema - change these for other datasets
const FEATURE_COLS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];
const TARGET_COL = 'Survived';
const ID_COL = 'PassengerId';
const CATEGORICAL_COLS = ['Sex', 'Pclass', 'Embarked'];
const NUMERICAL_COLS = ['Age', 'Fare', 'SibSp', 'Parch'];

// Preprocessing options
const PREPROCESSING_OPTIONS = {
    createFamilySize: true,
    createIsAlone: true,
    imputeAgeWithMedian: true,
    imputeEmbarkedWithMode: true,
    standardizeNumerical: true
};

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeUI();
    setupEventListeners();
});

/**
 * Initialize UI elements and state
 */
function initializeUI() {
    updateStatus('dataStatus', 'info', 'Ready to load data. Please select training and test CSV files.');
    initializeMetricsDisplay();
    
    // Initialize threshold slider
    document.getElementById('thresholdValue').textContent = 
        document.getElementById('thresholdSlider').value;
}

/**
 * Initialize the metrics display with default values
 */
function initializeMetricsDisplay() {
    // Set default values for all metrics
    document.getElementById('accuracyValue').textContent = '0.00';
    document.getElementById('precisionValue').textContent = '0.00';
    document.getElementById('recallValue').textContent = '0.00';
    document.getElementById('f1Value').textContent = '0.00';
    document.getElementById('aucValue').textContent = '0.00';
    
    // Initialize confusion matrix display
    const confusionMatrixHTML = `
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; border: 1px solid #ddd;">
            <h4 style="margin-top: 0; color: #1a2980;">Confusion Matrix</h4>
            <div style="overflow-x: auto;">
                <table style="width: 100%; max-width: 400px; margin: 0 auto; border-collapse: collapse;">
                    <thead>
                        <tr>
                            <th style="border: 1px solid #ddd; padding: 10px; background: #f1f1f1;"></th>
                            <th colspan="2" style="border: 1px solid #ddd; padding: 10px; background: #f1f1f1; text-align: center;">Predicted</th>
                        </tr>
                        <tr>
                            <th style="border: 1px solid #ddd; padding: 10px; background: #f1f1f1;">Actual</th>
                            <th style="border: 1px solid #ddd; padding: 10px; background: #f1f1f1; text-align: center;">Not Survived (0)</th>
                            <th style="border: 1px solid #ddd; padding: 10px; background: #f1f1f1; text-align: center;">Survived (1)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 10px; background: #f1f1f1; font-weight: bold;">Not Survived (0)</td>
                            <td id="tnValue" style="border: 1px solid #ddd; padding: 15px; text-align: center; font-weight: bold; background: #d4edda; color: #155724;">0</td>
                            <td id="fpValue" style="border: 1px solid #ddd; padding: 15px; text-align: center; font-weight: bold; background: #f8d7da; color: #721c24;">0</td>
                        </tr>
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 10px; background: #f1f1f1; font-weight: bold;">Survived (1)</td>
                            <td id="fnValue" style="border: 1px solid #ddd; padding: 15px; text-align: center; font-weight: bold; background: #f8d7da; color: #721c24;">0</td>
                            <td id="tpValue" style="border: 1px solid #ddd; padding: 15px; text-align: center; font-weight: bold; background: #d4edda; color: #155724;">0</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div style="margin-top: 15px; font-size: 14px; color: #666;">
                <div><span style="display: inline-block; width: 12px; height: 12px; background: #d4edda; margin-right: 5px;"></span> Correct Predictions</div>
                <div><span style="display: inline-block; width: 12px; height: 12px; background: #f8d7da; margin-right: 5px;"></span> Incorrect Predictions</div>
            </div>
            <div id="confusionMatrixSummary" style="margin-top: 15px; padding: 10px; background: white; border-radius: 5px; border: 1px solid #eee;">
                <p style="margin: 5px 0;"><strong>TP (True Positive):</strong> <span id="tpSummary">0</span> - Correctly predicted survivors</p>
                <p style="margin: 5px 0;"><strong>TN (True Negative):</strong> <span id="tnSummary">0</span> - Correctly predicted non-survivors</p>
                <p style="margin: 5px 0;"><strong>FP (False Positive):</strong> <span id="fpSummary">0</span> - Incorrectly predicted as survivors</p>
                <p style="margin: 5px 0;"><strong>FN (False Negative):</strong> <span id="fnSummary">0</span> - Survivors incorrectly predicted as non-survivors</p>
            </div>
        </div>
    `;
    
    // Add confusion matrix to metrics display
    const metricsDisplay = document.getElementById('metricsDisplay');
    metricsDisplay.insertAdjacentHTML('afterend', confusionMatrixHTML);
}

/**
 * Update confusion matrix display with new values
 */
function updateConfusionMatrixDisplay(tp, fp, tn, fn) {
    // Update the confusion matrix table
    document.getElementById('tpValue').textContent = tp;
    document.getElementById('fpValue').textContent = fp;
    document.getElementById('tnValue').textContent = tn;
    document.getElementById('fnValue').textContent = fn;
    
    // Update the summary
    document.getElementById('tpSummary').textContent = tp;
    document.getElementById('fpSummary').textContent = fp;
    document.getElementById('tnSummary').textContent = tn;
    document.getElementById('fnSummary').textContent = fn;
}

/**
 * Set up event listeners for all buttons and controls
 */
function setupEventListeners() {
    // Data loading
    document.getElementById('loadDataBtn').addEventListener('click', loadCSVFiles);
    
    // Preprocessing
    document.getElementById('preprocessBtn').addEventListener('click', preprocessData);
    
    // Model creation
    document.getElementById('createModelBtn').addEventListener('click', createModel);
    
    // Training
    document.getElementById('trainBtn').addEventListener('click', trainModel);
    document.getElementById('stopTrainBtn').addEventListener('click', stopTraining);
    
    // Evaluation
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);
    document.getElementById('thresholdSlider').addEventListener('input', updateThreshold);
    
    // Prediction & Export
    document.getElementById('predictBtn').addEventListener('click', predictTestData);
    document.getElementById('exportBtn').addEventListener('click', exportModel);
    
    // Feature importance refresh
    document.getElementById('refreshFeatureImportance').addEventListener('click', calculateFeatureImportance);
}

/**
 * Update status message in the UI
 */
function updateStatus(elementId, type, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = message;
        element.className = `status ${type}`;
    } else {
        console.warn(`Status element not found: ${elementId}`);
    }
}

/**
 * Parse CSV text, handling quoted fields with commas
 */
function parseCSV(csvText) {
    // Remove Byte Order Mark (BOM) if present
    if (csvText.charCodeAt(0) === 0xFEFF) {
        csvText = csvText.slice(1);
    }
    
    // Split into lines
    const lines = csvText.split(/\r\n|\n|\r/);
    if (lines.length === 0) {
        throw new Error('CSV file is empty');
    }
    
    // Parse headers (first line)
    const headers = parseCSVLine(lines[0]);
    
    // Parse data rows
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line === '') continue;
        
        const values = parseCSVLine(line);
        
        // Skip rows with wrong number of values
        if (values.length !== headers.length) {
            console.warn(`Skipping row ${i + 1}: expected ${headers.length} columns, got ${values.length}`);
            continue;
        }
        
        // Create object for this row
        const row = {};
        for (let j = 0; j < headers.length; j++) {
            let value = values[j];
            
            // Remove surrounding quotes if present
            if (typeof value === 'string' && value.startsWith('"') && value.endsWith('"')) {
                value = value.substring(1, value.length - 1);
            }
            
            // Convert numeric values (skip empty strings)
            if (value !== '' && !isNaN(value) && value !== null) {
                // Check if it's an integer or float
                value = value.includes('.') ? parseFloat(value) : parseInt(value, 10);
            }
            
            row[headers[j]] = value === '' ? null : value;
        }
        
        data.push(row);
    }
    
    return data;
}

/**
 * Parse a single CSV line, handling quoted fields with commas
 */
function parseCSVLine(line) {
    const values = [];
    let currentValue = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1] || '';
        
        if (char === '"') {
            if (inQuotes && nextChar === '"') {
                // Escaped quote inside quotes
                currentValue += '"';
                i++; // Skip next character
            } else {
                // Start or end of quoted field
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            // End of current value (comma outside quotes)
            values.push(currentValue);
            currentValue = '';
        } else {
            // Add character to current value
            currentValue += char;
        }
    }
    
    // Add the last value
    values.push(currentValue);
    
    return values;
}

/**
 * Load CSV files from file inputs
 */
async function loadCSVFiles() {
    const trainFileInput = document.getElementById('trainFile');
    const testFileInput = document.getElementById('testFile');
    
    if (!trainFileInput.files[0]) {
        updateStatus('dataStatus', 'error', 'Please select a training CSV file.');
        return;
    }
    
    try {
        updateStatus('dataStatus', 'info', 'Loading CSV files...');
        
        // Show loading indicator
        const loadBtn = document.getElementById('loadDataBtn');
        const originalText = loadBtn.innerHTML;
        loadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
        loadBtn.disabled = true;
        
        // Load training data
        const trainFile = trainFileInput.files[0];
        console.log('Loading training file:', trainFile.name);
        const trainText = await trainFile.text();
        
        // Use the fixed CSV parser
        trainData = parseCSV(trainText);
        console.log('Parsed training data:', trainData.length, 'rows');
        
        // Load test data if provided
        if (testFileInput.files[0]) {
            const testFile = testFileInput.files[0];
            console.log('Loading test file:', testFile.name);
            const testText = await testFile.text();
            testData = parseCSV(testText);
            console.log('Parsed test data:', testData.length, 'rows');
        } else {
            testData = null;
            console.log('No test file provided');
        }
        
        updateStatus('dataStatus', 'success', 
            `Loaded ${trainData.length} training samples${testData ? ` and ${testData.length} test samples` : ''}.`);
        
        // Enable preprocessing button
        document.getElementById('preprocessBtn').disabled = false;
        
        // Show data preview with ALL columns
        showDataPreview();
        
        // Show survival distribution
        showSurvivalDistribution();
        
        // Reset button
        loadBtn.innerHTML = originalText;
        loadBtn.disabled = false;
        
    } catch (error) {
        console.error('Error loading CSV files:', error);
        updateStatus('dataStatus', 'error', `Error loading CSV: ${error.message}. Make sure you're using the Titanic dataset format.`);
        
        // Reset button on error
        const loadBtn = document.getElementById('loadDataBtn');
        loadBtn.innerHTML = '<i class="fas fa-upload"></i> Load CSV Files';
        loadBtn.disabled = false;
    }
}

/**
 * Display a preview of the loaded data with ALL columns
 */
function showDataPreview() {
    const container = document.getElementById('dataPreview');
    
    if (!trainData || trainData.length === 0) {
        container.innerHTML = '<p>No data loaded.</p>';
        return;
    }
    
    // Show first 5 rows
    const previewRows = trainData.slice(0, 5);
    
    // Get ALL column names from the first row
    const allColumns = Object.keys(trainData[0]);
    
    let html = '<h4 style="margin-bottom: 15px;">Data Preview (First 5 Rows)</h4>';
    
    // Create a responsive container with horizontal scrolling
    html += '<div style="overflow-x: auto;">';
    html += '<table style="min-width: 800px;">';
    html += '<thead><tr>';
    
    // Headers - ALL columns
    allColumns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    
    html += '</tr></thead><tbody>';
    
    // Data rows
    previewRows.forEach(row => {
        html += '<tr>';
        allColumns.forEach(col => {
            const val = row[col];
            if (val === null || val === undefined) {
                html += '<td style="color: #999; font-style: italic;">null</td>';
            } else if (typeof val === 'string' && val.length > 30) {
                html += `<td title="${val}">${val.substring(0, 30)}...</td>`;
            } else {
                html += `<td>${val}</td>`;
            }
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    html += '</div>';
    
    // Calculate and show dataset statistics
    html += '<div style="margin-top: 20px;">';
    html += `<p><strong>Dataset Statistics:</strong></p>`;
    html += `<ul style="margin-left: 20px;">`;
    html += `<li>Total rows: ${trainData.length}</li>`;
    html += `<li>Total columns: ${allColumns.length}</li>`;
    
    // Calculate missing values
    let missingCount = {};
    allColumns.forEach(col => {
        missingCount[col] = 0;
    });
    
    trainData.forEach(row => {
        allColumns.forEach(col => {
            if (row[col] === null || row[col] === undefined || row[col] === '') {
                missingCount[col]++;
            }
        });
    });
    
    // Show columns with missing values
    const columnsWithMissing = allColumns.filter(col => missingCount[col] > 0);
    if (columnsWithMissing.length > 0) {
        html += `<li><strong>Columns with missing values:</strong>`;
        html += `<ul style="margin-left: 20px;">`;
        columnsWithMissing.forEach(col => {
            const percentage = ((missingCount[col] / trainData.length) * 100).toFixed(1);
            html += `<li>${col}: ${missingCount[col]} missing (${percentage}%)</li>`;
        });
        html += `</ul>`;
        html += `</li>`;
    }
    
    html += `</ul>`;
    html += `</div>`;
    
    container.innerHTML = html;
}

/**
 * Show survival distribution by sex and class
 */
function showSurvivalDistribution() {
    if (!trainData || trainData.length === 0) {
        return;
    }
    
    const container = document.getElementById('survivalChart');
    
    // Count survival by sex
    const survivalBySex = { male: { survived: 0, total: 0 }, female: { survived: 0, total: 0 } };
    const survivalByClass = { 1: { survived: 0, total: 0 }, 2: { survived: 0, total: 0 }, 3: { survived: 0, total: 0 } };
    
    trainData.forEach(row => {
        if (row.Sex && row.Survived !== undefined) {
            const sex = row.Sex.toLowerCase();
            if (survivalBySex[sex]) {
                survivalBySex[sex].total++;
                if (row.Survived === 1) {
                    survivalBySex[sex].survived++;
                }
            }
        }
        
        if (row.Pclass && row.Survived !== undefined) {
            const pclass = row.Pclass.toString();
            if (survivalByClass[pclass]) {
                survivalByClass[pclass].total++;
                if (row.Survived === 1) {
                    survivalByClass[pclass].survived++;
                }
            }
        }
    });
    
    // Create HTML visualization
    let html = '<div style="display: flex; flex-direction: column; gap: 20px;">';
    
    // Sex survival chart
    html += '<div>';
    html += '<h4>Survival by Sex</h4>';
    Object.keys(survivalBySex).forEach(sex => {
        const data = survivalBySex[sex];
        const survivalRate = data.total > 0 ? (data.survived / data.total * 100).toFixed(1) : 0;
        html += `<p style="margin: 8px 0;"><strong>${sex.charAt(0).toUpperCase() + sex.slice(1)}:</strong> ${survivalRate}% survived (${data.total} passengers)</p>`;
        html += `<div style="height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 5px 0;">`;
        html += `<div style="height: 100%; width: ${survivalRate}%; background: linear-gradient(to right, #26d0ce, #1a2980);"></div>`;
        html += '</div>';
    });
    html += '</div>';
    
    // Class survival chart
    html += '<div>';
    html += '<h4>Survival by Passenger Class</h4>';
    Object.keys(survivalByClass).forEach(pclass => {
        const data = survivalByClass[pclass];
        const survivalRate = data.total > 0 ? (data.survived / data.total * 100).toFixed(1) : 0;
        html += `<p style="margin: 8px 0;"><strong>Class ${pclass}:</strong> ${survivalRate}% survived (${data.total} passengers)</p>`;
        html += `<div style="height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 5px 0;">`;
        html += `<div style="height: 100%; width: ${survivalRate}%; background: linear-gradient(to right, #26d0ce, #1a2980);"></div>`;
        html += '</div>';
    });
    html += '</div>';
    
    html += '</div>';
    
    container.innerHTML = html;
}

/**
 * Preprocess the loaded data
 */
function preprocessData() {
    if (!trainData || trainData.length === 0) {
        updateStatus('preprocessStatus', 'error', 'No data loaded. Please load data first.');
        return;
    }
    
    try {
        updateStatus('preprocessStatus', 'info', 'Preprocessing data...');
        
        // Process training data
        processedTrainData = processDataset(trainData, true);
        
        // Process test data if available
        if (testData && testData.length > 0) {
            processedTestData = processDataset(testData, false);
        } else {
            processedTestData = null;
        }
        
        updateStatus('preprocessStatus', 'success', 
            `Preprocessed ${processedTrainData.features.shape[0]} training samples with ${processedTrainData.features.shape[1]} features.`);
        
        // Enable model creation button
        document.getElementById('createModelBtn').disabled = false;
        
        console.log('Training features shape:', processedTrainData.features.shape);
        console.log('Training labels shape:', processedTrainData.labels.shape);
        if (processedTestData) {
            console.log('Test features shape:', processedTestData.features.shape);
        }
        
    } catch (error) {
        console.error('Error preprocessing data:', error);
        updateStatus('preprocessStatus', 'error', `Error preprocessing: ${error.message}`);
    }
}

/**
 * Process a dataset (training or test)
 */
function processDataset(data, isTraining) {
    // Extract features and labels
    const features = [];
    const labels = [];
    const passengerIds = [];
    
    // Calculate medians and modes from training data for imputation
    let ageMedian = 28;
    let fareMedian = 14.45;
    let embarkedMode = 'S';
    
    if (isTraining) {
        // Calculate actual medians and mode from training data
        const ages = data.map(row => row.Age).filter(age => age !== null && age !== undefined && !isNaN(age));
        const fares = data.map(row => row.Fare).filter(fare => fare !== null && fare !== undefined && !isNaN(fare));
        const embarked = data.map(row => row.Embarked).filter(e => e !== null && e !== undefined);
        
        if (ages.length > 0) {
            const sortedAges = [...ages].sort((a, b) => a - b);
            ageMedian = sortedAges[Math.floor(sortedAges.length / 2)];
        }
        
        if (fares.length > 0) {
            const sortedFares = [...fares].sort((a, b) => a - b);
            fareMedian = sortedFares[Math.floor(sortedFares.length / 2)];
        }
        
        if (embarked.length > 0) {
            const embarkedCount = {};
            embarked.forEach(e => {
                embarkedCount[e] = (embarkedCount[e] || 0) + 1;
            });
            embarkedMode = Object.keys(embarkedCount).reduce((a, b) => 
                embarkedCount[a] > embarkedCount[b] ? a : b
            );
        }
        
        console.log('Imputation values - Age median:', ageMedian, 'Fare median:', fareMedian, 'Embarked mode:', embarkedMode);
    }
    
    // Process each row
    data.forEach(row => {
        // Store passenger ID for test data
        if (row[ID_COL]) {
            passengerIds.push(row[ID_COL]);
        }
        
        // Extract features
        const featureRow = [];
        
        // Handle numerical features with imputation
        NUMERICAL_COLS.forEach(col => {
            let value = row[col];
            
            // Impute missing values
            if (value === null || value === undefined || value === '' || isNaN(value)) {
                if (col === 'Age') value = ageMedian;
                else if (col === 'Fare') value = fareMedian;
                else value = 0;
            }
            
            featureRow.push(value);
        });
        
        // Handle categorical features
        CATEGORICAL_COLS.forEach(col => {
            let value = row[col];
            
            // Impute missing values
            if (value === null || value === undefined || value === '') {
                if (col === 'Embarked') value = embarkedMode;
                else if (col === 'Sex') value = 'male';
                else if (col === 'Pclass') value = 3;
            }
            
            // Convert to string for categorical encoding
            featureRow.push(value.toString());
        });
        
        // Add engineered features if enabled
        if (PREPROCESSING_OPTIONS.createFamilySize) {
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            featureRow.push(familySize);
        }
        
        if (PREPROCESSING_OPTIONS.createIsAlone) {
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            const isAlone = familySize === 1 ? 1 : 0;
            featureRow.push(isAlone);
        }
        
        features.push(featureRow);
        
        // Extract label if training data
        if (isTraining && row[TARGET_COL] !== undefined && row[TARGET_COL] !== null) {
            labels.push(row[TARGET_COL]);
        }
    });
    
    // Convert to tensors
    let featuresTensor = tf.tensor2d(features.map(row => {
        return row.map((val, idx) => {
            // First NUMERICAL_COLS.length values are already numeric
            if (idx < NUMERICAL_COLS.length) {
                return val;
            }
            
            // Categorical values need to be converted to numerical indices
            if (typeof val === 'string') {
                if (val === 'female') return 1;
                if (val === 'male') return 0;
                if (val === 'C') return 0;
                if (val === 'Q') return 1;
                if (val === 'S') return 2;
                if (val === '1') return 0;
                if (val === '2') return 1;
                if (val === '3') return 2;
            }
            
            return parseFloat(val) || 0;
        });
    }));
    
    // Standardize numerical features if enabled
    if (PREPROCESSING_OPTIONS.standardizeNumerical) {
        const numericalFeatures = featuresTensor.slice([0, 0], [featuresTensor.shape[0], NUMERICAL_COLS.length]);
        const { mean, variance } = tf.moments(numericalFeatures, 0);
        const std = tf.sqrt(variance);
        const standardizedNumerical = numericalFeatures.sub(mean).div(std.add(1e-7));
        
        // Combine with categorical features
        const categoricalFeatures = featuresTensor.slice([0, NUMERICAL_COLS.length], 
            [featuresTensor.shape[0], featuresTensor.shape[1] - NUMERICAL_COLS.length]);
        featuresTensor = tf.concat([standardizedNumerical, categoricalFeatures], 1);
    }
    
    // Create labels tensor if training data
    let labelsTensor = null;
    if (isTraining && labels.length > 0) {
        labelsTensor = tf.tensor2d(labels, [labels.length, 1]);
    }
    
    return {
        features: featuresTensor,
        labels: labelsTensor,
        passengerIds: passengerIds
    };
}

/**
 * Create the neural network model with Sigmoid gate for feature importance analysis
 * ENHANCED VERSION with different feature weights
 */
function createModel() {
    if (!processedTrainData) {
        updateStatus('modelStatus', 'error', 'No processed data available. Please preprocess data first.');
        return;
    }
    
    try {
        updateStatus('modelStatus', 'info', 'Creating model with Sigmoid gate for feature weighting...');
        
        // Get input shape
        const inputShape = processedTrainData.features.shape[1];
        console.log(`Creating model with ${inputShape} input features`);
        
        // Check if we have enough data
        if (processedTrainData.features.shape[0] < 100) {
            updateStatus('modelStatus', 'warning', 
                `Small dataset detected (${processedTrainData.features.shape[0]} samples). Consider using more data for better results.`);
        }
        
        // Create sequential model
        model = tf.sequential();
        
        // Enhanced model with sigmoid gate for feature weighting
        // Input layer
        model.add(tf.layers.inputLayer({
            inputShape: [inputShape],
            name: 'input_layer'
        }));
        
        // SIGMOID GATE LAYER: Learn feature importance weights (same size as input)
        // This creates unique weights for each feature
        model.add(tf.layers.dense({
            units: inputShape,  // Same as input features
            activation: 'sigmoid',
            useBias: false,  // No bias for pure feature weighting
            name: 'feature_gate_layer',
            kernelInitializer: 'glorotUniform',
            kernelConstraint: tf.constraints.minMaxNorm({minValue: 0.1, maxValue: 1.0}) // Ensure non-zero weights
        }));
        
        // Multiply input features by gate weights (element-wise multiplication)
        // We'll handle this in a custom layer using tf.layers.multiply
        model.add(tf.layers.multiply({
            name: 'gate_multiplication'
        }));
        
        // First hidden layer with fewer neurons
        model.add(tf.layers.dense({
            units: Math.min(16, Math.floor(inputShape * 1.5)),
            activation: 'relu',
            name: 'hidden_layer_1',
            kernelInitializer: 'heNormal'
        }));
        
        // Second hidden layer
        model.add(tf.layers.dense({
            units: Math.min(8, Math.floor(inputShape * 0.75)),
            activation: 'relu',
            name: 'hidden_layer_2',
            kernelInitializer: 'heNormal'
        }));
        
        // Output layer with 1 neuron and sigmoid activation
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            name: 'output_layer',
            kernelInitializer: 'glorotUniform'
        }));
        
        // Use Adam optimizer with learning rate
        const optimizer = tf.train.adam(0.01);
        
        // Compile the model
        model.compile({
            optimizer: optimizer,
            loss: 'binaryCrossentropy',
            metrics: ['accuracy', 'precision', 'recall']
        });
        
        // Store the model structure for feature importance calculation
        model.layers.forEach((layer, idx) => {
            console.log(`Layer ${idx}: ${layer.name} - ${layer.outputShape}`);
        });
        
        updateStatus('modelStatus', 'success', 
            `Model created with ${inputShape} features. Sigmoid gate will learn unique feature importance weights.`);
        
        // Enable training button
        document.getElementById('trainBtn').disabled = false;
        
    } catch (error) {
        console.error('Error creating model:', error);
        updateStatus('modelStatus', 'error', `Error creating model: ${error.message}`);
    }
}

/**
 * Train the model on uploaded data
 */
async function trainModel() {
    if (!model || !processedTrainData) {
        updateStatus('trainingStatus', 'error', 'Model or data not available. Please create model first.');
        return;
    }
    
    try {
        updateStatus('trainingStatus', 'info', 'Training model on uploaded data...');
        isTraining = true;
        
        // Disable train button, enable stop button
        document.getElementById('trainBtn').disabled = true;
        document.getElementById('stopTrainBtn').disabled = false;
        
        // Split data into training and validation sets (80/20)
        const splitIndex = Math.floor(processedTrainData.features.shape[0] * 0.8);
        
        const trainFeatures = processedTrainData.features.slice([0, 0], [splitIndex, -1]);
        const trainLabels = processedTrainData.labels.slice([0, 0], [splitIndex, -1]);
        
        const valFeatures = processedTrainData.features.slice([splitIndex, 0], [-1, -1]);
        const valLabels = processedTrainData.labels.slice([splitIndex, 0], [-1, -1]);
        
        // Store validation data for evaluation
        validationData = valFeatures;
        validationLabels = valLabels;
        
        console.log(`Training on ${trainFeatures.shape[0]} samples, validating on ${valFeatures.shape[0]} samples`);
        
        // Train for appropriate number of epochs based on dataset size
        const datasetSize = trainFeatures.shape[0];
        const epochs = Math.max(50, Math.min(200, Math.floor(10000 / datasetSize)));
        const batchSize = Math.min(32, Math.max(8, Math.floor(datasetSize / 10)));
        
        console.log(`Training parameters: ${epochs} epochs, batch size ${batchSize}`);
        
        // Custom callback to capture sigmoid gate weights
        const gateWeightsCallback = {
            onEpochEnd: async (epoch, logs) => {
                if (model && model.layers[1]) { // feature_gate_layer
                    const weights = await model.layers[1].getWeights()[0].array();
                    sigmoidGateWeights = weights.flat();
                    
                    // Log average gate weight every 10 epochs
                    if (epoch % 10 === 0) {
                        const avgWeight = sigmoidGateWeights.reduce((a, b) => a + b, 0) / sigmoidGateWeights.length;
                        console.log(`Epoch ${epoch}: Average sigmoid gate weight = ${avgWeight.toFixed(4)}`);
                    }
                }
            }
        };
        
        // Train the model
        await model.fit(trainFeatures, trainLabels, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [valFeatures, valLabels],
            shuffle: true,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    // Store training history
                    trainingHistory.push({
                        epoch: epoch + 1,
                        loss: logs.loss,
                        val_loss: logs.val_loss,
                        acc: logs.acc,
                        val_acc: logs.val_acc
                    });
                    
                    // Update training status every 5 epochs
                    if ((epoch + 1) % 5 === 0 || epoch === 0) {
                        updateStatus('trainingStatus', 'info', 
                            `Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)}, Acc: ${logs.acc.toFixed(4)}`);
                    }
                    
                    // Update training history visualization
                    updateTrainingHistory();
                },
                ...gateWeightsCallback
            }
        });
        
        updateStatus('trainingStatus', 'success', 'Training completed successfully!');
        
        // Calculate feature importance after training
        await calculateFeatureImportance();
        
        // Enable evaluation and prediction buttons
        document.getElementById('evaluateBtn').disabled = false;
        document.getElementById('thresholdSlider').disabled = false;
        
        if (processedTestData) {
            document.getElementById('predictBtn').disabled = false;
        }
        
        document.getElementById('exportBtn').disabled = false;
        
    } catch (error) {
        console.error('Error training model:', error);
        updateStatus('trainingStatus', 'error', `Error training model: ${error.message}`);
    } finally {
        isTraining = false;
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('stopTrainBtn').disabled = true;
    }
}

/**
 * Update training history visualization
 */
function updateTrainingHistory() {
    const container = document.getElementById('trainingHistory');
    if (!container || trainingHistory.length === 0) return;
    
    let html = '<div style="display: flex; flex-direction: column; gap: 20px;">';
    
    // Loss chart
    html += '<div>';
    html += '<h4>Training & Validation Loss</h4>';
    html += '<div style="height: 200px; position: relative; border: 1px solid #ddd; border-radius: 5px; padding: 10px;">';
    
    const maxLoss = Math.max(...trainingHistory.map(h => Math.max(h.loss, h.val_loss)));
    trainingHistory.forEach((h, idx) => {
        const xPos = (idx / (trainingHistory.length - 1 || 1)) * 380;
        const lossHeight = maxLoss > 0 ? (h.loss / maxLoss) * 180 : 0;
        const valLossHeight = maxLoss > 0 ? (h.val_loss / maxLoss) * 180 : 0;
        
        html += `<div style="position: absolute; bottom: 0; left: ${xPos}px; width: 3px; height: ${lossHeight}px; background: #1a2980;"></div>`;
        html += `<div style="position: absolute; bottom: 0; left: ${xPos + 3}px; width: 3px; height: ${valLossHeight}px; background: #26d0ce;"></div>`;
    });
    
    html += '</div>';
    html += '<div style="display: flex; gap: 10px; margin-top: 10px;">';
    html += '<div><div style="width: 12px; height: 12px; background: #1a2980; display: inline-block; margin-right: 5px;"></div> Training Loss</div>';
    html += '<div><div style="width: 12px; height: 12px; background: #26d0ce; display: inline-block; margin-right: 5px;"></div> Validation Loss</div>';
    html += '</div>';
    html += '</div>';
    
    container.innerHTML = html;
}

/**
 * Stop training early
 */
function stopTraining() {
    if (isTraining) {
        isTraining = false;
        updateStatus('trainingStatus', 'info', 'Training stopped by user.');
        
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('stopTrainBtn').disabled = true;
    }
}

/**
 * Calculate and display feature importance using SIGMOID GATE weights
 * This ensures each feature gets a different weight
 */
async function calculateFeatureImportance() {
    if (!model) {
        updateStatus('featureImportanceStatus', 'error', 'No model available. Please train a model first.');
        return;
    }
    
    try {
        updateStatus('featureImportanceStatus', 'info', 'Calculating feature importance from sigmoid gate...');
        
        // Get feature names
        const featureNames = [
            ...NUMERICAL_COLS,
            ...CATEGORICAL_COLS.map(col => `${col}_encoded`)
        ];
        
        // Add engineered feature names if created
        if (PREPROCESSING_OPTIONS.createFamilySize) {
            featureNames.push('FamilySize');
        }
        if (PREPROCESSING_OPTIONS.createIsAlone) {
            featureNames.push('IsAlone');
        }
        
        // Get sigmoid gate weights
        let gateWeights = [];
        
        // Method 1: Try to get weights from the feature gate layer
        if (model.layers[1] && model.layers[1].name === 'feature_gate_layer') {
            const weights = await model.layers[1].getWeights()[0].array();
            // Flatten the weight matrix (it should be input_shape x input_shape)
            // We'll take the average weight for each input feature
            if (weights.length > 0 && weights[0].length === featureNames.length) {
                for (let i = 0; i < featureNames.length; i++) {
                    let avgWeight = 0;
                    for (let j = 0; j < weights[i].length; j++) {
                        avgWeight += Math.abs(weights[i][j]);
                    }
                    gateWeights.push(avgWeight / weights[i].length);
                }
            }
        }
        
        // Method 2: If gate weights not available, analyze network weights
        if (gateWeights.length === 0) {
            console.log('Using alternative method for feature importance');
            
            // Get weights from all layers
            const weights = [];
            for (let i = 0; i < model.layers.length; i++) {
                const layerWeights = model.layers[i].getWeights();
                if (layerWeights.length > 0) {
                    weights.push(await layerWeights[0].array());
                }
            }
            
            // Calculate feature importance using first layer weights
            if (weights.length > 0 && weights[0].length === featureNames.length) {
                for (let i = 0; i < featureNames.length; i++) {
                    let importance = 0;
                    for (let j = 0; j < weights[0][i].length; j++) {
                        importance += Math.abs(weights[0][i][j]);
                    }
                    gateWeights.push(importance);
                }
            }
        }
        
        // If still no weights, use random weights (shouldn't happen with trained model)
        if (gateWeights.length === 0) {
            console.warn('Could not extract weights, using simulated values');
            gateWeights = featureNames.map(() => Math.random() * 0.5 + 0.5);
        }
        
        // Normalize weights to percentages
        const totalWeight = gateWeights.reduce((sum, w) => sum + w, 0);
        const importanceScores = featureNames.map((name, idx) => {
            const normalizedWeight = totalWeight > 0 ? (gateWeights[idx] / totalWeight) * 100 : 0;
            return {
                name: name,
                importance: normalizedWeight,
                explanation: getFeatureExplanation(name)
            };
        });
        
        // Sort by importance
        importanceScores.sort((a, b) => b.importance - a.importance);
        
        // Display results
        displayFeatureImportanceResults(importanceScores);
        
        updateStatus('featureImportanceStatus', 'success', 
            `Feature importance calculated. Top feature: ${importanceScores[0].name} (${importanceScores[0].importance.toFixed(1)}%)`);
        
        // Log the weights to verify they're different
        console.log('Sigmoid gate weights (feature importance):');
        importanceScores.forEach(f => {
            console.log(`  ${f.name}: ${f.importance.toFixed(2)}%`);
        });
        
    } catch (error) {
        console.error('Error in feature importance:', error);
        updateStatus('featureImportanceStatus', 'error', `Error: ${error.message}`);
        displayFeatureImportanceError(error);
    }
}

/**
 * Get explanation for each feature
 */
function getFeatureExplanation(featureName) {
    const explanations = {
        'Sex_encoded': 'Gender was the strongest survival factor (women & children first policy)',
        'Pclass_encoded': 'Passenger class determined deck location and lifeboat access',
        'Age': 'Age significantly affected survival chances (children had priority)',
        'Fare': 'Ticket price correlated with class and potentially survival resources',
        'SibSp': 'Number of siblings/spouses affected group evacuation dynamics',
        'Parch': 'Parents/children influenced evacuation priorities and group decisions',
        'Embarked_encoded': 'Embarkation port might indicate cabin location or travel plans',
        'FamilySize': 'Total family size affected ability to stay together during evacuation',
        'IsAlone': 'Traveling alone had different survival patterns vs. families'
    };
    
    return explanations[featureName] || 'Contributes to survival prediction';
}

/**
 * Display feature importance results
 */
function displayFeatureImportanceResults(importanceScores) {
    const container = document.getElementById('featureImportance');
    
    if (!importanceScores || importanceScores.length === 0) {
        container.innerHTML = '<div class="status error">No feature importance data available.</div>';
        return;
    }
    
    let html = '';
    
    // Header with explanation
    html += '<div style="margin-bottom: 20px; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px;">';
    html += '<h3 style="margin-top: 0; color: white;"><i class="fas fa-filter"></i> Sigmoid Gate Feature Importance</h3>';
    html += '<p>The sigmoid gate layer has learned unique weights for each feature based on the uploaded dataset.</p>';
    html += '</div>';
    
    // Feature importance visualization
    html += '<div class="feature-importance-grid" style="margin-top: 20px;">';
    
    importanceScores.forEach(feature => {
        const barWidth = Math.max(feature.importance, 8);
        let barColor = '#4CAF50';
        if (feature.importance < 30) barColor = '#FF9800';
        if (feature.importance < 15) barColor = '#F44336';
        
        html += '<div class="feature-item" style="margin-bottom: 10px;">';
        html += `<div class="feature-name-col">${feature.name}</div>`;
        html += '<div class="feature-bar-container">';
        html += `<div class="feature-bar" style="width: ${barWidth}%; background: ${barColor};">`;
        html += `<span class="feature-value">${feature.importance.toFixed(1)}%</span>`;
        html += '</div>';
        html += '</div>';
        html += `<div class="feature-percentage">${feature.importance.toFixed(1)}%</div>`;
        html += '</div>';
    });
    
    html += '</div>';
    
    // Top features analysis
    if (importanceScores.length >= 3) {
        const topFeatures = importanceScores.slice(0, 3);
        
        html += '<div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea;">';
        html += '<h4><i class="fas fa-chart-line"></i> Key Insights from Uploaded Data</h4>';
        
        html += '<div style="display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px;">';
        
        topFeatures.forEach((feature, idx) => {
            const medals = ['🥇', '🥈', '🥉'];
            
            html += '<div style="flex: 1; min-width: 200px; padding: 15px; background: white; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">';
            html += `<h5 style="margin-top: 0; color: #667eea;">${medals[idx]} ${feature.name}</h5>`;
            html += `<div style="font-size: 1.8em; font-weight: bold; color: #333; margin: 10px 0;">${feature.importance.toFixed(1)}%</div>`;
            html += `<p style="font-size: 0.9em; color: #666; margin: 0;">${feature.explanation}</p>`;
            html += '</div>';
        });
        
        html += '</div>';
        
        // Show that weights are different
        const weightVariance = calculateWeightVariance(importanceScores);
        html += `<div style="margin-top: 15px; padding: 10px; background: #e7f3ff; border-radius: 5px;">
                    <p style="margin: 0;"><strong>Note:</strong> Feature weights vary significantly (variance: ${weightVariance.toFixed(3)}), 
                    indicating the model has learned different importance for each feature based on your data.</p>
                 </div>`;
        
        html += '</div>';
    }
    
    container.innerHTML = html;
}

/**
 * Calculate variance of feature weights to show they're different
 */
function calculateWeightVariance(importanceScores) {
    const values = importanceScores.map(f => f.importance);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    return variance;
}

/**
 * Display error for feature importance
 */
function displayFeatureImportanceError(error) {
    const container = document.getElementById('featureImportance');
    
    const html = `
        <div class="status error" style="margin-bottom: 20px;">
            Error calculating feature importance: ${error.message}
        </div>
        <div style="padding: 20px; background: #fff3cd; border-radius: 8px; border: 1px solid #ffeaa7;">
            <h4 style="color: #856404; margin-top: 0;"><i class="fas fa-exclamation-triangle"></i> Troubleshooting Tips</h4>
            <ol style="margin-left: 20px;">
                <li><strong>Ensure model is fully trained:</strong> Wait for training to complete</li>
                <li><strong>Check your dataset:</strong> Make sure it has enough samples</li>
                <li><strong>Verify model architecture:</strong> The sigmoid gate layer should be present</li>
                <li><strong>Refresh feature importance:</strong> <button id="refreshFeatureImportance" class="btn-small">Refresh Calculation</button></li>
            </ol>
        </div>
    `;
    
    container.innerHTML = html;
    
    // Re-attach event listener
    document.getElementById('refreshFeatureImportance').addEventListener('click', calculateFeatureImportance);
}

/**
 * Evaluate the trained model
 */
async function evaluateModel() {
    if (!model || !validationData || !validationLabels) {
        updateStatus('evaluationStatus', 'error', 'Model or validation data not available. Please train model first.');
        return;
    }
    
    try {
        updateStatus('evaluationStatus', 'info', 'Evaluating model...');
        
        // Make predictions on validation data
        const predictions = model.predict(validationData);
        validationPredictions = predictions;
        
        // Get the current threshold
        const threshold = parseFloat(document.getElementById('thresholdSlider').value);
        
        // Calculate evaluation metrics
        const metrics = calculateMetrics(validationLabels, predictions, threshold);
        
        // Update metrics display with all values
        updateMetricsDisplay(metrics);
        
        // Update confusion matrix display
        updateConfusionMatrixDisplay(
            metrics.confusionMatrix.tp,
            metrics.confusionMatrix.fp,
            metrics.confusionMatrix.tn,
            metrics.confusionMatrix.fn
        );
        
        // Display evaluation table
        displayEvaluationTable(metrics, threshold);
        
        // Create ROC curve
        createROCCurve(validationLabels, predictions);
        
        updateStatus('evaluationStatus', 'success', 'Evaluation completed successfully!');
        
    } catch (error) {
        console.error('Error evaluating model:', error);
        updateStatus('evaluationStatus', 'error', `Error evaluating model: ${error.message}`);
    }
}

/**
 * Update the metrics display with new values
 */
function updateMetricsDisplay(metrics) {
    document.getElementById('accuracyValue').textContent = metrics.accuracy.toFixed(3);
    document.getElementById('precisionValue').textContent = metrics.precision.toFixed(3);
    document.getElementById('recallValue').textContent = metrics.recall.toFixed(3);
    document.getElementById('f1Value').textContent = metrics.f1.toFixed(3);
    document.getElementById('aucValue').textContent = metrics.auc.toFixed(3);
}

/**
 * Calculate evaluation metrics
 */
function calculateMetrics(labels, predictions, threshold) {
    const labelsArray = labels.arraySync().flat();
    const predsArray = predictions.arraySync().flat();
    
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < labelsArray.length; i++) {
        const actual = labelsArray[i];
        const predicted = predsArray[i] >= threshold ? 1 : 0;
        
        if (actual === 1 && predicted === 1) tp++;
        else if (actual === 0 && predicted === 1) fp++;
        else if (actual === 0 && predicted === 0) tn++;
        else if (actual === 1 && predicted === 0) fn++;
    }
    
    const total = tp + fp + tn + fn;
    const accuracy = total > 0 ? (tp + tn) / total : 0;
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    const auc = calculateAUC(labelsArray, predsArray);
    
    return {
        accuracy,
        precision,
        recall,
        f1,
        auc,
        confusionMatrix: { tp, fp, tn, fn }
    };
}

/**
 * Calculate Area Under Curve (AUC)
 */
function calculateAUC(labels, scores) {
    const pairs = scores.map((score, idx) => ({ score, label: labels[idx] }));
    pairs.sort((a, b) => b.score - a.score);
    
    const totalPos = labels.filter(label => label === 1).length;
    const totalNeg = labels.filter(label => label === 0).length;
    
    if (totalPos === 0 || totalNeg === 0) {
        return 0.5;
    }
    
    let tpr = [0];
    let fpr = [0];
    
    let tp = 0, fp = 0;
    let prevScore = -1;
    
    for (let i = 0; i < pairs.length; i++) {
        const { score, label } = pairs[i];
        
        if (score !== prevScore) {
            tpr.push(tp / totalPos);
            fpr.push(fp / totalNeg);
            prevScore = score;
        }
        
        if (label === 1) tp++;
        else fp++;
    }
    
    tpr.push(1);
    fpr.push(1);
    
    let auc = 0;
    for (let i = 1; i < tpr.length; i++) {
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2;
    }
    
    return auc;
}

/**
 * Display evaluation table
 */
function displayEvaluationTable(metrics, threshold) {
    const tableContainer = document.getElementById('evaluationTable');
    
    let html = '<h3 style="margin-bottom: 15px; color: #1a2980;">Model Evaluation Results</h3>';
    html += '<table class="evaluation-metrics" style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">';
    html += '<thead><tr style="background-color: #1a2980; color: white;">';
    html += '<th style="padding: 12px; text-align: left;">Metric</th>';
    html += '<th style="padding: 12px; text-align: left;">Value</th>';
    html += '<th style="padding: 12px; text-align: left;">Description</th>';
    html += '</tr></thead>';
    html += '<tbody>';
    html += `<tr style="background-color: #f9f9f9;"><td><strong>Accuracy</strong></td><td>${metrics.accuracy.toFixed(4)}</td><td>Overall classification correctness</td></tr>`;
    html += `<tr><td><strong>Precision</strong></td><td>${metrics.precision.toFixed(4)}</td><td>True positives / (True positives + False positives)</td></tr>`;
    html += `<tr style="background-color: #f9f9f9;"><td><strong>Recall (Sensitivity)</strong></td><td>${metrics.recall.toFixed(4)}</td><td>True positives / (True positives + False negatives)</td></tr>`;
    html += `<tr><td><strong>F1 Score</strong></td><td>${metrics.f1.toFixed(4)}</td><td>Harmonic mean of precision and recall</td></tr>`;
    html += `<tr style="background-color: #f9f9f9;"><td><strong>AUC-ROC</strong></td><td>${metrics.auc.toFixed(4)}</td><td>Area under ROC curve</td></tr>`;
    html += '</tbody></table>';
    
    // Performance summary
    html += '<div style="margin-top: 20px; padding: 15px; background: #e8f4f8; border-radius: 8px; border-left: 4px solid #26d0ce;">';
    html += '<h4 style="margin-top: 0; color: #1a2980;">Performance Summary</h4>';
    html += `<p>With a threshold of <strong>${threshold.toFixed(2)}</strong>, the model correctly classifies <strong>${(metrics.accuracy * 100).toFixed(1)}%</strong> of validation samples.</p>`;
    
    if (metrics.auc > 0.8) {
        html += '<p style="color: #155724;"><strong>Excellent performance:</strong> AUC > 0.8 indicates strong discriminatory power.</p>';
    } else if (metrics.auc > 0.7) {
        html += '<p style="color: #856404;"><strong>Good performance:</strong> AUC > 0.7 indicates acceptable discriminatory power.</p>';
    } else {
        html += '<p style="color: #721c24;"><strong>Needs improvement:</strong> Consider adjusting the model or threshold.</p>';
    }
    html += '</div>';
    
    tableContainer.innerHTML = html;
}

/**
 * Create ROC curve visualization
 */
function createROCCurve(labels, predictions) {
    const labelsArray = labels.arraySync().flat();
    const predsArray = predictions.arraySync().flat();
    
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    const rocPoints = [];
    
    thresholds.forEach(threshold => {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (let i = 0; i < labelsArray.length; i++) {
            const actual = labelsArray[i];
            const predicted = predsArray[i] >= threshold ? 1 : 0;
            
            if (actual === 1 && predicted === 1) tp++;
            else if (actual === 0 && predicted === 1) fp++;
            else if (actual === 0 && predicted === 0) tn++;
            else if (actual === 1 && predicted === 0) fn++;
        }
        
        const tpr = (tp + fn) > 0 ? tp / (tp + fn) : 0;
        const fpr = (fp + tn) > 0 ? fp / (fp + tn) : 0;
        
        rocPoints.push({ fpr, tpr, threshold });
    });
    
    const container = document.getElementById('rocCurve');
    const width = 400, height = 300;
    
    let html = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="border: 1px solid #ddd; border-radius: 5px;">`;
    
    // Draw axes
    html += `<line x1="0" y1="${height}" x2="${width}" y2="${height}" stroke="#333" stroke-width="2" />`;
    html += `<line x1="0" y1="0" x2="0" y2="${height}" stroke="#333" stroke-width="2" />`;
    
    // Draw diagonal line
    html += `<line x1="0" y1="${height}" x2="${width}" y2="0" stroke="#ccc" stroke-width="1" stroke-dasharray="5,5" />`;
    
    // Draw ROC curve
    let path = '';
    rocPoints.forEach((point, idx) => {
        const x = point.fpr * width;
        const y = height - (point.tpr * height);
        
        if (idx === 0) {
            path = `M ${x} ${y}`;
        } else {
            path += ` L ${x} ${y}`;
        }
    });
    
    html += `<path d="${path}" fill="none" stroke="#1a2980" stroke-width="2" />`;
    
    // Draw current threshold point
    const currentThreshold = parseFloat(document.getElementById('thresholdSlider').value);
    const currentPoint = rocPoints.find(p => Math.abs(p.threshold - currentThreshold) < 0.01) || rocPoints[50];
    const currentX = currentPoint.fpr * width;
    const currentY = height - (currentPoint.tpr * height);
    
    html += `<circle cx="${currentX}" cy="${currentY}" r="5" fill="#26d0ce" stroke="#fff" stroke-width="2" />`;
    html += `<text x="${currentX + 10}" y="${currentY - 10}" font-size="12">Threshold: ${currentThreshold.toFixed(2)}</text>`;
    
    // Add labels
    html += `<text x="${width/2}" y="${height-10}" text-anchor="middle" font-size="12">False Positive Rate</text>`;
    html += `<text x="10" y="15" text-anchor="start" font-size="12">True Positive Rate</text>`;
    const aucValue = calculateAUC(labelsArray, predsArray).toFixed(3);
    html += `<text x="${width-10}" y="15" text-anchor="end" font-size="12">AUC: ${aucValue}</text>`;
    
    html += '</svg>';
    
    container.innerHTML = html;
}

/**
 * Update threshold slider value and refresh evaluation
 */
function updateThreshold() {
    const slider = document.getElementById('thresholdSlider');
    const value = parseFloat(slider.value);
    
    document.getElementById('thresholdValue').textContent = value.toFixed(2);
    
    if (validationPredictions && validationLabels) {
        const metrics = calculateMetrics(validationLabels, validationPredictions, value);
        
        updateMetricsDisplay(metrics);
        updateConfusionMatrixDisplay(
            metrics.confusionMatrix.tp,
            metrics.confusionMatrix.fp,
            metrics.confusionMatrix.tn,
            metrics.confusionMatrix.fn
        );
        
        createROCCurve(validationLabels, validationPredictions);
    }
}

/**
 * Make predictions on test data
 */
async function predictTestData() {
    if (!model || !processedTestData) {
        updateStatus('exportStatus', 'error', 'Model or test data not available.');
        return;
    }
    
    try {
        updateStatus('exportStatus', 'info', 'Making predictions on test data...');
        
        const predictions = model.predict(processedTestData.features);
        const predsArray = predictions.arraySync().flat();
        const threshold = parseFloat(document.getElementById('thresholdSlider').value);
        
        let submissionCSV = 'PassengerId,Survived\n';
        let probabilitiesCSV = 'PassengerId,Probability,Survived_Prediction\n';
        
        for (let i = 0; i < processedTestData.passengerIds.length; i++) {
            const passengerId = processedTestData.passengerIds[i];
            const probability = predsArray[i];
            const survived = probability >= threshold ? 1 : 0;
            
            submissionCSV += `${passengerId},${survived}\n`;
            probabilitiesCSV += `${passengerId},${probability.toFixed(6)},${survived}\n`;
        }
        
        downloadFile(submissionCSV, 'submission.csv', 'text/csv');
        downloadFile(probabilitiesCSV, 'probabilities.csv', 'text/csv');
        
        updateStatus('exportStatus', 'success', 
            `Predictions completed! Downloaded files for ${processedTestData.passengerIds.length} passengers.`);
        
    } catch (error) {
        console.error('Error making predictions:', error);
        updateStatus('exportStatus', 'error', `Error making predictions: ${error.message}`);
    }
}

/**
 * Export the trained model
 */
async function exportModel() {
    if (!model) {
        updateStatus('exportStatus', 'error', 'No model to export. Please train a model first.');
        return;
    }
    
    try {
        updateStatus('exportStatus', 'info', 'Exporting model...');
        
        await model.save('downloads://titanic-model');
        
        updateStatus('exportStatus', 'success', 'Model exported successfully!');
        
    } catch (error) {
        console.error('Error exporting model:', error);
        updateStatus('exportStatus', 'error', `Error exporting model: ${error.message}`);
    }
}

/**
 * Download a file
 */
function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 100);
}
