/**
 * Titanic Survival Classifier using TensorFlow.js
 * Runs entirely in the browser - no server required
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
    updateStatus('dataStatus', 'info', 'Ready to load data. Click "Load CSV Files" or "Load Sample Data" to begin.');
}

/**
 * Set up event listeners for all buttons and controls
 */
function setupEventListeners() {
    // Data loading
    document.getElementById('loadDataBtn').addEventListener('click', loadCSVFiles);
    document.getElementById('loadSampleBtn').addEventListener('click', loadSampleData);
    
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
}

/**
 * Update status message in the UI
 * @param {string} elementId - ID of the status element
 * @param {string} type - 'success', 'error', or 'info'
 * @param {string} message - Status message
 */
function updateStatus(elementId, type, message) {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.className = `status ${type}`;
}

/**
 * Parse CSV text, handling quoted fields with commas
 * @param {string} csvText - Raw CSV text
 * @returns {Array} Array of objects representing the CSV data
 */
function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim() !== '');
    if (lines.length === 0) {
        throw new Error('CSV file is empty');
    }
    
    // Parse headers
    const headers = parseCSVLine(lines[0]);
    
    // Parse data rows
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        
        // Skip rows with wrong number of values
        if (values.length !== headers.length) {
            console.warn(`Skipping row ${i}: expected ${headers.length} columns, got ${values.length}`);
            continue;
        }
        
        // Create object for this row
        const row = {};
        for (let j = 0; j < headers.length; j++) {
            let value = values[j];
            
            // Convert numeric values
            if (!isNaN(value) && value !== '') {
                value = parseFloat(value);
            }
            
            row[headers[j]] = value;
        }
        
        data.push(row);
    }
    
    return data;
}

/**
 * Parse a single CSV line, handling quoted fields with commas
 * @param {string} line - A single line from CSV
 * @returns {Array} Array of values
 */
function parseCSVLine(line) {
    const values = [];
    let currentValue = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1] || '';
        
        if (char === '"') {
            // Toggle quote mode
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            // End of current value
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
        
        // Load training data
        const trainFile = trainFileInput.files[0];
        const trainText = await trainFile.text();
        trainData = parseCSV(trainText);
        
        // Load test data if provided
        if (testFileInput.files[0]) {
            const testFile = testFileInput.files[0];
            const testText = await testFile.text();
            testData = parseCSV(testText);
        }
        
        updateStatus('dataStatus', 'success', 
            `Loaded ${trainData.length} training samples${testData ? ` and ${testData.length} test samples` : ''}.`);
        
        // Enable preprocessing button
        document.getElementById('preprocessBtn').disabled = false;
        
        // Show data preview
        showDataPreview();
        
        // Show survival distribution
        showSurvivalDistribution();
        
    } catch (error) {
        console.error('Error loading CSV files:', error);
        updateStatus('dataStatus', 'error', `Error loading CSV: ${error.message}`);
    }
}

/**
 * Load sample Titanic data (hardcoded subset for demo)
 */
function loadSampleData() {
    updateStatus('dataStatus', 'info', 'Loading sample Titanic data...');
    
    // Sample Titanic data (subset for demo)
    const sampleTrainData = [
        {PassengerId: 1, Survived: 0, Pclass: 3, Sex: 'male', Age: 22, SibSp: 1, Parch: 0, Fare: 7.25, Embarked: 'S'},
        {PassengerId: 2, Survived: 1, Pclass: 1, Sex: 'female', Age: 38, SibSp: 1, Parch: 0, Fare: 71.28, Embarked: 'C'},
        {PassengerId: 3, Survived: 1, Pclass: 3, Sex: 'female', Age: 26, SibSp: 0, Parch: 0, Fare: 7.92, Embarked: 'S'},
        {PassengerId: 4, Survived: 1, Pclass: 1, Sex: 'female', Age: 35, SibSp: 1, Parch: 0, Fare: 53.1, Embarked: 'S'},
        {PassengerId: 5, Survived: 0, Pclass: 3, Sex: 'male', Age: 35, SibSp: 0, Parch: 0, Fare: 8.05, Embarked: 'S'},
        {PassengerId: 6, Survived: 0, Pclass: 3, Sex: 'male', Age: null, SibSp: 0, Parch: 0, Fare: 8.46, Embarked: 'Q'},
        {PassengerId: 7, Survived: 0, Pclass: 1, Sex: 'male', Age: 54, SibSp: 0, Parch: 0, Fare: 51.86, Embarked: 'S'},
        {PassengerId: 8, Survived: 0, Pclass: 3, Sex: 'male', Age: 2, SibSp: 3, Parch: 1, Fare: 21.08, Embarked: 'S'},
        {PassengerId: 9, Survived: 1, Pclass: 3, Sex: 'female', Age: 27, SibSp: 0, Parch: 2, Fare: 11.13, Embarked: 'S'},
        {PassengerId: 10, Survived: 1, Pclass: 2, Sex: 'female', Age: 14, SibSp: 1, Parch: 0, Fare: 30.07, Embarked: 'C'}
    ];
    
    const sampleTestData = [
        {PassengerId: 11, Pclass: 3, Sex: 'male', Age: 4, SibSp: 1, Parch: 1, Fare: 16.7, Embarked: 'S'},
        {PassengerId: 12, Pclass: 1, Sex: 'female', Age: 58, SibSp: 0, Parch: 0, Fare: 26.55, Embarked: 'S'},
        {PassengerId: 13, Pclass: 3, Sex: 'male', Age: 20, SibSp: 0, Parch: 0, Fare: 8.05, Embarked: 'S'},
        {PassengerId: 14, Pclass: 3, Sex: 'male', Age: 39, SibSp: 1, Parch: 5, Fare: 31.28, Embarked: 'S'},
        {PassengerId: 15, Pclass: 3, Sex: 'female', Age: 14, SibSp: 0, Parch: 0, Fare: 7.85, Embarked: 'S'}
    ];
    
    trainData = sampleTrainData;
    testData = sampleTestData;
    
    updateStatus('dataStatus', 'success', 
        `Loaded ${trainData.length} sample training records and ${testData.length} test records.`);
    
    // Enable preprocessing button
    document.getElementById('preprocessBtn').disabled = false;
    
    // Show data preview
    showDataPreview();
    
    // Show survival distribution
    showSurvivalDistribution();
}

/**
 * Display a preview of the loaded data
 */
function showDataPreview() {
    const container = document.getElementById('dataPreview');
    
    if (!trainData || trainData.length === 0) {
        container.innerHTML = '<p>No data loaded.</p>';
        return;
    }
    
    // Show first 5 rows
    const previewRows = trainData.slice(0, 5);
    
    let html = '<table>';
    html += '<thead><tr>';
    
    // Headers
    Object.keys(previewRows[0]).forEach(col => {
        html += `<th>${col}</th>`;
    });
    
    html += '</tr></thead><tbody>';
    
    // Data rows
    previewRows.forEach(row => {
        html += '<tr>';
        Object.values(row).forEach(val => {
            html += `<td>${val === null || val === undefined ? '' : val}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    html += `<p>Showing 5 of ${trainData.length} rows. Dataset shape: ${trainData.length} rows × ${Object.keys(trainData[0]).length} columns</p>`;
    
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
            survivalBySex[sex].total++;
            if (row.Survived === 1) {
                survivalBySex[sex].survived++;
            }
        }
        
        if (row.Pclass && row.Survived !== undefined) {
            const pclass = row.Pclass.toString();
            survivalByClass[pclass].total++;
            if (row.Survived === 1) {
                survivalByClass[pclass].survived++;
            }
        }
    });
    
    // Calculate percentages
    const sexData = [];
    const classData = [];
    
    Object.keys(survivalBySex).forEach(sex => {
        const data = survivalBySex[sex];
        const survivalRate = data.total > 0 ? (data.survived / data.total * 100).toFixed(1) : 0;
        sexData.push({
            sex: sex.charAt(0).toUpperCase() + sex.slice(1),
            survivalRate: parseFloat(survivalRate),
            count: data.total
        });
    });
    
    Object.keys(survivalByClass).forEach(pclass => {
        const data = survivalByClass[pclass];
        const survivalRate = data.total > 0 ? (data.survived / data.total * 100).toFixed(1) : 0;
        classData.push({
            class: `Class ${pclass}`,
            survivalRate: parseFloat(survivalRate),
            count: data.total
        });
    });
    
    // Create HTML visualization
    let html = '<div style="display: flex; flex-wrap: wrap; gap: 20px;">';
    
    // Sex survival chart
    html += '<div style="flex: 1; min-width: 250px;">';
    html += '<h4>Survival by Sex</h4>';
    sexData.forEach(item => {
        html += `<p style="margin: 8px 0;"><strong>${item.sex}:</strong> ${item.survivalRate}% survived (${item.count} passengers)</p>`;
        html += `<div style="height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 5px 0;">`;
        html += `<div style="height: 100%; width: ${item.survivalRate}%; background: linear-gradient(to right, #26d0ce, #1a2980);"></div>`;
        html += '</div>';
    });
    html += '</div>';
    
    // Class survival chart
    html += '<div style="flex: 1; min-width: 250px;">';
    html += '<h4>Survival by Passenger Class</h4>';
    classData.forEach(item => {
        html += `<p style="margin: 8px 0;"><strong>${item.class}:</strong> ${item.survivalRate}% survived (${item.count} passengers)</p>`;
        html += `<div style="height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 5px 0;">`;
        html += `<div style="height: 100%; width: ${item.survivalRate}%; background: linear-gradient(to right, #26d0ce, #1a2980);"></div>`;
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
        }
        
        updateStatus('preprocessStatus', 'success', 
            `Preprocessed ${processedTrainData.features.shape[0]} training samples with ${processedTrainData.features.shape[1]} features.`);
        
        // Enable model creation button
        document.getElementById('createModelBtn').disabled = false;
        
        // Log feature info
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
 * @param {Array} data - The dataset to process
 * @param {boolean} isTraining - Whether this is training data (has labels)
 * @returns {Object} Processed features and labels (if training)
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
        const ages = data.map(row => row.Age).filter(age => age !== null && age !== undefined);
        const fares = data.map(row => row.Fare).filter(fare => fare !== null && fare !== undefined);
        const embarked = data.map(row => row.Embarked).filter(e => e !== null && e !== undefined);
        
        if (ages.length > 0) {
            ages.sort((a, b) => a - b);
            ageMedian = ages[Math.floor(ages.length / 2)];
        }
        
        if (fares.length > 0) {
            fares.sort((a, b) => a - b);
            fareMedian = fares[Math.floor(fares.length / 2)];
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
            if (value === null || value === undefined || value === '') {
                if (col === 'Age') value = ageMedian;
                else if (col === 'Fare') value = fareMedian;
                else value = 0;
            }
            
            featureRow.push(value);
        });
        
        // Handle categorical features with one-hot encoding placeholders
        // We'll do actual one-hot encoding after collecting all data
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
        if (isTraining && row[TARGET_COL] !== undefined) {
            labels.push(row[TARGET_COL]);
        }
    });
    
    // Convert to tensors
    let featuresTensor = tf.tensor2d(features.map(row => {
        // Convert categorical values to numerical indices for one-hot encoding
        return row.map((val, idx) => {
            // First NUMERICAL_COLS.length values are already numeric
            if (idx < NUMERICAL_COLS.length) {
                return val;
            }
            
            // Categorical values need to be converted
            // For simplicity in this demo, we'll use simple encoding
            // In a full implementation, you would use proper one-hot encoding
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
 * Create the neural network model
 */
function createModel() {
    if (!processedTrainData) {
        updateStatus('modelStatus', 'error', 'No processed data available. Please preprocess data first.');
        return;
    }
    
    try {
        updateStatus('modelStatus', 'info', 'Creating model...');
        
        // Get input shape
        const inputShape = processedTrainData.features.shape[1];
        
        // Create sequential model
        model = tf.sequential();
        
        // Hidden layer with 16 neurons and ReLU activation
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [inputShape]
        }));
        
        // Output layer with 1 neuron and sigmoid activation for binary classification
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
        
        // Compile the model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Print model summary
        console.log('Model summary:');
        model.summary();
        
        updateStatus('modelStatus', 'success', 
            `Model created with ${inputShape} input features and 1 output.`);
        
        // Enable training button
        document.getElementById('trainBtn').disabled = false;
        
    } catch (error) {
        console.error('Error creating model:', error);
        updateStatus('modelStatus', 'error', `Error creating model: ${error.message}`);
    }
}

/**
 * Train the model
 */
async function trainModel() {
    if (!model || !processedTrainData) {
        updateStatus('trainingStatus', 'error', 'Model or data not available. Please create model first.');
        return;
    }
    
    try {
        updateStatus('trainingStatus', 'info', 'Training model...');
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
        
        // Create tfjs-vis callback for live training plots
        const container = document.getElementById('trainingHistory');
        const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
        
        // Train the model
        const history = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationSplit: 0.2,
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
                    
                    // Update training status
                    updateStatus('trainingStatus', 'info', 
                        `Epoch ${epoch + 1}/50 - Loss: ${logs.loss.toFixed(4)}, Acc: ${logs.acc.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}, Val Acc: ${logs.val_acc.toFixed(4)}`);
                    
                    // Create simple training history visualization
                    if (container) {
                        let html = '<div style="display: flex; flex-wrap: wrap; gap: 20px;">';
                        
                        // Loss chart
                        html += '<div style="flex: 1; min-width: 300px;">';
                        html += '<h4>Training & Validation Loss</h4>';
                        html += '<div style="height: 200px; position: relative; border: 1px solid #ddd; border-radius: 5px; padding: 10px;">';
                        
                        const maxLoss = Math.max(...trainingHistory.map(h => Math.max(h.loss, h.val_loss)));
                        trainingHistory.forEach((h, idx) => {
                            const lossHeight = (h.loss / maxLoss) * 180;
                            const valLossHeight = (h.val_loss / maxLoss) * 180;
                            
                            html += `<div style="position: absolute; bottom: 0; left: ${idx * 10}px; width: 8px; height: ${lossHeight}px; background: #1a2980;"></div>`;
                            html += `<div style="position: absolute; bottom: 0; left: ${idx * 10 + 4}px; width: 8px; height: ${valLossHeight}px; background: #26d0ce;"></div>`;
                        });
                        
                        html += '</div>';
                        html += '<div style="display: flex; gap: 10px; margin-top: 10px;">';
                        html += '<div><div style="width: 12px; height: 12px; background: #1a2980; display: inline-block; margin-right: 5px;"></div> Training Loss</div>';
                        html += '<div><div style="width: 12px; height: 12px; background: #26d0ce; display: inline-block; margin-right: 5px;"></div> Validation Loss</div>';
                        html += '</div>';
                        html += '</div>';
                        
                        // Accuracy chart
                        html += '<div style="flex: 1; min-width: 300px;">';
                        html += '<h4>Training & Validation Accuracy</h4>';
                        html += '<div style="height: 200px; position: relative; border: 1px solid #ddd; border-radius: 5px; padding: 10px;">';
                        
                        trainingHistory.forEach((h, idx) => {
                            const accHeight = h.acc * 180;
                            const valAccHeight = h.val_acc * 180;
                            
                            html += `<div style="position: absolute; bottom: 0; left: ${idx * 10}px; width: 8px; height: ${accHeight}px; background: #1a2980;"></div>`;
                            html += `<div style="position: absolute; bottom: 0; left: ${idx * 10 + 4}px; width: 8px; height: ${valAccHeight}px; background: #26d0ce;"></div>`;
                        });
                        
                        html += '</div>';
                        html += '<div style="display: flex; gap: 10px; margin-top: 10px;">';
                        html += '<div><div style="width: 12px; height: 12px; background: #1a2980; display: inline-block; margin-right: 5px;"></div> Training Accuracy</div>';
                        html += '<div><div style="width: 12px; height: 12px; background: #26d0ce; display: inline-block; margin-right: 5px;"></div> Validation Accuracy</div>';
                        html += '</div>';
                        html += '</div>';
                        
                        html += '</div>';
                        container.innerHTML = html;
                    }
                },
                onTrainEnd: () => {
                    updateStatus('trainingStatus', 'success', 'Training completed successfully!');
                    isTraining = false;
                    
                    // Enable evaluation button
                    document.getElementById('evaluateBtn').disabled = false;
                    document.getElementById('thresholdSlider').disabled = false;
                    
                    // Enable prediction button if test data is available
                    if (processedTestData) {
                        document.getElementById('predictBtn').disabled = false;
                    }
                    
                    // Enable export button
                    document.getElementById('exportBtn').disabled = false;
                    
                    // Re-enable train button, disable stop button
                    document.getElementById('trainBtn').disabled = false;
                    document.getElementById('stopTrainBtn').disabled = true;
                    
                    // Calculate feature importance
                    calculateFeatureImportance();
                }
            }
        });
        
    } catch (error) {
        console.error('Error training model:', error);
        updateStatus('trainingStatus', 'error', `Error training model: ${error.message}`);
        isTraining = false;
        
        // Re-enable train button, disable stop button
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('stopTrainBtn').disabled = true;
    }
}

/**
 * Stop training early
 */
function stopTraining() {
    if (isTraining) {
        isTraining = false;
        updateStatus('trainingStatus', 'info', 'Training stopped by user.');
        
        // Re-enable train button, disable stop button
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('stopTrainBtn').disabled = true;
    }
}

/**
 * Calculate and display feature importance
 */
function calculateFeatureImportance() {
    if (!model) return;
    
    try {
        // Get the weights from the first dense layer
        const layer = model.layers[0];
        const weights = layer.getWeights()[0]; // Get the kernel weights
        
        // Calculate absolute mean weight for each feature
        const importance = weights.arraySync().map(neuronWeights => {
            return neuronWeights.map(w => Math.abs(w));
        });
        
        // Average across neurons to get feature importance
        const featureCount = importance[0].length;
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
        
        const avgImportance = new Array(featureCount).fill(0);
        
        importance.forEach(neuronWeights => {
            neuronWeights.forEach((weight, idx) => {
                avgImportance[idx] += weight;
            });
        });
        
        avgImportance.forEach((sum, idx) => {
            avgImportance[idx] = sum / importance.length;
        });
        
        // Apply sigmoid activation to importance scores
        const sigmoidImportance = avgImportance.map(score => {
            return 1 / (1 + Math.exp(-score));
        });
        
        // Normalize to percentages
        const total = sigmoidImportance.reduce((sum, val) => sum + val, 0);
        const normalizedImportance = sigmoidImportance.map(score => 
            total > 0 ? (score / total) * 100 : 0
        );
        
        // Create pairs of feature names and importance scores
        const featureImportancePairs = featureNames.map((name, idx) => ({
            name,
            importance: normalizedImportance[idx]
        }));
        
        // Sort by importance (descending)
        featureImportancePairs.sort((a, b) => b.importance - a.importance);
        
        // Display feature importance
        const container = document.getElementById('featureImportance');
        let html = '';
        
        featureImportancePairs.forEach(pair => {
            html += '<div class="feature-bar">';
            html += `<div class="feature-name">${pair.name}</div>`;
            html += `<div class="feature-bar-value" style="width: ${pair.importance * 3}px;">${pair.importance.toFixed(1)}%</div>`;
            html += '</div>';
        });
        
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error calculating feature importance:', error);
    }
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
        
        // Update metrics display
        document.getElementById('accuracyValue').textContent = metrics.accuracy.toFixed(3);
        document.getElementById('precisionValue').textContent = metrics.precision.toFixed(3);
        document.getElementById('recallValue').textContent = metrics.recall.toFixed(3);
        document.getElementById('f1Value').textContent = metrics.f1.toFixed(3);
        document.getElementById('aucValue').textContent = metrics.auc.toFixed(3);
        
        // Display evaluation table
        const tableContainer = document.getElementById('evaluationTable');
        let html = '<table>';
        html += '<thead><tr><th>Metric</th><th>Value</th><th>Description</th></tr></thead>';
        html += '<tbody>';
        html += `<tr><td>Accuracy</td><td>${metrics.accuracy.toFixed(4)}</td><td>Overall correctness</td></tr>`;
        html += `<tr><td>Precision</td><td>${metrics.precision.toFixed(4)}</td><td>True positives / (True positives + False positives)</td></tr>`;
        html += `<tr><td>Recall</td><td>${metrics.recall.toFixed(4)}</td><td>True positives / (True positives + False negatives)</td></tr>`;
        html += `<tr><td>F1 Score</td><td>${metrics.f1.toFixed(4)}</td><td>Harmonic mean of precision and recall</td></tr>`;
        html += `<tr><td>AUC</td><td>${metrics.auc.toFixed(4)}</td><td>Area under ROC curve</td></tr>`;
        html += '</tbody></table>';
        
        // Add confusion matrix
        html += '<h3 style="margin-top: 20px;">Confusion Matrix</h3>';
        html += '<table>';
        html += '<thead><tr><th></th><th>Predicted Negative</th><th>Predicted Positive</th></tr></thead>';
        html += '<tbody>';
        html += `<tr><td><strong>Actual Negative</strong></td><td>${metrics.confusionMatrix.tn}</td><td>${metrics.confusionMatrix.fp}</td></tr>`;
        html += `<tr><td><strong>Actual Positive</strong></td><td>${metrics.confusionMatrix.fn}</td><td>${metrics.confusionMatrix.tp}</td></tr>`;
        html += '</tbody></table>';
        
        tableContainer.innerHTML = html;
        
        // Create ROC curve visualization
        createROCCurve(validationLabels, predictions);
        
        updateStatus('evaluationStatus', 'success', 'Evaluation completed successfully!');
        
    } catch (error) {
        console.error('Error evaluating model:', error);
        updateStatus('evaluationStatus', 'error', `Error evaluating model: ${error.message}`);
    }
}

/**
 * Calculate evaluation metrics
 * @param {tf.Tensor} labels - True labels
 * @param {tf.Tensor} predictions - Predicted probabilities
 * @param {number} threshold - Classification threshold
 * @returns {Object} Evaluation metrics
 */
function calculateMetrics(labels, predictions, threshold) {
    // Convert tensors to arrays
    const labelsArray = labels.arraySync().flat();
    const predsArray = predictions.arraySync().flat();
    
    // Calculate confusion matrix
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < labelsArray.length; i++) {
        const actual = labelsArray[i];
        const predicted = predsArray[i] >= threshold ? 1 : 0;
        
        if (actual === 1 && predicted === 1) tp++;
        else if (actual === 0 && predicted === 1) fp++;
        else if (actual === 0 && predicted === 0) tn++;
        else if (actual === 1 && predicted === 0) fn++;
    }
    
    // Calculate metrics
    const accuracy = (tp + tn) / (tp + fp + tn + fn);
    const precision = tp > 0 ? tp / (tp + fp) : 0;
    const recall = tp > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    
    // Calculate AUC (simplified)
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
 * Calculate Area Under Curve (AUC) using trapezoidal rule
 * @param {Array} labels - True labels
 * @param {Array} scores - Predicted scores
 * @returns {number} AUC value
 */
function calculateAUC(labels, scores) {
    // Create pairs of scores and labels
    const pairs = scores.map((score, idx) => ({ score, label: labels[idx] }));
    
    // Sort by score descending
    pairs.sort((a, b) => b.score - a.score);
    
    // Calculate true positive rate and false positive rate at different thresholds
    const totalPos = labels.filter(label => label === 1).length;
    const totalNeg = labels.filter(label => label === 0).length;
    
    let tpr = [0]; // True Positive Rate
    let fpr = [0]; // False Positive Rate
    
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
    
    // Add final point
    tpr.push(1);
    fpr.push(1);
    
    // Calculate AUC using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < tpr.length; i++) {
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2;
    }
    
    return auc;
}

/**
 * Create ROC curve visualization
 * @param {tf.Tensor} labels - True labels
 * @param {tf.Tensor} predictions - Predicted probabilities
 */
function createROCCurve(labels, predictions) {
    const labelsArray = labels.arraySync().flat();
    const predsArray = predictions.arraySync().flat();
    
    // Calculate ROC curve points
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
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        rocPoints.push({ fpr, tpr, threshold });
    });
    
    // Create visualization
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
    
    // Add labels
    html += `<text x="${width/2}" y="${height-10}" text-anchor="middle" font-size="12">False Positive Rate</text>`;
    html += `<text x="10" y="15" text-anchor="start" font-size="12">True Positive Rate</text>`;
    html += `<text x="${width-10}" y="15" text-anchor="end" font-size="12">AUC: ${calculateAUC(labelsArray, predsArray).toFixed(3)}</text>`;
    
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
    
    // If validation predictions exist, update metrics
    if (validationPredictions && validationLabels) {
        const metrics = calculateMetrics(validationLabels, validationPredictions, value);
        
        // Update metrics display
        document.getElementById('accuracyValue').textContent = metrics.accuracy.toFixed(3);
        document.getElementById('precisionValue').textContent = metrics.precision.toFixed(3);
        document.getElementById('recallValue').textContent = metrics.recall.toFixed(3);
        document.getElementById('f1Value').textContent = metrics.f1.toFixed(3);
        
        // Update confusion matrix in evaluation table
        const tableContainer = document.getElementById('evaluationTable');
        if (tableContainer.innerHTML) {
            // Update confusion matrix part
            let html = tableContainer.innerHTML;
            const confusionMatrixRegex = /<td><strong>Actual Negative<\/strong><\/td><td>(\d+)<\/td><td>(\d+)<\/td>[\s\S]*?<td><strong>Actual Positive<\/strong><\/td><td>(\d+)<\/td><td>(\d+)<\/td>/;
            html = html.replace(confusionMatrixRegex, 
                `<td><strong>Actual Negative</strong></td><td>${metrics.confusionMatrix.tn}</td><td>${metrics.confusionMatrix.fp}</td></tr><tr><td><strong>Actual Positive</strong></td><td>${metrics.confusionMatrix.fn}</td><td>${metrics.confusionMatrix.tp}</td>`);
            
            tableContainer.innerHTML = html;
        }
        
        // Update ROC curve
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
        
        // Make predictions
        const predictions = model.predict(processedTestData.features);
        const predsArray = predictions.arraySync().flat();
        
        // Get threshold
        const threshold = parseFloat(document.getElementById('thresholdSlider').value);
        
        // Create submission file
        let submissionCSV = 'PassengerId,Survived\n';
        let probabilitiesCSV = 'PassengerId,Probability\n';
        
        for (let i = 0; i < processedTestData.passengerIds.length; i++) {
            const passengerId = processedTestData.passengerIds[i];
            const probability = predsArray[i];
            const survived = probability >= threshold ? 1 : 0;
            
            submissionCSV += `${passengerId},${survived}\n`;
            probabilitiesCSV += `${passengerId},${probability.toFixed(6)}\n`;
        }
        
        // Download submission file
        downloadFile(submissionCSV, 'submission.csv', 'text/csv');
        
        // Download probabilities file
        downloadFile(probabilitiesCSV, 'probabilities.csv', 'text/csv');
        
        updateStatus('exportStatus', 'success', 
            `Predictions completed! Downloaded submission.csv and probabilities.csv for ${processedTestData.passengerIds.length} passengers.`);
        
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
        
        // Save the model
        await model.save('downloads://titanic-tfjs-model');
        
        updateStatus('exportStatus', 'success', 'Model exported successfully!');
        
    } catch (error) {
        console.error('Error exporting model:', error);
        updateStatus('exportStatus', 'error', `Error exporting model: ${error.message}`);
    }
}

/**
 * Download a file
 * @param {string} content - File content
 * @param {string} filename - File name
 * @param {string} type - MIME type
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
