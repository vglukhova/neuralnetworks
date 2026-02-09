/**
 * TensorFlow.js Titanic Binary Classifier
 * 
 * This application implements a shallow neural network for binary classification
 * on the Kaggle Titanic dataset. It runs entirely in the browser using TensorFlow.js.
 * 
 * Data Flow: CSV → Parsing → Preprocessing → Tensor Conversion → Model Training → Evaluation → Prediction → Export
 * 
 * Key Components:
 * 1. Data loading with PapaParse (handles quoted fields with commas)
 * 2. Data preprocessing and feature engineering
 * 3. Neural network model definition and training
 * 4. Model evaluation with interactive metrics
 * 5. Sigmoid gate feature importance visualization
 * 6. Prediction generation and export functionality
 */

// Global state variables to track application state
let appState = {
    rawTrainData: null,
    rawTestData: null,
    processedTrainData: null,
    processedTestData: null,
    featureNames: [],
    model: null,
    trainingHistory: null,
    validationData: null,
    testPredictions: null,
    evaluationMetrics: null,
    featureWeights: null,
    isDataLoaded: false,
    isPreprocessed: false,
    isModelCreated: false,
    isTrained: false,
    isEvaluated: false
};

// DOM elements
const domElements = {};

/**
 * Initialize the application by setting up DOM references and event listeners
 */
function initApp() {
    console.log("Initializing Titanic Classifier Application");
    
    // Initialize DOM element references
    const elementIds = [
        'loadDataBtn', 'preprocessBtn', 'createModelBtn', 'trainBtn', 'evaluateBtn',
        'predictBtn', 'exportModelBtn', 'exportPredictionsBtn', 'featureImportanceBtn',
        'dataStatus', 'preprocessStatus', 'modelStatus', 'trainStatus', 'metricsStatus',
        'predictStatus', 'exportStatus', 'dataPreview', 'dataStats', 'dataCharts',
        'preprocessedInfo', 'modelSummary', 'trainingCharts', 'rocChart',
        'featureImportanceChart', 'featureWeightsTable',
        'thresholdSlider', 'thresholdValue',
        'trueNeg', 'falsePos', 'falseNeg', 'truePos',
        'accuracyValue', 'precisionValue', 'recallValue', 'f1Value', 'aucValue',
        'trainFile', 'testFile', 'toggleFamilySize', 'toggleIsAlone'
    ];
    
    elementIds.forEach(id => {
        domElements[id] = document.getElementById(id);
    });
    
    // Set up event listeners
    domElements.loadDataBtn.addEventListener('click', loadData);
    domElements.preprocessBtn.addEventListener('click', preprocessData);
    domElements.createModelBtn.addEventListener('click', createModel);
    domElements.trainBtn.addEventListener('click', trainModel);
    domElements.evaluateBtn.addEventListener('click', evaluateModel);
    domElements.predictBtn.addEventListener('click', generatePredictions);
    domElements.exportModelBtn.addEventListener('click', exportModel);
    domElements.exportPredictionsBtn.addEventListener('click', exportPredictions);
    domElements.featureImportanceBtn.addEventListener('click', calculateFeatureImportance);
    
    // Set up threshold slider for interactive evaluation
    domElements.thresholdSlider.addEventListener('input', function() {
        const threshold = parseFloat(this.value);
        domElements.thresholdValue.textContent = threshold.toFixed(2);
        if (appState.evaluationMetrics) {
            updateMetricsDisplay(threshold);
        }
    });
    
    // Initial status message
    updateStatus('dataStatus', 'Ready to load Titanic dataset CSV files', 'info');
    
    console.log("Application initialization complete");
}

/**
 * Update status message in a specific DOM element
 * @param {string} elementId - ID of the status element
 * @param {string} message - Status message text
 * @param {string} type - Status type: 'info', 'success', 'warning', 'error'
 */
function updateStatus(elementId, message, type = 'info') {
    const element = domElements[elementId];
    if (!element) return;
    
    element.textContent = message;
    element.className = `status-message status-${type}`;
    
    // Log to console for debugging
    const logMethod = type === 'error' ? 'error' : 'info';
    console[logMethod](`${elementId}: ${message}`);
}

/**
 * Load CSV data files using PapaParse library
 * PapaParse handles quoted fields containing commas correctly
 */
async function loadData() {
    console.log("Starting data loading process");
    
    const trainFile = domElements.trainFile.files[0];
    const testFile = domElements.testFile.files[0];
    
    if (!trainFile || !testFile) {
        updateStatus('dataStatus', 'Please select both train.csv and test.csv files', 'error');
        return;
    }
    
    try {
        updateStatus('dataStatus', 'Loading CSV files with PapaParse...', 'info');
        
        // Load training data
        const trainData = await parseCSV(trainFile);
        console.log(`Training data loaded: ${trainData.length} rows, ${Object.keys(trainData[0]).length} columns`);
        
        // Load test data  
        const testData = await parseCSV(testFile);
        console.log(`Test data loaded: ${testData.length} rows, ${Object.keys(testData[0]).length} columns`);
        
        // Validate data structure
        const requiredColumns = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];
        if (trainFile.name.includes('train')) {
            requiredColumns.push('Survived');
        }
        
        const sampleRow = trainData[0];
        const missingColumns = requiredColumns.filter(col => !(col in sampleRow));
        
        if (missingColumns.length > 0) {
            updateStatus('dataStatus', `Missing required columns: ${missingColumns.join(', ')}`, 'error');
            return;
        }
        
        // Store raw data
        appState.rawTrainData = trainData;
        appState.rawTestData = testData;
        appState.isDataLoaded = true;
        
        // Update UI
        updateStatus('dataStatus', `Data loaded successfully! Train: ${trainData.length} rows, Test: ${testData.length} rows`, 'success');
        domElements.preprocessBtn.disabled = false;
        
        // Display data preview
        displayDataPreview(trainData);
        
        // Calculate and display data statistics
        displayDataStats(trainData);
        
        // Create data visualization charts
        createDataCharts(trainData);
        
    } catch (error) {
        console.error('Error loading data:', error);
        updateStatus('dataStatus', `Error loading data: ${error.message}`, 'error');
    }
}

/**
 * Parse CSV file using PapaParse library
 * @param {File} file - CSV file object
 * @returns {Promise<Array>} - Parsed data as array of objects
 */
function parseCSV(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,          // Use first row as column names
            dynamicTyping: true,   // Convert numeric values to numbers
            skipEmptyLines: true,  // Skip empty lines
            complete: (results) => {
                if (results.errors.length > 0) {
                    console.warn('CSV parsing warnings:', results.errors);
                }
                resolve(results.data);
            },
            error: (error) => {
                reject(new Error(`CSV parsing failed: ${error.message}`));
            }
        });
    });
}

/**
 * Display a preview of the loaded data
 * @param {Array} data - Data array to display
 */
function displayDataPreview(data) {
    const previewCount = Math.min(10, data.length);
    const previewData = data.slice(0, previewCount);
    
    let html = '<table><thead><tr>';
    
    // Create header row
    const columns = Object.keys(previewData[0]);
    columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // Create data rows
    previewData.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            const value = row[col];
            html += `<td>${value !== null && value !== undefined ? value : '<em>null</em>'}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    domElements.dataPreview.innerHTML = html;
}

/**
 * Calculate and display data statistics
 * @param {Array} data - Training data array
 */
function displayDataStats(data) {
    const totalRows = data.length;
    const columns = Object.keys(data[0]);
    
    let statsHTML = `<p><strong>Dataset Shape:</strong> ${totalRows} rows × ${columns.length} columns</p>`;
    
    // Calculate missing values percentage for key columns
    const keyColumns = ['Age', 'Fare', 'Embarked', 'Cabin'];
    statsHTML += '<p><strong>Missing Values:</strong></p><ul>';
    
    keyColumns.forEach(col => {
        if (columns.includes(col)) {
            const missingCount = data.filter(row => row[col] === null || row[col] === undefined || row[col] === '').length;
            const missingPercent = ((missingCount / totalRows) * 100).toFixed(1);
            statsHTML += `<li>${col}: ${missingCount} (${missingPercent}%)</li>`;
        }
    });
    
    statsHTML += '</ul>';
    
    // Survival statistics
    if ('Survived' in data[0]) {
        const survivedCount = data.filter(row => row.Survived === 1).length;
        const survivalRate = ((survivedCount / totalRows) * 100).toFixed(1);
        statsHTML += `<p><strong>Survival Rate:</strong> ${survivedCount}/${totalRows} (${survivalRate}%)</p>`;
    }
    
    domElements.dataStats.innerHTML = statsHTML;
}

/**
 * Create visualization charts for the loaded data
 * @param {Array} data - Training data array
 */
function createDataCharts(data) {
    if (!data || data.length === 0) return;
    
    // Prepare data for visualizations
    const survivalBySex = {};
    const survivalByClass = {};
    
    data.forEach(row => {
        // Survival by Sex
        if (row.Sex && row.Survived !== undefined) {
            const key = `${row.Sex}_${row.Survived}`;
            survivalBySex[key] = (survivalBySex[key] || 0) + 1;
        }
        
        // Survival by Pclass
        if (row.Pclass && row.Survived !== undefined) {
            const key = `Class ${row.Pclass}_${row.Survived}`;
            survivalByClass[key] = (survivalByClass[key] || 0) + 1;
        }
    });
    
    // Create visualization surface
    const surface = { name: 'Data Distribution', tab: 'Data Analysis' };
    
    // Convert data to tfjs-vis format
    const sexData = {
        values: [
            { x: 'Male - Died', y: survivalBySex['male_0'] || 0 },
            { x: 'Male - Survived', y: survivalBySex['male_1'] || 0 },
            { x: 'Female - Died', y: survivalBySex['female_0'] || 0 },
            { x: 'Female - Survived', y: survivalBySex['female_1'] || 0 }
        ]
    };
    
    const classData = {
        values: [
            { x: '1st Class - Died', y: survivalByClass['Class 1_0'] || 0 },
            { x: '1st Class - Survived', y: survivalByClass['Class 1_1'] || 0 },
            { x: '2nd Class - Died', y: survivalByClass['Class 2_0'] || 0 },
            { x: '2nd Class - Survived', y: survivalByClass['Class 2_1'] || 0 },
            { x: '3rd Class - Died', y: survivalByClass['Class 3_0'] || 0 },
            { x: '3rd Class - Survived', y: survivalByClass['Class 3_1'] || 0 }
        ]
    };
    
    // Render charts
    try {
        tfvis.render.barchart(domElements.dataCharts, sexData, {
            xLabel: 'Category',
            yLabel: 'Count',
            width: 400,
            height: 300
        });
        
        // Add class chart below sex chart
        setTimeout(() => {
            const classSurface = { name: 'Survival by Passenger Class', tab: 'Data Analysis' };
            tfvis.render.barchart(domElements.dataCharts, classData, {
                xLabel: 'Category',
                yLabel: 'Count',
                width: 400,
                height: 300
            });
        }, 100);
    } catch (error) {
        console.error('Error creating charts:', error);
        domElements.dataCharts.innerHTML = `<p>Chart rendering error: ${error.message}</p>`;
    }
}

/**
 * Preprocess the loaded data: handle missing values, encode categorical variables, normalize numerical features
 */
function preprocessData() {
    console.log("Starting data preprocessing");
    
    if (!appState.isDataLoaded || !appState.rawTrainData) {
        updateStatus('preprocessStatus', 'Please load data first', 'error');
        return;
    }
    
    try {
        updateStatus('preprocessStatus', 'Preprocessing data...', 'info');
        
        // Extract feature flags from UI
        const addFamilySize = domElements.toggleFamilySize.checked;
        const addIsAlone = domElements.toggleIsAlone.checked;
        
        // Preprocess training data
        const trainResult = preprocessDataset(appState.rawTrainData, true, addFamilySize, addIsAlone);
        appState.processedTrainData = trainResult;
        
        // Preprocess test data (using training stats for consistency)
        const testResult = preprocessDataset(appState.rawTestData, false, addFamilySize, addIsAlone, trainResult.stats);
        appState.processedTestData = testResult;
        
        // Store feature names for later use
        appState.featureNames = trainResult.featureNames;
        
        appState.isPreprocessed = true;
        
        // Update UI
        updateStatus('preprocessStatus', 
            `Preprocessing complete! ${trainResult.features.shape[1]} features created`, 
            'success');
        
        // Display preprocessing info
        displayPreprocessingInfo(trainResult);
        
        // Enable model creation button
        domElements.createModelBtn.disabled = false;
        
    } catch (error) {
        console.error('Error preprocessing data:', error);
        updateStatus('preprocessStatus', `Preprocessing error: ${error.message}`, 'error');
    }
}

/**
 * Preprocess a single dataset
 * @param {Array} data - Raw data array
 * @param {boolean} isTraining - Whether this is training data
 * @param {boolean} addFamilySize - Whether to add FamilySize feature
 * @param {boolean} addIsAlone - Whether to add IsAlone feature
 * @param {Object} trainingStats - Statistics from training data (for test data preprocessing)
 * @returns {Object} - Preprocessed data and metadata
 */
function preprocessDataset(data, isTraining, addFamilySize, addIsAlone, trainingStats = null) {
    console.log(`Preprocessing ${isTraining ? 'training' : 'test'} dataset with ${data.length} rows`);
    
    const stats = isTraining ? {} : trainingStats;
    
    // Calculate statistics from training data
    if (isTraining) {
        // Calculate median age
        const ages = data.map(row => row.Age).filter(age => age !== null && age !== undefined);
        stats.medianAge = ages.length > 0 ? 
            ages.sort((a, b) => a - b)[Math.floor(ages.length / 2)] : 30;
        
        // Calculate median fare
        const fares = data.map(row => row.Fare).filter(fare => fare !== null && fare !== undefined);
        stats.medianFare = fares.length > 0 ? 
            fares.sort((a, b) => a - b)[Math.floor(fares.length / 2)] : 32;
        
        // Calculate mean and std for standardization
        stats.ageMean = ages.reduce((sum, age) => sum + age, 0) / ages.length;
        stats.ageStd = Math.sqrt(ages.reduce((sum, age) => sum + Math.pow(age - stats.ageMean, 2), 0) / ages.length);
        
        stats.fareMean = fares.reduce((sum, fare) => sum + fare, 0) / fares.length;
        stats.fareStd = Math.sqrt(fares.reduce((sum, fare) => sum + Math.pow(fare - stats.fareMean, 2), 0) / fares.length);
        
        // Find mode for Embarked
        const embarkedCounts = {};
        data.forEach(row => {
            if (row.Embarked) {
                embarkedCounts[row.Embarked] = (embarkedCounts[row.Embarked] || 0) + 1;
            }
        });
        stats.embarkedMode = Object.keys(embarkedCounts).reduce((a, b) => 
            embarkedCounts[a] > embarkedCounts[b] ? a : b, 'S');
        
        console.log('Calculated preprocessing stats:', stats);
    }
    
    // Process each row
    const processedFeatures = [];
    const processedLabels = isTraining ? [] : null;
    const passengerIds = [];
    
    data.forEach(row => {
        const features = [];
        passengerIds.push(row.PassengerId);
        
        // Handle missing Age - impute with median
        let age = row.Age;
        if (age === null || age === undefined || isNaN(age)) {
            age = stats.medianAge;
        }
        // Standardize Age
        features.push((age - stats.ageMean) / stats.ageStd);
        
        // Handle missing Fare - impute with median
        let fare = row.Fare;
        if (fare === null || fare === undefined || isNaN(fare)) {
            fare = stats.medianF
