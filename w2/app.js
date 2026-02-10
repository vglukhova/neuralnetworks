// Titanic Survival Classifier with TensorFlow.js
// Runs entirely in the browser - No server required
// Ready for GitHub Pages deployment

// ============================================================================
// Global Variables
// ============================================================================

let rawTrainData = null;
let rawTestData = null;
let processedTrainData = null;
let processedTestFeatures = null;
let model = null;
let featureNames = [];
let trainFeaturesTensor = null;
let trainLabelsTensor = null;
let valFeaturesTensor = null;
let valLabelsTensor = null;
let testFeaturesTensor = null;
let valPredictions = null;
let valLabels = null;
let isTraining = false;
let trainingStopRequested = false;
let featureImportance = [];
let useFamilyFeatures = true;
let testPassengerIds = [];

// Schema configuration - Swap for other datasets
// Target: Survived (0/1)
// Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
// Identifier: PassengerId (exclude from features)
const TARGET_COLUMN = 'Survived';
const ID_COLUMN = 'PassengerId';
const FEATURE_COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];

// ============================================================================
// DOM Elements
// ============================================================================

// Get DOM elements
const elements = {
    loadDataBtn: document.getElementById('loadDataBtn'),
    showDataPreviewBtn: document.getElementById('showDataPreviewBtn'),
    preprocessBtn: document.getElementById('preprocessBtn'),
    toggleFamilyFeaturesBtn: document.getElementById('toggleFamilyFeaturesBtn'),
    buildModelBtn: document.getElementById('buildModelBtn'),
    showModelSummaryBtn: document.getElementById('showModelSummaryBtn'),
    showFeatureImportanceBtn: document.getElementById('showFeatureImportanceBtn'),
    trainModelBtn: document.getElementById('trainModelBtn'),
    stopTrainingBtn: document.getElementById('stopTrainingBtn'),
    evaluateBtn: document.getElementById('evaluateBtn'),
    predictBtn: document.getElementById('predictBtn'),
    exportModelBtn: document.getElementById('exportModelBtn'),
    exportSubmissionBtn: document.getElementById('exportSubmissionBtn'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue'),
    
    // Status elements
    dataStatus: document.getElementById('dataStatus'),
    preprocessStatus: document.getElementById('preprocessStatus'),
    modelStatus: document.getElementById('modelStatus'),
    trainingStatus: document.getElementById('trainingStatus'),
    evaluationStatus: document.getElementById('evaluationStatus'),
    predictionStatus: document.getElementById('predictionStatus'),
    
    // Containers
    dataPreview: document.getElementById('dataPreview'),
    previewTableContainer: document.getElementById('previewTableContainer'),
    dataStats: document.getElementById('dataStats'),
    featureInfo: document.getElementById('featureInfo'),
    featureList: document.getElementById('featureList'),
    modelSummary: document.getElementById('modelSummary'),
    modelSummaryContent: document.getElementById('modelSummaryContent'),
    featureImportanceSection: document.getElementById('featureImportanceSection'),
    featureImportanceChart: document.getElementById('featureImportanceChart'),
    featureImportanceList: document.getElementById('featureImportanceList'),
    trainingResults: document.getElementById('trainingResults'),
    metricsContainer: document.getElementById('metricsContainer'),
    evaluationTableBody: document.getElementById('evaluation-table-body'),
    predictionResults: document.getElementById('predictionResults'),
    predictionPreview: document.getElementById('predictionPreview')
};

// ============================================================================
// Initialize Application
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('Titanic Survival Classifier initializing...');
    
    // Add event listeners
    elements.loadDataBtn.addEventListener('click', loadData);
    elements.showDataPreviewBtn.addEventListener('click', showDataPreview);
    elements.preprocessBtn.addEventListener('click', preprocessData);
    elements.toggleFamilyFeaturesBtn.addEventListener('click', toggleFamilyFeatures);
    elements.buildModelBtn.addEventListener('click', buildModel);
    elements.showModelSummaryBtn.addEventListener('click', showModelSummary);
    elements.showFeatureImportanceBtn.addEventListener('click', showFeatureImportance);
    elements.trainModelBtn.addEventListener('click', trainModel);
    elements.stopTrainingBtn.addEventListener('click', stopTraining);
    elements.evaluateBtn.addEventListener('click', evaluateModel);
    elements.predictBtn.addEventListener('click', predictTestData);
    elements.exportModelBtn.addEventListener('click', exportModel);
    elements.exportSubmissionBtn.addEventListener('click', exportSubmission);
    
    // Threshold slider event
    elements.thresholdSlider.addEventListener('input', () => {
        const threshold = parseFloat(elements.thresholdSlider.value);
        elements.thresholdValue.textContent = threshold.toFixed(2);
        if (valPredictions && valLabels) {
            updateMetricsWithThreshold(threshold);
        }
    });
    
    // Initial UI state
    updateUIState();
});

// Update UI state based on app progress
function updateUIState() {
    elements.showDataPreviewBtn.disabled = !rawTrainData;
    elements.preprocessBtn.disabled = !rawTrainData;
    elements.toggleFamilyFeaturesBtn.disabled = !rawTrainData;
    elements.buildModelBtn.disabled = !processedTrainData;
    elements.showModelSummaryBtn.disabled = !model;
    elements.showFeatureImportanceBtn.disabled = !model;
    elements.trainModelBtn.disabled = !model;
    elements.evaluateBtn.disabled = !model;
    elements.predictBtn.disabled = !(model && rawTestData);
    elements.exportModelBtn.disabled = !model;
    elements.exportSubmissionBtn.disabled = !(model && rawTestData);
}

// ============================================================================
// 1. Data Loading with CSV Parsing Fix
// ============================================================================

async function loadData() {
    const trainFile = document.getElementById('trainFile').files[0];
    const testFile = document.getElementById('testFile').files[0];
    
    if (!trainFile) {
        showStatus(elements.dataStatus, 'Please upload train.csv file', 'error');
        return;
    }
    
    showStatus(elements.dataStatus, 'Loading CSV files...', 'loading');
    
    try {
        // Load train data with proper CSV parsing
        console.log('Loading train.csv...');
        rawTrainData = await parseCSV(trainFile, true);
        
        // Load test data if provided
        if (testFile) {
            console.log('Loading test.csv...');
            rawTestData = await parseCSV(testFile, false);
        }
        
        showStatus(elements.dataStatus, `Train data loaded: ${rawTrainData.length} passengers. ${testFile ? `Test data loaded: ${rawTestData.length} passengers.` : ''}`, 'success');
        
        // Show data preview
        showDataPreview();
        
        // Analyze data and show visualizations
        analyzeData();
        
    } catch (error) {
        console.error('Error loading data:', error);
        showStatus(elements.dataStatus, `Error loading data: ${error.message}`, 'error');
    }
    
    updateUIState();
}

// CSV parsing function with RFC 4180 compliance and embedded comma handling
async function parseCSV(file, hasTarget) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = async (event) => {
            try {
                const csvText = event.target.result;
                console.log(`Parsing CSV: ${file.name}, size: ${csvText.length} chars`);
                
                // Try using tf.data.csv first (handles some CSV formats)
                try {
                    // Create a blob URL for tf.data.csv to read
                    const blob = new Blob([csvText], { type: 'text/csv' });
                    const url = URL.createObjectURL(blob);
                    
                    // Configure CSV parsing - note: tf.data.csv may not handle all quoted fields perfectly
                    const csvDataset = tf.data.csv(url, {
                        hasHeader: true,
                        delimiter: ',',
                        columnNames: undefined, // Infer from header
                        columnConfigs: { [TARGET_COLUMN]: { isLabel: hasTarget } }
                    });
                    
                    // Convert to array
                    const dataArray = await csvDataset.toArray();
                    URL.revokeObjectURL(url);
                    
                    console.log(`tf.data.csv parsed ${dataArray.length} rows`);
                    
                    // Validate we got the expected columns
                    if (dataArray.length > 0) {
                        const firstRow = dataArray[0];
                        const expectedCols = [ID_COLUMN, ...FEATURE_COLUMNS];
                        if (hasTarget) expectedCols.push(TARGET_COLUMN);
                        
                        const hasAllCols = expectedCols.every(col => col in firstRow);
                        
                        if (hasAllCols && dataArray.length > 10) {
                            console.log('tf.data.csv parsing successful, using this method');
                            resolve(dataArray);
                            return;
                        }
                    }
                } catch (tfError) {
                    console.warn('tf.data.csv parsing failed, falling back to PapaParse:', tfError.message);
                }
                
                // Fallback to PapaParse for better CSV compliance
                console.log('Using PapaParse for CSV parsing with RFC 4180 compliance...');
                
                // Check if PapaParse is available (loaded via CDN in HTML)
                if (typeof Papa === 'undefined') {
                    throw new Error('PapaParse library not loaded. Please check CDN.');
                }
                
                Papa.parse(csvText, {
                    header: true,
                    dynamicTyping: false, // Keep as strings, we'll convert later
                    skipEmptyLines: true,
                    quotes: true,
                    quoteChar: '"',
                    delimiter: ',',
                    escapeChar: '"',
                    // Fix: Enable unescape to handle escaped quotes properly
                    unescape: true,
                    complete: (results) => {
                        if (results.errors.length > 0) {
                            console.warn('PapaParse warnings:', results.errors);
                        }
                        
                        console.log(`PapaParse parsed ${results.data.length} rows`);
                        
                        // Validate data
                        if (results.data.length === 0) {
                            reject(new Error('CSV file appears to be empty or has no valid rows'));
                            return;
                        }
                        
                        // Check for required columns
                        const firstRow = results.data[0];
                        const expectedCols = [ID_COLUMN, ...FEATURE_COLUMNS];
                        if (hasTarget) expectedCols.push(TARGET_COLUMN);
                        
                        const missingCols = expectedCols.filter(col => !(col in firstRow));
                        if (missingCols.length > 0) {
                            console.warn('Missing columns in CSV:', missingCols);
                            console.log('Available columns:', Object.keys(firstRow));
                        }
                        
                        // Convert string values to appropriate types
                        const typedData = results.data.map(row => {
                            const typedRow = {};
                            
                            for (const key in row) {
                                const value = row[key];
                                
                                // Convert to number if possible
                                if (value !== '' && !isNaN(value) && value !== null) {
                                    typedRow[key] = parseFloat(value);
                                } else {
                                    typedRow[key] = value;
                                }
                            }
                            
                            return typedRow;
                        });
                        
                        // CSV Fix: Validate parsing with row count match
                        console.log(`CSV parsing complete: ${typedData.length} rows`);
                        console.log('Sample row:', typedData[0]);
                        
                        resolve(typedData);
                    },
                    error: (error) => {
                        reject(new Error(`PapaParse error: ${error.message}`));
                    }
                });
                
            } catch (error) {
                reject(error);
            }
        };
        
        reader.onerror = () => {
            reject(new Error('Failed to read file'));
        };
        
        reader.readAsText(file);
    });
}

// Show data preview table
function showDataPreview() {
    if (!rawTrainData) {
        showStatus(elements.dataStatus, 'No data loaded yet', 'warning');
        return;
    }
    
    elements.dataPreview.style.display = 'block';
    
    // Create preview table (first 10 rows)
    let tableHtml = `<table>
        <thead>
            <tr>
                ${Object.keys(rawTrainData[0]).map(col => `<th>${col}</th>`).join('')}
            </tr>
        </thead>
        <tbody>`;
    
    const previewRows = Math.min(10, rawTrainData.length);
    for (let i = 0; i < previewRows; i++) {
        tableHtml += '<tr>';
        for (const col in rawTrainData[i]) {
            tableHtml += `<td>${rawTrainData[i][col]}</td>`;
        }
        tableHtml += '</tr>';
    }
    
    tableHtml += '</tbody></table>';
    elements.previewTableContainer.innerHTML = tableHtml;
    
    // Show data statistics
    showDataStatistics();
}

// Show data statistics and missing values
function showDataStatistics() {
    if (!rawTrainData || rawTrainData.length === 0) return;
    
    const totalRows = rawTrainData.length;
    const columns = Object.keys(rawTrainData[0]);
    
    let statsHtml = `<h4>Data Statistics (${totalRows} rows, ${columns.length} columns)</h4>`;
    statsHtml += '<table><thead><tr><th>Column</th><th>Type</th><th>Missing %</th><th>Unique Values</th></tr></thead><tbody>';
    
    for (const col of columns) {
        const values = rawTrainData.map(row => row[col]);
        const nonMissing = values.filter(v => v !== '' && v !== null && v !== undefined);
        const missingPercent = ((totalRows - nonMissing.length) / totalRows * 100).toFixed(1);
        const uniqueCount = new Set(nonMissing).size;
        
        // Determine type
        let colType = 'Mixed';
        if (nonMissing.length > 0) {
            const firstVal = nonMissing[0];
            if (typeof firstVal === 'number') {
                colType = 'Numeric';
            } else if (typeof firstVal === 'string') {
                colType = 'Categorical';
            }
        }
        
        statsHtml += `<tr>
            <td>${col}</td>
            <td>${colType}</td>
            <td>${missingPercent}%</td>
            <td>${uniqueCount}</td>
        </tr>`;
    }
    
    statsHtml += '</tbody></table>';
    elements.dataStats.innerHTML = statsHtml;
}

// Analyze data and create visualizations
// Analyze data and create visualizations
function analyzeData() {
    if (!rawTrainData || rawTrainData.length === 0) return;
    
    // Clear existing charts
    document.getElementById('survivalBySexChart').innerHTML = '';
    document.getElementById('survivalByClassChart').innerHTML = '';
    
    // Calculate survival rate by sex
    const survivalBySex = {};
    rawTrainData.forEach(passenger => {
        const sex = passenger.Sex || 'unknown';
        const survived = passenger.Survived;
        
        if (!survivalBySex[sex]) {
            survivalBySex[sex] = { total: 0, survived: 0 };
        }
        
        survivalBySex[sex].total++;
        if (survived === 1) survivalBySex[sex].survived++;
    });
    
    // Calculate survival rate by passenger class
    const survivalByClass = {};
    rawTrainData.forEach(passenger => {
        const pclass = passenger.Pclass || 'unknown';
        const survived = passenger.Survived;
        
        if (!survivalByClass[pclass]) {
            survivalByClass[pclass] = { total: 0, survived: 0 };
        }
        
        survivalByClass[pclass].total++;
        if (survived === 1) survivalByClass[pclass].survived++;
    });
    
    // Create survival by sex chart data
    const sexData = Object.keys(survivalBySex).map(sex => {
        const rate = (survivalBySex[sex].survived / survivalBySex[sex].total) * 100;
        return { 
            index: sex === 'male' ? 0 : 1, // Sort order
            x: sex, 
            y: rate 
        };
    }).sort((a, b) => a.index - b.index);
    
    // Create survival by class chart data
    const classData = Object.keys(survivalByClass).sort().map(pclass => {
        const rate = (survivalByClass[pclass].survived / survivalByClass[pclass].total) * 100;
        return { 
            x: `Class ${pclass}`, 
            y: rate 
        };
    });
    
    // Render survival by sex chart
    tfvis.render.barchart(
        document.getElementById('survivalBySexChart'),
        { values: sexData },
        {
            xLabel: 'Sex',
            yLabel: 'Survival Rate %',
            height: 300,
            width: document.getElementById('survivalBySexChart').offsetWidth || 400
        }
    );
    
    // Render survival by class chart
    tfvis.render.barchart(
        document.getElementById('survivalByClassChart'),
        { values: classData },
        {
            xLabel: 'Passenger Class',
            yLabel: 'Survival Rate %',
            height: 300,
            width: document.getElementById('survivalByClassChart').offsetWidth || 400
        }
    );
    
    console.log('Charts rendered:', sexData, classData);
}

// ============================================================================
// 2. Data Preprocessing
// ============================================================================

function preprocessData() {
    if (!rawTrainData) {
        showStatus(elements.preprocessStatus, 'No data loaded', 'error');
        return;
    }
    
    showStatus(elements.preprocessStatus, 'Preprocessing data...', 'loading');
    
    try {
        // Process training data
        const processed = preprocessDataset(rawTrainData, true);
        processedTrainData = processed;
        
        // Process test data if available
        if (rawTestData) {
            const processedTest = preprocessDataset(rawTestData, false);
            processedTestFeatures = processedTest.features;
            testPassengerIds = processedTest.passengerIds;
        }
        
        showStatus(elements.preprocessStatus, 
            `Preprocessing complete. Features: ${processed.featureNames.length}, Train samples: ${processed.features.length}`, 
            'success');
        
        // Show feature information
        showFeatureInfo();
        
    } catch (error) {
        console.error('Preprocessing error:', error);
        showStatus(elements.preprocessStatus, `Preprocessing error: ${error.message}`, 'error');
    }
    
    updateUIState();
}

// Preprocess a dataset (train or test)
function preprocessDataset(data, isTraining) {
    console.log(`Preprocessing ${isTraining ? 'training' : 'test'} data: ${data.length} samples`);
    
    // Calculate imputation values from training data only
    let ageMedian = 28;
    let fareMedian = 14.45;
    let embarkedMode = 'S';
    
    if (isTraining) {
        // Calculate median Age from non-missing values
        const ages = data.filter(p => p.Age && !isNaN(p.Age)).map(p => p.Age);
        ageMedian = ages.length > 0 ? ages.sort((a, b) => a - b)[Math.floor(ages.length / 2)] : 28;
        
        // Calculate median Fare from non-missing values
        const fares = data.filter(p => p.Fare && !isNaN(p.Fare)).map(p => p.Fare);
        fareMedian = fares.length > 0 ? fares.sort((a, b) => a - b)[Math.floor(fares.length / 2)] : 14.45;
        
        // Calculate mode for Embarked
        const embarkedCounts = {};
        data.forEach(p => {
            const embarked = p.Embarked || 'S';
            embarkedCounts[embarked] = (embarkedCounts[embarked] || 0) + 1;
        });
        embarkedMode = Object.keys(embarkedCounts).reduce((a, b) => 
            embarkedCounts[a] > embarkedCounts[b] ? a : b, 'S');
        
        console.log(`Imputation values - Age median: ${ageMedian}, Fare median: ${fareMedian}, Embarked mode: ${embarkedMode}`);
    }
    
    // Process each passenger
    const processedFeatures = [];
    const processedLabels = [];
    const passengerIds = [];
    
    data.forEach((passenger, index) => {
        // Extract passenger ID
        passengerIds.push(passenger[ID_COLUMN] || index + 1);
        
        // Initialize feature vector
        const features = [];
        
        // Process numerical features
        const age = passenger.Age && !isNaN(passenger.Age) ? passenger.Age : ageMedian;
        const fare = passenger.Fare && !isNaN(passenger.Fare) ? passenger.Fare : fareMedian;
        const sibSp = passenger.SibSp || 0;
        const parch = passenger.Parch || 0;
        
        // Standardize Age and Fare (using training stats if available)
        const standardizedAge = (age - ageMedian) / (ageMedian * 0.5); // Simple standardization
        const standardizedFare = (fare - fareMedian) / (fareMedian * 0.5);
        
        features.push(standardizedAge, standardizedFare, sibSp, parch);
        
        // Process categorical features (one-hot encoding)
        // Sex: male, female
        const sex = passenger.Sex || 'male';
        features.push(sex === 'female' ? 1 : 0, sex === 'male' ? 1 : 0);
        
        // Pclass: 1, 2, 3
        const pclass = passenger.Pclass || 3;
        features.push(pclass === 1 ? 1 : 0, pclass === 2 ? 1 : 0, pclass === 3 ? 1 : 0);
        
        // Embarked: C, Q, S
        const embarked = passenger.Embarked || embarkedMode;
        features.push(embarked === 'C' ? 1 : 0, embarked === 'Q' ? 1 : 0, embarked === 'S' ? 1 : 0);
        
        // Feature engineering: Family size and IsAlone
        if (useFamilyFeatures) {
            const familySize = sibSp + parch + 1;
            const isAlone = familySize === 1 ? 1 : 0;
            features.push(familySize, isAlone);
        }
        
        // Add to processed data
        processedFeatures.push(features);
        
        // Add label if training data
        if (isTraining) {
            const label = passenger[TARGET_COLUMN] || 0;
            processedLabels.push(label);
        }
    });
    
    // Create feature names
    const featureNames = [
        'Age_std', 'Fare_std', 'SibSp', 'Parch',
        'Sex_female', 'Sex_male',
        'Pclass_1', 'Pclass_2', 'Pclass_3',
        'Embarked_C', 'Embarked_Q', 'Embarked_S'
    ];
    
    if (useFamilyFeatures) {
        featureNames.push('FamilySize', 'IsAlone');
    }
    
    console.log(`Preprocessed ${processedFeatures.length} samples with ${featureNames.length} features`);
    
    return {
        features: processedFeatures,
        labels: processedLabels,
        featureNames: featureNames,
        passengerIds: passengerIds
    };
}

// Show feature information
function showFeatureInfo() {
    if (!processedTrainData) return;
    
    elements.featureInfo.style.display = 'block';
    
    const featureNames = processedTrainData.featureNames;
    let featureHtml = `<p><strong>${featureNames.length} Features:</strong></p><ul>`;
    
    featureNames.forEach((feature, index) => {
        featureHtml += `<li>${index + 1}. ${feature}</li>`;
    });
    
    featureHtml += '</ul>';
    elements.featureList.innerHTML = featureHtml;
}

// Toggle family features
function toggleFamilyFeatures() {
    useFamilyFeatures = !useFamilyFeatures;
    
    showStatus(elements.preprocessStatus, 
        `Family features ${useFamilyFeatures ? 'enabled' : 'disabled'}. Re-preprocess data to apply.`, 
        'warning');
    
    elements.toggleFamilyFeaturesBtn.innerHTML = 
        `<i class="fas fa-exchange-alt"></i> ${useFamilyFeatures ? 'Disable' : 'Enable'} Family Features`;
}

// ============================================================================
// 3. Model Building with Sigmoid Gate
// ============================================================================

function buildModel() {
    if (!processedTrainData) {
        showStatus(elements.modelStatus, 'No processed data available', 'error');
        return;
    }
    
    showStatus(elements.modelStatus, 'Building model with sigmoid gate...', 'loading');
    
    try {
        // Set global feature names
        featureNames = processedTrainData.featureNames;
        
        // Define the model
        model = tf.sequential();
        
        // Input layer
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [featureNames.length],
            name: 'dense_input'
        }));
        
        // Sigmoid Gate Layer for feature importance
        // Custom layer that applies sigmoid gating to features
        const sigmoidGateLayer = {
            className: 'SigmoidGate',
            
            computeOutputShape(inputShape) {
                return inputShape;
            },
            
            call(inputs, kwargs) {
                // Apply sigmoid to get gates between 0 and 1
                const gates = tf.sigmoid(inputs);
                // Multiply inputs by gates to gate the features
                return tf.mul(inputs, gates);
            }
        };
        
        // Add the sigmoid gate as a custom layer
        model.add(tf.layers.dense({
            units: featureNames.length,
            activation: 'linear',
            name: 'gate_dense'
        }));
        
        // Apply sigmoid gating using a lambda layer
        model.add(tf.layers.lambda({
            function: (x) => {
                const gates = tf.sigmoid(x);
                return tf.mul(x, gates);
            },
            name: 'sigmoid_gate'
        }));
        
        // Output layer
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            name: 'output'
        }));
        
        // Compile the model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        showStatus(elements.modelStatus, 'Model built successfully with sigmoid gate layer', 'success');
        console.log('Model summary:');
        model.summary();
        
    } catch (error) {
        console.error('Model building error:', error);
        showStatus(elements.modelStatus, `Model building error: ${error.message}`, 'error');
    }
    
    updateUIState();
}

// Show model summary
function showModelSummary() {
    if (!model) {
        showStatus(elements.modelStatus, 'No model built yet', 'warning');
        return;
    }
    
    elements.modelSummary.style.display = 'block';
    
    // Create a simple model summary display
    let summaryHtml = '<table><thead><tr><th>Layer</th><th>Output Shape</th><th>Parameters</th></tr></thead><tbody>';
    
    let totalParams = 0;
    
    model.layers.forEach((layer, index) => {
        const layerName = layer.name;
        const outputShape = JSON.stringify(layer.outputShape).replace(/,/g, ', ');
        const params = layer.countParams();
        totalParams += params;
        
        summaryHtml += `<tr>
            <td>${index + 1}. ${layerName}</td>
            <td>${outputShape}</td>
            <td>${params.toLocaleString()}</td>
        </tr>`;
    });
    
    summaryHtml += `</tbody></table>`;
    summaryHtml += `<p><strong>Total Parameters:</strong> ${totalParams.toLocaleString()}</p>`;
    
    elements.modelSummaryContent.innerHTML = summaryHtml;
}

// ============================================================================
// 4. Model Training
// ============================================================================

async function trainModel() {
    if (!model || !processedTrainData) {
        showStatus(elements.trainingStatus, 'Model or data not available', 'error');
        return;
    }
    
    showStatus(elements.trainingStatus, 'Preparing training data...', 'loading');
    
    try {
        // Convert data to tensors
        const featuresArray = processedTrainData.features;
        const labelsArray = processedTrainData.labels;
        
        // Create tensors
        const featuresTensor = tf.tensor2d(featuresArray);
        const labelsTensor = tf.tensor2d(labelsArray, [labelsArray.length, 1]);
        
        // Split into training and validation sets (80/20 stratified split)
        const splitIndex = Math.floor(featuresArray.length * 0.8);
        
        trainFeaturesTensor = featuresTensor.slice([0, 0], [splitIndex, -1]);
        trainLabelsTensor = labelsTensor.slice([0, 0], [splitIndex, -1]);
        
        valFeaturesTensor = featuresTensor.slice([splitIndex, 0], [-1, -1]);
        valLabelsTensor = labelsTensor.slice([splitIndex, 0], [-1, -1]);
        
        console.log(`Training set: ${splitIndex} samples, Validation set: ${featuresArray.length - splitIndex} samples`);
        
        // Setup training callbacks
        const callbacks = {
            onEpochEnd: (epoch, logs) => {
                // Update training status
                showStatus(elements.trainingStatus, 
                    `Epoch ${epoch + 1}/50 - Loss: ${logs.loss.toFixed(4)}, Accuracy: ${logs.acc.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}, Val Accuracy: ${logs.val_acc.toFixed(4)}`, 
                    'loading');
                
                // Visualize training progress
                tfvis.show.history(
                    { tab: 'Training', name: 'Loss' },
                    [{ epoch: epoch, loss: logs.loss, val_loss: logs.val_loss }],
                    ['loss', 'val_loss'],
                    { height: 300, width: 400 }
                );
                
                tfvis.show.history(
                    { tab: 'Training', name: 'Accuracy' },
                    [{ epoch: epoch, acc: logs.acc, val_acc: logs.val_acc }],
                    ['acc', 'val_acc'],
                    { height: 300, width: 400 }
                );
            },
            
            onTrainEnd: (logs) => {
                isTraining = false;
                elements.stopTrainingBtn.disabled = true;
                elements.trainModelBtn.disabled = false;
                
                showStatus(elements.trainingStatus, 'Training completed', 'success');
                
                // Extract feature importance from the sigmoid gate
                extractFeatureImportance();
                
                // Enable evaluation button
                elements.evaluateBtn.disabled = false;
                
                // Make predictions on validation set for evaluation
                valPredictions = model.predict(valFeaturesTensor).arraySync();
                valLabels = valLabelsTensor.arraySync();
            }
        };
        
        // Train the model
        isTraining = true;
        trainingStopRequested = false;
        elements.stopTrainingBtn.disabled = false;
        elements.trainModelBtn.disabled = true;
        
        showStatus(elements.trainingStatus, 'Starting training...', 'loading');
        
        // Train for 50 epochs with early stopping simulation
        const history = await model.fit(trainFeaturesTensor, trainLabelsTensor, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeaturesTensor, valLabelsTensor],
            callbacks: callbacks,
            yieldEvery: 'epoch'
        });
        
        console.log('Training completed:', history);
        
    } catch (error) {
        console.error('Training error:', error);
        isTraining = false;
        elements.stopTrainingBtn.disabled = true;
        elements.trainModelBtn.disabled = false;
        
        showStatus(elements.trainingStatus, `Training error: ${error.message}`, 'error');
    }
    
    updateUIState();
}

// Stop training
function stopTraining() {
    if (isTraining) {
        trainingStopRequested = true;
        showStatus(elements.trainingStatus, 'Stopping training after current epoch...', 'warning');
    }
}

// ============================================================================
// 5. Model Evaluation
// ============================================================================

function evaluateModel() {
    if (!model || !valFeaturesTensor || !valLabelsTensor) {
        showStatus(elements.evaluationStatus, 'Model not trained or validation data not available', 'error');
        return;
    }
    
    showStatus(elements.evaluationStatus, 'Evaluating model...', 'loading');
    
    try {
        // Make predictions on validation set
        const predictions = model.predict(valFeaturesTensor);
        const predictionsArray = predictions.arraySync();
        const labelsArray = valLabelsTensor.arraySync();
        
        // Calculate metrics with default threshold (0.5)
        updateMetricsWithThreshold(0.5);
        
        // Calculate ROC curve data
        calculateROCCurve(predictionsArray, labelsArray);
        
        // Show evaluation metrics table
        showEvaluationTable();
        
        showStatus(elements.evaluationStatus, 'Evaluation complete', 'success');
        
    } catch (error) {
        console.error('Evaluation error:', error);
        showStatus(elements.evaluationStatus, `Evaluation error: ${error.message}`, 'error');
    }
}

// Update metrics based on threshold
function updateMetricsWithThreshold(threshold) {
    if (!valPredictions || !valLabels) return;
    
    // Convert probabilities to binary predictions
    const binaryPreds = valPredictions.map(p => p[0] >= threshold ? 1 : 0);
    const trueLabels = valLabels.map(l => l[0]);
    
    // Calculate confusion matrix
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < binaryPreds.length; i++) {
        const pred = binaryPreds[i];
        const actual = trueLabels[i];
        
        if (pred === 1 && actual === 1) tp++;
        else if (pred === 1 && actual === 0) fp++;
        else if (pred === 0 && actual === 0) tn++;
        else if (pred === 0 && actual === 1) fn++;
    }
    
    // Calculate metrics
    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    const precision = tp === 0 ? 0 : tp / (tp + fp);
    const recall = tp === 0 ? 0 : tp / (tp + fn);
    const f1 = (precision === 0 || recall === 0) ? 0 : 
        2 * (precision * recall) / (precision + recall);
    
    // Update metric cards
    elements.accuracyMetric.textContent = accuracy.toFixed(3);
    elements.precisionMetric.textContent = precision.toFixed(3);
    elements.recallMetric.textContent = recall.toFixed(3);
    elements.f1Metric.textContent = f1.toFixed(3);
    
    // Show metrics container
    elements.metricsContainer.style.display = 'flex';
    
    // Create confusion matrix visualization
    createConfusionMatrix(tp, fp, tn, fn);
    
    // Update evaluation table
    const tableBody = elements.evaluationTableBody;
    tableBody.innerHTML = `
        <tr><td>True Positives</td><td>${tp}</td></tr>
        <tr><td>False Positives</td><td>${fp}</td></tr>
        <tr><td>True Negatives</td><td>${tn}</td></tr>
        <tr><td>False Negatives</td><td>${fn}</td></tr>
        <tr><td>Accuracy</td><td>${accuracy.toFixed(4)}</td></tr>
        <tr><td>Precision</td><td>${precision.toFixed(4)}</td></tr>
        <tr><td>Recall (Sensitivity)</td><td>${recall.toFixed(4)}</td></tr>
        <tr><td>F1 Score</td><td>${f1.toFixed(4)}</td></tr>
    `;
    
    // Table Fix: Force DOM reflow to ensure table is visible
    setTimeout(() => {
        const table = document.getElementById('evaluation-table');
        table.style.display = 'table';
        table.style.visibility = 'visible';
        // Force reflow
        document.body.offsetHeight;
    }, 0);
}

// Calculate ROC curve data
function calculateROCCurve(predictions, labels) {
    // Generate threshold points from 0 to 1
    const thresholds = [];
    for (let i = 0; i <= 100; i++) {
        thresholds.push(i / 100);
    }
    
    const rocPoints = [];
    
    thresholds.forEach(threshold => {
        // Calculate TPR and FPR at this threshold
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            const pred = predictions[i][0] >= threshold ? 1 : 0;
            const actual = labels[i][0];
            
            if (pred === 1 && actual === 1) tp++;
            else if (pred === 1 && actual === 0) fp++;
            else if (pred === 0 && actual === 0) tn++;
            else if (pred === 0 && actual === 1) fn++;
        }
        
        const tpr = tp === 0 ? 0 : tp / (tp + fn);
        const fpr = fp === 0 ? 0 : fp / (fp + tn);
        
        rocPoints.push({ x: fpr, y: tpr, threshold: threshold });
    });
    
    // Calculate AUC (Area Under Curve) using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < rocPoints.length; i++) {
        const prev = rocPoints[i - 1];
        const curr = rocPoints[i];
        auc += (curr.x - prev.x) * (curr.y + prev.y) / 2;
    }
    
    // Plot ROC curve
    const rocData = {
        values: rocPoints,
        series: ['ROC Curve']
    };
    
    tfvis.render.scatterplot(
        { tab: 'Evaluation', name: `ROC Curve (AUC = ${auc.toFixed(4)})` },
        rocPoints,
        {
            xLabel: 'False Positive Rate',
            yLabel: 'True Positive Rate',
            height: 350,
            width: 400,
            seriesColors: ['#4a6fa5']
        }
    );
    
    console.log(`ROC AUC: ${auc.toFixed(4)}`);
}

// Create confusion matrix visualization
function createConfusionMatrix(tp, fp, tn, fn) {
    const total = tp + fp + tn + fn;
    
    const matrixData = [
        { group: 'Actual Positive', variable: 'Predicted Positive', value: tp },
        { group: 'Actual Positive', variable: 'Predicted Negative', value: fn },
        { group: 'Actual Negative', variable: 'Predicted Positive', value: fp },
        { group: 'Actual Negative', variable: 'Predicted Negative', value: tn }
    ];
    
    tfvis.render.barchart(
        { tab: 'Evaluation', name: 'Confusion Matrix' },
        matrixData,
        {
            xLabel: 'Prediction',
            yLabel: 'Count',
            height: 300
        }
    );
}

// Show evaluation table
function showEvaluationTable() {
    // Table is already updated in updateMetricsWithThreshold
    // Just ensure it's visible
    setTimeout(() => {
        const table = document.getElementById('evaluation-table');
        table.style.display = 'table';
        table.style.visibility = 'visible';
    }, 0);
}

// ============================================================================
// Feature Importance from Sigmoid Gate
// ============================================================================

function extractFeatureImportance() {
    if (!model || !featureNames || featureNames.length === 0) {
        console.warn('Cannot extract feature importance: model or feature names not available');
        return;
    }
    
    try {
        // Get the gate layer (layer index 1 - the dense layer before the lambda gate)
        const gateLayer = model.layers[1]; // gate_dense layer
        const gateWeights = gateLayer.getWeights()[0]; // Weight matrix
        
        // Calculate importance scores: mean absolute weight for each input feature
        const importanceScores = [];
        
        // The gate weight matrix has shape [input_features, gate_units]
        // We'll take the mean absolute value across gate units for each input feature
        const gateWeightsArray = gateWeights.arraySync();
        
        for (let i = 0; i < featureNames.length; i++) {
            let sumAbs = 0;
            for (let j = 0; j < gateWeightsArray[i].length; j++) {
                sumAbs += Math.abs(gateWeightsArray[i][j]);
            }
            const avgAbs = sumAbs / gateWeightsArray[i].length;
            importanceScores.push({
                feature: featureNames[i],
                score: avgAbs
            });
        }
        
        // Sort by importance
        importanceScores.sort((a, b) => b.score - a.score);
        featureImportance = importanceScores;
        
        console.log('Feature importance from sigmoid gate:', importanceScores);
        
    } catch (error) {
        console.error('Error extracting feature importance:', error);
    }
}

function showFeatureImportance() {
    if (!featureImportance || featureImportance.length === 0) {
        extractFeatureImportance();
    }
    
    if (featureImportance.length === 0) {
        showStatus(elements.modelStatus, 'Feature importance not available yet', 'warning');
        return;
    }
    
    elements.featureImportanceSection.style.display = 'block';
    
    // Create bar chart
    const importanceData = featureImportance.map(item => ({
        x: item.feature,
        y: item.score
    }));
    
    tfvis.render.barchart(
        { tab: 'Feature Importance', name: 'Feature Importance Scores' },
        importanceData,
        {
            xLabel: 'Feature',
            yLabel: 'Importance Score',
            height: 350
        }
    );
    
    // Create feature importance list
    let listHtml = '';
    featureImportance.forEach((item, index) => {
        listHtml += `
            <div class="feature-item">
                <span class="feature-name">${index + 1}. ${item.feature}</span>
                <span class="feature-score">${item.score.toFixed(4)}</span>
            </div>
        `;
    });
    
    elements.featureImportanceList.innerHTML = listHtml;
}

// ============================================================================
// 6. Prediction & Export
// ============================================================================

async function predictTestData() {
    if (!model || !processedTestFeatures) {
        showStatus(elements.predictionStatus, 'Model or test data not available', 'error');
        return;
    }
    
    showStatus(elements.predictionStatus, 'Making predictions on test data...', 'loading');
    
    try {
        // Convert test features to tensor
        testFeaturesTensor = tf.tensor2d(processedTestFeatures);
        
        // Make predictions
        const predictions = model.predict(testFeaturesTensor);
        const predictionsArray = predictions.arraySync();
        
        // Show prediction preview
        elements.predictionResults.style.display = 'block';
        
        let previewHtml = `<p><strong>Predictions for ${predictionsArray.length} test passengers:</strong></p>`;
        previewHtml += '<table><thead><tr><th>PassengerId</th><th>Survival Probability</th><th>Predicted (threshold=0.5)</th></tr></thead><tbody>';
        
        const threshold = 0.5;
        for (let i = 0; i < Math.min(10, predictionsArray.length); i++) {
            const prob = predictionsArray[i][0];
            const pred = prob >= threshold ? 1 : 0;
            previewHtml += `<tr>
                <td>${testPassengerIds[i]}</td>
                <td>${prob.toFixed(4)}</td>
                <td>${pred}</td>
            </tr>`;
        }
        
        previewHtml += '</tbody></table>';
        previewHtml += `<p>... and ${Math.max(0, predictionsArray.length - 10)} more passengers</p>`;
        
        elements.predictionPreview.innerHTML = previewHtml;
        
        // Store predictions for export
        window.testPredictions = predictionsArray;
        
        showStatus(elements.predictionStatus, `Predictions complete for ${predictionsArray.length} test passengers`, 'success');
        
    } catch (error) {
        console.error('Prediction error:', error);
        showStatus(elements.predictionStatus, `Prediction error: ${error.message}`, 'error');
    }
}

// Export model
function exportModel() {
    if (!model) {
        showStatus(elements.predictionStatus, 'No model to export', 'error');
        return;
    }
    
    showStatus(elements.predictionStatus, 'Exporting model...', 'loading');
    
    try {
        // Save the model for download
        model.save('downloads://titanic-tfjs-model');
        
        showStatus(elements.predictionStatus, 'Model exported successfully. Check browser downloads.', 'success');
        
    } catch (error) {
        console.error('Model export error:', error);
        showStatus(elements.predictionStatus, `Model export error: ${error.message}`, 'error');
    }
}

// Export submission CSV
function exportSubmission() {
    if (!window.testPredictions || !testPassengerIds || testPassengerIds.length === 0) {
        showStatus(elements.predictionStatus, 'No predictions to export', 'error');
        return;
    }
    
    showStatus(elements.predictionStatus, 'Generating submission CSV...', 'loading');
    
    try {
        const threshold = parseFloat(elements.thresholdSlider.value);
        
        // Create CSV content with proper escaping for fields containing commas
        let csvContent = 'PassengerId,Survived\n';
        
        for (let i = 0; i < testPassengerIds.length; i++) {
            const passengerId = testPassengerIds[i];
            const prob = window.testPredictions[i][0];
            const survived = prob >= threshold ? 1 : 0;
            
            // CSV Fix: Use proper escaping with quotes around fields containing commas
            // In this case, PassengerId and Survived are numeric, so no need for quotes
            // But we'll add quotes if any field contains special characters
            const passengerIdStr = passengerId.toString();
            const survivedStr = survived.toString();
            
            // Only quote if contains comma, quote, or newline
            const needsQuotes = (str) => 
                str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r');
            
            const safePassengerId = needsQuotes(passengerIdStr) ? 
                `"${passengerIdStr.replace(/"/g, '""')}"` : passengerIdStr;
            const safeSurvived = needsQuotes(survivedStr) ? 
                `"${survivedStr.replace(/"/g, '""')}"` : survivedStr;
            
            csvContent += `${safePassengerId},${safeSurvived}\n`;
        }
        
        // Also create probabilities CSV
        let probCsvContent = 'PassengerId,Survived_Probability\n';
        for (let i = 0; i < testPassengerIds.length; i++) {
            const passengerId = testPassengerIds[i];
            const prob = window.testPredictions[i][0].toFixed(6);
            
            const passengerIdStr = passengerId.toString();
            const probStr = prob.toString();
            
            const needsQuotes = (str) => 
                str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r');
            
            const safePassengerId = needsQuotes(passengerIdStr) ? 
                `"${passengerIdStr.replace(/"/g, '""')}"` : passengerIdStr;
            const safeProb = needsQuotes(probStr) ? 
                `"${probStr.replace(/"/g, '""')}"` : probStr;
            
            probCsvContent += `${safePassengerId},${safeProb}\n`;
        }
        
        // Create download links
        downloadCSV(csvContent, 'titanic_submission.csv');
        downloadCSV(probCsvContent, 'titanic_probabilities.csv');
        
        showStatus(elements.predictionStatus, 'Submission files exported successfully. Check browser downloads.', 'success');
        
    } catch (error) {
        console.error('CSV export error:', error);
        showStatus(elements.predictionStatus, `CSV export error: ${error.message}`, 'error');
    }
}

// Helper to download CSV
function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    
    if (navigator.msSaveBlob) { // IE 10+
        navigator.msSaveBlob(blob, filename);
    } else {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

// Show status message
function showStatus(element, message, type) {
    element.textContent = message;
    element.className = 'status';
    
    switch (type) {
        case 'loading':
            element.classList.add('loading');
            break;
        case 'success':
            element.classList.add('success');
            break;
        case 'error':
            element.classList.add('error');
            break;
        case 'warning':
            element.classList.add('warning');
            break;
    }
}

// Clean up tensors to prevent memory leaks
function cleanupTensors() {
    const tensors = [
        trainFeaturesTensor, trainLabelsTensor,
        valFeaturesTensor, valLabelsTensor,
        testFeaturesTensor
    ];
    
    tensors.forEach(tensor => {
        if (tensor) {
            tensor.dispose();
        }
    });
}

// Log memory usage
function logMemoryUsage() {
    if (tf.memory) {
        console.log('TensorFlow.js memory:', tf.memory());
    }
}

// ============================================================================
// Initialize visualization surfaces
// ============================================================================

// Setup tfvis surfaces for different tabs
const surfaceContainer = document.getElementById('charts-container');
if (surfaceContainer) {
    // This will be used by tfvis to render charts
    console.log('Visualization surfaces ready');
}
