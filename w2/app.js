// Titanic Binary Classifier using TensorFlow.js
// Runs entirely in the browser - no server required

// Global variables to store data, model, and results
let rawTrainData = null;
let rawTestData = null;
let processedTrainData = null;
let processedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;
let testProbabilities = null;
let featureNames = [];
let featureImportances = [];
let isTraining = false;
let trainingController = null;

// Schema definition - easily swappable for other datasets
// Target: Survived (0/1). Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.
const TARGET_COLUMN = 'Survived';
const ID_COLUMN = 'PassengerId';
const FEATURE_COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];
const CATEGORICAL_COLUMNS = ['Pclass', 'Sex', 'Embarked'];
const NUMERICAL_COLUMNS = ['Age', 'SibSp', 'Parch', 'Fare'];

// DOM Elements
const elements = {
    trainFile: document.getElementById('trainFile'),
    testFile: document.getElementById('testFile'),
    loadDataBtn: document.getElementById('loadDataBtn'),
    trainStatus: document.getElementById('trainStatus'),
    testStatus: document.getElementById('testStatus'),
    dataPreview: document.getElementById('dataPreview'),
    dataInfo: document.getElementById('dataInfo'),
    preprocessBtn: document.getElementById('preprocessBtn'),
    visualizeBtn: document.getElementById('visualizeBtn'),
    preprocessStatus: document.getElementById('preprocessStatus'),
    familyFeatures: document.getElementById('familyFeatures'),
    createModelBtn: document.getElementById('createModelBtn'),
    summaryBtn: document.getElementById('summaryBtn'),
    modelStatus: document.getElementById('modelStatus'),
    modelSummary: document.getElementById('modelSummary'),
    trainBtn: document.getElementById('trainBtn'),
    stopTrainBtn: document.getElementById('stopTrainBtn'),
    trainingPlots: document.getElementById('trainingPlots'),
    evaluateBtn: document.getElementById('evaluateBtn'),
    rocBtn: document.getElementById('rocBtn'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue'),
    metricsDisplay: document.getElementById('metricsDisplay'),
    rocDisplay: document.getElementById('rocDisplay'),
    featureImportanceStatus: document.getElementById('featureImportanceStatus'),
    featureImportance: document.getElementById('featureImportance'),
    predictBtn: document.getElementById('predictBtn'),
    exportBtn: document.getElementById('exportBtn'),
    saveModelBtn: document.getElementById('saveModelBtn'),
    predictStatus: document.getElementById('predictStatus'),
    predictResults: document.getElementById('predictResults')
};

// Initialize event listeners
function initializeEventListeners() {
    elements.loadDataBtn.addEventListener('click', loadData);
    elements.preprocessBtn.addEventListener('click', preprocessData);
    elements.visualizeBtn.addEventListener('click', visualizeData);
    elements.createModelBtn.addEventListener('click', createModel);
    elements.summaryBtn.addEventListener('click', showModelSummary);
    elements.trainBtn.addEventListener('click', trainModel);
    elements.stopTrainBtn.addEventListener('click', stopTraining);
    elements.evaluateBtn.addEventListener('click', evaluateModel);
    elements.rocBtn.addEventListener('click', plotROCCurve);
    elements.thresholdSlider.addEventListener('input', updateThreshold);
    elements.predictBtn.addEventListener('click', predictTestData);
    elements.exportBtn.addEventListener('click', exportResults);
    elements.saveModelBtn.addEventListener('click', saveModel);
    
    // Update threshold display when slider changes
    elements.thresholdSlider.addEventListener('input', function() {
        elements.thresholdValue.textContent = this.value;
    });
}

// Parse CSV with proper quote handling
function parseCSV(csvText) {
    const lines = csvText.split('\n');
    const result = [];
    
    if (lines.length === 0) return result;
    
    // Extract headers (first row)
    const headers = parseCSVLine(lines[0]);
    
    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim() === '') continue;
        
        const values = parseCSVLine(lines[i]);
        if (values.length !== headers.length) {
            console.warn(`Line ${i+1} has ${values.length} values, expected ${headers.length}. Skipping.`);
            continue;
        }
        
        const row = {};
        headers.forEach((header, index) => {
            // Remove surrounding quotes if present and convert to appropriate type
            let value = values[index];
            if (value === '') {
                row[header] = null;
            } else {
                // Try to convert to number if possible
                const numValue = parseFloat(value);
                row[header] = isNaN(numValue) ? value : numValue;
            }
        });
        result.push(row);
    }
    
    return result;
}

// Parse a single CSV line with proper quote handling
function parseCSVLine(line) {
    const values = [];
    let currentValue = '';
    let insideQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = i < line.length - 1 ? line[i + 1] : '';
        
        if (char === '"') {
            if (insideQuotes && nextChar === '"') {
                // Escaped quote
                currentValue += '"';
                i++; // Skip next character
            } else {
                // Start or end of quoted field
                insideQuotes = !insideQuotes;
            }
        } else if (char === ',' && !insideQuotes) {
            // End of field
            values.push(currentValue);
            currentValue = '';
        } else {
            currentValue += char;
        }
    }
    
    // Add the last value
    values.push(currentValue);
    
    return values;
}

// Load and parse CSV files
async function loadData() {
    const trainFile = elements.trainFile.files[0];
    const testFile = elements.testFile.files[0];
    
    if (!trainFile) {
        alert('Please select a training CSV file.');
        return;
    }
    
    try {
        // Load training data
        const trainText = await readFileAsText(trainFile);
        rawTrainData = parseCSV(trainText);
        elements.trainStatus.textContent = `Loaded ${rawTrainData.length} training samples`;
        elements.trainStatus.className = 'status success';
        
        // Load test data if available
        if (testFile) {
            const testText = await readFileAsText(testFile);
            rawTestData = parseCSV(testText);
            elements.testStatus.textContent = `Loaded ${rawTestData.length} test samples`;
            elements.testStatus.className = 'status success';
        } else {
            elements.testStatus.textContent = 'No test file selected (optional)';
            elements.testStatus.className = 'status';
        }
        
        // Show data preview
        showDataPreview();
        
        // Update button states
        elements.preprocessBtn.disabled = false;
        elements.visualizeBtn.disabled = false;
        
        // Show data info
        analyzeData();
        
    } catch (error) {
        alert(`Error loading CSV files: ${error.message}`);
        console.error(error);
    }
}

// Read file as text
function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// Show preview of loaded data
function showDataPreview() {
    if (!rawTrainData || rawTrainData.length === 0) return;
    
    const previewCount = Math.min(10, rawTrainData.length);
    const headers = Object.keys(rawTrainData[0]);
    
    let html = '<table class="evaluation-table"><thead><tr>';
    headers.forEach(header => {
        html += `<th>${header}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    for (let i = 0; i < previewCount; i++) {
        html += '<tr>';
        headers.forEach(header => {
            html += `<td>${rawTrainData[i][header] ?? 'null'}</td>`;
        });
        html += '</tr>';
    }
    
    html += '</tbody></table>';
    html += `<p>Showing ${previewCount} of ${rawTrainData.length} rows</p>`;
    
    elements.dataPreview.innerHTML = html;
}

// Analyze and display data information
function analyzeData() {
    if (!rawTrainData || rawTrainData.length === 0) return;
    
    const totalRows = rawTrainData.length;
    const columns = Object.keys(rawTrainData[0]);
    
    let info = `<strong>Dataset Shape:</strong> ${totalRows} rows × ${columns.length} columns<br>`;
    info += `<strong>Columns:</strong> ${columns.join(', ')}<br>`;
    
    // Check for missing values
    let missingInfo = '<strong>Missing Values:</strong><br>';
    columns.forEach(col => {
        const missingCount = rawTrainData.filter(row => row[col] === null || row[col] === undefined || row[col] === '').length;
        const missingPercent = (missingCount / totalRows * 100).toFixed(1);
        if (missingCount > 0) {
            missingInfo += `${col}: ${missingCount} (${missingPercent}%)<br>`;
        }
    });
    
    if (missingInfo === '<strong>Missing Values:</strong><br>') {
        missingInfo += 'None';
    }
    
    info += missingInfo;
    
    // Target distribution
    if (rawTrainData[0].hasOwnProperty(TARGET_COLUMN)) {
        const survivedCount = rawTrainData.filter(row => row[TARGET_COLUMN] === 1).length;
        const notSurvivedCount = rawTrainData.filter(row => row[TARGET_COLUMN] === 0).length;
        const survivalRate = (survivedCount / totalRows * 100).toFixed(1);
        
        info += `<strong>Target Distribution:</strong><br>`;
        info += `Survived (1): ${survivedCount} (${survivalRate}%)<br>`;
        info += `Not Survived (0): ${notSurvivedCount} (${(100 - survivalRate).toFixed(1)}%)`;
    }
    
    elements.dataInfo.innerHTML = info;
    elements.dataInfo.className = 'status';
}

// Preprocess the data
function preprocessData() {
    if (!rawTrainData || rawTrainData.length === 0) {
        alert('Please load training data first.');
        return;
    }
    
    try {
        elements.preprocessStatus.textContent = 'Preprocessing data...';
        elements.preprocessStatus.className = 'status';
        
        // Extract features and labels
        const {features, labels, featureNames: fNames} = preprocessDataset(rawTrainData, true);
        
        // Store processed data
        processedTrainData = {features, labels};
        featureNames = fNames;
        
        // Preprocess test data if available
        if (rawTestData && rawTestData.length > 0) {
            const testFeatures = preprocessDataset(rawTestData, false).features;
            processedTestData = {features: testFeatures};
        }
        
        elements.preprocessStatus.textContent = `Preprocessing complete. Features: ${features.shape[1]}, Samples: ${features.shape[0]}`;
        elements.preprocessStatus.className = 'status success';
        
        // Update button states
        elements.createModelBtn.disabled = false;
        
        console.log('Preprocessed features shape:', features.shape);
        console.log('Feature names:', featureNames);
        
    } catch (error) {
        elements.preprocessStatus.textContent = `Error during preprocessing: ${error.message}`;
        elements.preprocessStatus.className = 'status error';
        console.error(error);
    }
}

// Main preprocessing function
function preprocessDataset(data, isTraining) {
    const featuresArray = [];
    const labelsArray = [];
    
    // Calculate statistics from training data only
    let ageMedian = 28;
    let fareMedian = 14.45;
    let embarkedMode = 'S';
    
    if (isTraining) {
        // Calculate statistics
        const ages = data.map(row => row.Age).filter(age => age !== null);
        const fares = data.map(row => row.Fare).filter(fare => fare !== null);
        const embarked = data.map(row => row.Embarked).filter(e => e !== null);
        
        ageMedian = ages.length > 0 ? median(ages) : 28;
        fareMedian = fares.length > 0 ? median(fares) : 14.45;
        embarkedMode = embarked.length > 0 ? mode(embarked) : 'S';
        
        console.log('Preprocessing statistics:', {ageMedian, fareMedian, embarkedMode});
    }
    
    // Process each row
    for (const row of data) {
        const featureValues = [];
        
        // Handle Pclass (categorical)
        const pclass = row.Pclass || 3;
        const pclassOneHot = oneHotEncode(pclass, [1, 2, 3]);
        featureValues.push(...pclassOneHot);
        
        // Handle Sex (categorical)
        const sex = row.Sex || 'male';
        const sexOneHot = oneHotEncode(sex, ['male', 'female']);
        featureValues.push(...sexOneHot);
        
        // Handle Age (numerical, impute with median)
        let age = row.Age;
        if (age === null || age === undefined) {
            age = ageMedian;
        }
        // Standardize age (z-score normalization would be better but we'll use min-max for simplicity)
        const standardizedAge = (age - 0) / (80 - 0); // Approximate min-max
        featureValues.push(standardizedAge);
        
        // Handle SibSp (numerical)
        const sibSp = row.SibSp || 0;
        featureValues.push(sibSp / 8); // Normalize by max value
        
        // Handle Parch (numerical)
        const parch = row.Parch || 0;
        featureValues.push(parch / 6); // Normalize by max value
        
        // Handle Fare (numerical, impute with median)
        let fare = row.Fare;
        if (fare === null || fare === undefined) {
            fare = fareMedian;
        }
        const standardizedFare = (fare - 0) / (512 - 0); // Normalize by max approximate fare
        featureValues.push(standardizedFare);
        
        // Handle Embarked (categorical, impute with mode)
        let embarked = row.Embarked;
        if (embarked === null || embarked === undefined) {
            embarked = embarkedMode;
        }
        const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);
        featureValues.push(...embarkedOneHot);
        
        // Optional: Add family features
        if (elements.familyFeatures.checked) {
            const familySize = sibSp + parch + 1;
            const isAlone = familySize === 1 ? 1 : 0;
            
            featureValues.push(familySize / 11); // Normalize by max family size
            featureValues.push(isAlone);
        }
        
        featuresArray.push(featureValues);
        
        // Add label if training data
        if (isTraining && row.hasOwnProperty(TARGET_COLUMN)) {
            labelsArray.push(row[TARGET_COLUMN]);
        }
    }
    
    // Create feature names
    const featureNames = [
        'Pclass_1', 'Pclass_2', 'Pclass_3',
        'Sex_male', 'Sex_female',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Embarked_C', 'Embarked_Q', 'Embarked_S'
    ];
    
    if (elements.familyFeatures.checked) {
        featureNames.push('FamilySize', 'IsAlone');
    }
    
    // Convert to tensors
    const features = tf.tensor2d(featuresArray);
    const labels = isTraining ? tf.tensor1d(labelsArray) : null;
    
    return {features, labels, featureNames};
}

// Helper functions
function median(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function mode(arr) {
    const counts = {};
    arr.forEach(val => counts[val] = (counts[val] || 0) + 1);
    return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
}

function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(value);
    if (index !== -1) {
        encoding[index] = 1;
    }
    return encoding;
}

// Visualize the data
function visualizeData() {
    if (!rawTrainData || rawTrainData.length === 0) {
        alert('Please load data first.');
        return;
    }
    
    // Create visualizations with tfjs-vis
    const surfaceNames = ['Survival by Sex', 'Survival by Pclass'];
    
    // Survival by Sex
    const sexSurvival = {};
    rawTrainData.forEach(row => {
        if (row.Sex && row.Survived !== undefined) {
            const key = row.Sex;
            if (!sexSurvival[key]) {
                sexSurvival[key] = {survived: 0, total: 0};
            }
            sexSurvival[key].total++;
            if (row.Survived === 1) {
                sexSurvival[key].survived++;
            }
        }
    });
    
    const sexData = Object.keys(sexSurvival).map(sex => ({
        x: sex,
        y: (sexSurvival[sex].survived / sexSurvival[sex].total) * 100
    }));
    
    // Survival by Pclass
    const pclassSurvival = {};
    rawTrainData.forEach(row => {
        if (row.Pclass && row.Survived !== undefined) {
            const key = `Class ${row.Pclass}`;
            if (!pclassSurvival[key]) {
                pclassSurvival[key] = {survived: 0, total: 0};
            }
            pclassSurvival[key].total++;
            if (row.Survived === 1) {
                pclassSurvival[key].survived++;
            }
        }
    });
    
    const pclassData = Object.keys(pclassSurvival).map(pclass => ({
        x: pclass,
        y: (pclassSurvival[pclass].survived / pclassSurvival[pclass].total) * 100
    }));
    
    // Render charts
    tfvis.render.barchart(
        {name: 'Survival Rate by Sex', tab: 'Visualization'},
        {values: sexData},
        {xLabel: 'Sex', yLabel: 'Survival Rate (%)', height: 300}
    );
    
    tfvis.render.barchart(
        {name: 'Survival Rate by Passenger Class', tab: 'Visualization'},
        {values: pclassData},
        {xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)', height: 300}
    );
}

// Create the neural network model
function createModel() {
    if (!processedTrainData) {
        alert('Please preprocess data first.');
        return;
    }
    
    try {
        elements.modelStatus.textContent = 'Creating model...';
        elements.modelStatus.className = 'status';
        
        // Get input shape
        const inputShape = processedTrainData.features.shape[1];
        
        // Create sequential model
        model = tf.sequential();
        
        // Hidden layer with 16 units, ReLU activation
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [inputShape],
            kernelInitializer: 'glorotNormal',
            name: 'hidden_layer'
        }));
        
        // Output layer with 1 unit, sigmoid activation for binary classification
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            name: 'output_layer'
        }));
        
        // Compile the model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        elements.modelStatus.textContent = 'Model created successfully!';
        elements.modelStatus.className = 'status success';
        
        // Update button states
        elements.summaryBtn.disabled = false;
        elements.trainBtn.disabled = false;
        
    } catch (error) {
        elements.modelStatus.textContent = `Error creating model: ${error.message}`;
        elements.modelStatus.className = 'status error';
        console.error(error);
    }
}

// Show model summary
function showModelSummary() {
    if (!model) {
        alert('Please create a model first.');
        return;
    }
    
    // Count parameters
    let totalParams = 0;
    let trainableParams = 0;
    let nonTrainableParams = 0;
    
    model.summary(undefined, undefined, (layer, layerSummary) => {
        totalParams += layerSummary.totalParams;
        trainableParams += layerSummary.trainableParams;
        nonTrainableParams += layerSummary.nonTrainableParams;
    });
    
    let summaryHTML = '<h3>Model Architecture</h3>';
    summaryHTML += '<table class="evaluation-table">';
    summaryHTML += '<tr><th>Layer (type)</th><th>Output Shape</th><th>Param #</th></tr>';
    
    // Add layers
    model.layers.forEach((layer, i) => {
        const layerType = layer.getClassName();
        const outputShape = JSON.stringify(layer.outputShape.slice(1));
        const params = layer.countParams();
        
        summaryHTML += `<tr>
            <td>${layer.name} (${layerType})</td>
            <td>${outputShape}</td>
            <td>${params.toLocaleString()}</td>
        </tr>`;
    });
    
    summaryHTML += `<tr>
        <td colspan="2"><strong>Total Parameters</strong></td>
        <td><strong>${totalParams.toLocaleString()}</strong></td>
    </tr>`;
    summaryHTML += `<tr>
        <td colspan="2"><strong>Trainable Parameters</strong></td>
        <td><strong>${trainableParams.toLocaleString()}</strong></td>
    </tr>`;
    summaryHTML += `<tr>
        <td colspan="2"><strong>Non-trainable Parameters</strong></td>
        <td><strong>${nonTrainableParams.toLocaleString()}</strong></td>
    </tr>`;
    summaryHTML += '</table>';
    
    elements.modelSummary.innerHTML = summaryHTML;
}

// Train the model
async function trainModel() {
    if (!model || !processedTrainData) {
        alert('Please create model and preprocess data first.');
        return;
    }
    
    try {
        elements.trainStatus.textContent = 'Training model...';
        elements.trainStatus.className = 'status';
        elements.trainBtn.disabled = true;
        elements.stopTrainBtn.disabled = false;
        isTraining = true;
        
        // Create validation split (80/20 stratified)
        const {trainFeatures, trainLabels, valFeatures, valLabels} = createValidationSplit(
            processedTrainData.features, 
            processedTrainData.labels
        );
        
        // Store validation data for later evaluation
        validationData = valFeatures;
        validationLabels = valLabels;
        
        // Prepare for early stopping
        let bestValLoss = Infinity;
        let patienceCounter = 0;
        const patience = 5;
        
        // Training configuration
        const epochs = 50;
        const batchSize = 32;
        
        // Create training controller for early stopping
        trainingController = new tf.Callback();
        trainingController.onEpochEnd = async (epoch, logs) => {
            // Check for early stopping
            if (logs.val_loss < bestValLoss) {
                bestValLoss = logs.val_loss;
                patienceCounter = 0;
            } else {
                patienceCounter++;
                if (patienceCounter >= patience && epoch >= 10) {
                    model.stopTraining = true;
                    elements.trainStatus.textContent += `\nEarly stopping at epoch ${epoch + 1}`;
                }
            }
            
            // Update status
            elements.trainStatus.textContent = `Epoch ${epoch + 1}/${epochs} - loss: ${logs.loss.toFixed(4)}, accuracy: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`;
        };
        
        // Train the model
        const history = await model.fit(trainFeatures, trainLabels, {
            epochs,
            batchSize,
            validationData: [valFeatures, valLabels],
            callbacks: [
                trainingController,
                tfvis.show.fitCallbacks(
                    {name: 'Training Performance', tab: 'Training'},
                    ['loss', 'val_loss', 'acc', 'val_acc'],
                    {callbacks: ['onEpochEnd']}
                )
            ],
            verbose: 0
        });
        
        trainingHistory = history;
        
        // Clean up tensors
        trainFeatures.dispose();
        trainLabels.dispose();
        valFeatures.dispose();
        valLabels.dispose();
        
        elements.trainStatus.textContent += '\nTraining complete!';
        elements.trainStatus.className = 'status success';
        
        // Update button states
        elements.evaluateBtn.disabled = false;
        elements.rocBtn.disabled = false;
        elements.predictBtn.disabled = false;
        elements.saveModelBtn.disabled = false;
        
        // Calculate feature importance
        calculateFeatureImportance();
        
    } catch (error) {
        elements.trainStatus.textContent = `Error during training: ${error.message}`;
        elements.trainStatus.className = 'status error';
        console.error(error);
    } finally {
        elements.trainBtn.disabled = false;
        elements.stopTrainBtn.disabled = true;
        isTraining = false;
        trainingController = null;
    }
}

// Create validation split (stratified)
function createValidationSplit(features, labels, splitRatio = 0.2) {
    // Get indices for each class
    const class0Indices = [];
    const class1Indices = [];
    
    const labelsArray = labels.arraySync();
    for (let i = 0; i < labelsArray.length; i++) {
        if (labelsArray[i] === 0) {
            class0Indices.push(i);
        } else {
            class1Indices.push(i);
        }
    }
    
    // Shuffle indices
    tf.util.shuffle(class0Indices);
    tf.util.shuffle(class1Indices);
    
    // Calculate split sizes for each class
    const class0ValSize = Math.floor(class0Indices.length * splitRatio);
    const class1ValSize = Math.floor(class1Indices.length * splitRatio);
    
    // Get validation indices
    const valIndices = [
        ...class0Indices.slice(0, class0ValSize),
        ...class1Indices.slice(0, class1ValSize)
    ];
    
    // Get training indices
    const trainIndices = [
        ...class0Indices.slice(class0ValSize),
        ...class1Indices.slice(class1ValSize)
    ];
    
    // Shuffle indices
    tf.util.shuffle(valIndices);
    tf.util.shuffle(trainIndices);
    
    // Create tensors
    const trainFeatures = tf.gather(features, trainIndices);
    const trainLabels = tf.gather(labels, trainIndices);
    const valFeatures = tf.gather(features, valIndices);
    const valLabels = tf.gather(labels, valIndices);
    
    return {trainFeatures, trainLabels, valFeatures, valLabels};
}

// Stop training
function stopTraining() {
    if (trainingController && isTraining) {
        model.stopTraining = true;
        elements.trainStatus.textContent += '\nTraining stopped by user.';
        elements.stopTrainBtn.disabled = true;
        isTraining = false;
    }
}

// Evaluate the model
async function evaluateModel() {
    if (!model || !validationData || !validationLabels) {
        alert('Please train the model first.');
        return;
    }
    
    try {
        // Make predictions on validation set
        const probs = model.predict(validationData);
        validationPredictions = probs;
        
        // Calculate metrics with current threshold
        const threshold = parseFloat(elements.thresholdSlider.value);
        updateMetrics(threshold);
        
    } catch (error) {
        alert(`Error during evaluation: ${error.message}`);
        console.error(error);
    }
}

// Update metrics based on threshold
function updateMetrics(threshold) {
    if (!validationPredictions || !validationLabels) return;
    
    // Get predictions and labels as arrays
    const probs = validationPredictions.arraySync();
    const labels = validationLabels.arraySync();
    
    // Calculate confusion matrix
    let truePositives = 0;
    let falsePositives = 0;
    let trueNegatives = 0;
    let falseNegatives = 0;
    
    for (let i = 0; i < labels.length; i++) {
        const prediction = probs[i][0] >= threshold ? 1 : 0;
        const actual = labels[i];
        
        if (prediction === 1 && actual === 1) truePositives++;
        else if (prediction === 1 && actual === 0) falsePositives++;
        else if (prediction === 0 && actual === 0) trueNegatives++;
        else if (prediction === 0 && actual === 1) falseNegatives++;
    }
    
    // Calculate metrics
    const accuracy = (truePositives + trueNegatives) / labels.length;
    const precision = truePositives / (truePositives + falsePositives) || 0;
    const recall = truePositives / (truePositives + falseNegatives) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    // Display confusion matrix
    let html = '<h3>Confusion Matrix</h3>';
    html += '<table class="evaluation-table">';
    html += '<tr><th></th><th colspan="2">Predicted</th></tr>';
    html += '<tr><th rowspan="2">Actual</th><th>Negative (0)</th><th>Positive (1)</th></tr>';
    html += `<tr><td>${trueNegatives}</td><td>${falsePositives}</td></tr>`;
    html += `<tr><th>Negative (0)</th><td colspan="2">${trueNegatives + falsePositives}</td></tr>`;
    html += `<tr><th>Positive (1)</th><td>${falseNegatives}</td><td>${truePositives}</td></tr>`;
    html += `<tr><th></th><td colspan="2">${falseNegatives + truePositives}</td></tr>`;
    html += '</table>';
    
    // Display metrics
    html += '<div class="metrics-grid">';
    html += `<div class="metric-box"><div>Accuracy</div><div class="metric-value">${accuracy.toFixed(4)}</div></div>`;
    html += `<div class="metric-box"><div>Precision</div><div class="metric-value">${precision.toFixed(4)}</div></div>`;
    html += `<div class="metric-box"><div>Recall</div><div class="metric-value">${recall.toFixed(4)}</div></div>`;
    html += `<div class="metric-box"><div>F1-Score</div><div class="metric-value">${f1.toFixed(4)}</div></div>`;
    html += '</div>';
    
    elements.metricsDisplay.innerHTML = html;
}

// Plot ROC curve
async function plotROCCurve() {
    if (!validationPredictions || !validationLabels) {
        alert('Please evaluate the model first.');
        return;
    }
    
    try {
        // Get predictions and labels
        const probs = validationPredictions.arraySync().flat();
        const labels = validationLabels.arraySync();
        
        // Calculate ROC curve points
        const thresholds = Array.from({length: 101}, (_, i) => i / 100);
        const rocPoints = [];
        
        for (const threshold of thresholds) {
            let truePositives = 0;
            let falsePositives = 0;
            let trueNegatives = 0;
            let falseNegatives = 0;
            
            for (let i = 0; i < labels.length; i++) {
                const prediction = probs[i] >= threshold ? 1 : 0;
                const actual = labels[i];
                
                if (prediction === 1 && actual === 1) truePositives++;
                else if (prediction === 1 && actual === 0) falsePositives++;
                else if (prediction === 0 && actual === 0) trueNegatives++;
                else if (prediction === 0 && actual === 1) falseNegatives++;
            }
            
            const tpr = truePositives / (truePositives + falseNegatives) || 0;
            const fpr = falsePositives / (falsePositives + trueNegatives) || 0;
            
            rocPoints.push({x: fpr, y: tpr, threshold});
        }
        
        // Calculate AUC (trapezoidal rule)
        let auc = 0;
        for (let i = 1; i < rocPoints.length; i++) {
            auc += (rocPoints[i].x - rocPoints[i-1].x) * 
                   (rocPoints[i].y + rocPoints[i-1].y) / 2;
        }
        
        // Render ROC curve
        tfvis.render.linechart(
            {name: 'ROC Curve', tab: 'Evaluation'},
            {values: rocPoints, series: ['ROC']},
            {
                xLabel: 'False Positive Rate',
                yLabel: 'True Positive Rate',
                height: 400,
                width: 600,
                seriesColors: ['#3498db']
            }
        );
        
        // Display AUC
        elements.rocDisplay.innerHTML = `<div class="metric-box">
            <div>Area Under ROC Curve (AUC)</div>
            <div class="metric-value">${auc.toFixed(4)}</div>
        </div>`;
        
    } catch (error) {
        alert(`Error plotting ROC curve: ${error.message}`);
        console.error(error);
    }
}

// Calculate feature importance using Sigmoid gate
function calculateFeatureImportance() {
    if (!model || !featureNames || featureNames.length === 0) {
        return;
    }
    
    try {
        elements.featureImportanceStatus.textContent = 'Calculating feature importance...';
        elements.featureImportanceStatus.className = 'status';
        
        // Get weights from the first (hidden) layer
        const hiddenLayer = model.getLayer('hidden_layer');
        const weights = hiddenLayer.getWeights()[0]; // Weight matrix (input_features x hidden_units)
        
        // Transpose to get (hidden_units x input_features)
        const weightsTransposed = weights.transpose();
        
        // For each feature, calculate importance as sigmoid(W * feature_range)
        // We approximate feature_range as 1 for normalized features
        const featureRange = 1.0;
        
        const importanceScores = [];
        const weightsArray = weightsTransposed.arraySync();
        
        for (let i = 0; i < featureNames.length; i++) {
            // Calculate sum of absolute weights for this feature across all hidden units
            let sum = 0;
            for (let j = 0; j < weightsArray.length; j++) {
                sum += Math.abs(weightsArray[j][i]);
            }
            
            // Apply sigmoid to get importance score
            const importance = 1 / (1 + Math.exp(-sum * featureRange));
            importanceScores.push({
                name: featureNames[i],
                importance: importance,
                weightSum: sum
            });
        }
        
        // Sort by importance (descending)
        importanceScores.sort((a, b) => b.importance - a.importance);
        
        // Store for later use
        featureImportances = importanceScores;
        
        // Display top features
        displayFeatureImportance(importanceScores);
        
        elements.featureImportanceStatus.textContent = 'Feature importance calculated.';
        elements.featureImportanceStatus.className = 'status success';
        
    } catch (error) {
        elements.featureImportanceStatus.textContent = `Error calculating feature importance: ${error.message}`;
        elements.featureImportanceStatus.className = 'status error';
        console.error(error);
    }
}

// Display feature importance
function displayFeatureImportance(importanceScores) {
    const topN = Math.min(10, importanceScores.length);
    
    let html = `<h4>Top ${topN} Most Important Features</h4>`;
    html += '<table class="evaluation-table">';
    html += '<tr><th>Rank</th><th>Feature</th><th>Importance Score</th><th>Weight Sum</th></tr>';
    
    for (let i = 0; i < topN; i++) {
        const feature = importanceScores[i];
        html += `<tr>
            <td>${i + 1}</td>
            <td>${feature.name}</td>
            <td>${feature.importance.toFixed(4)}</td>
            <td>${feature.weightSum.toFixed(4)}</td>
        </tr>`;
    }
    
    html += '</table>';
    
    // Add bar chart visualization
    html += '<div class="feature-importance">';
    
    // Find max importance for scaling
    const maxImportance = importanceScores[0].importance;
    
    for (let i = 0; i < topN; i++) {
        const feature = importanceScores[i];
        const widthPercent = (feature.importance / maxImportance) * 100;
        
        html += `<div class="feature-label">
            <span>${feature.name}</span>
            <span>${feature.importance.toFixed(4)}</span>
        </div>`;
        html += `<div class="feature-bar" style="width: ${widthPercent}%"></div>`;
    }
    
    html += '</div>';
    
    elements.featureImportance.innerHTML = html;
}

// Predict on test data
async function predictTestData() {
    if (!model || !processedTestData) {
        alert('Please load test data and train the model first.');
        return;
    }
    
    try {
        elements.predictStatus.textContent = 'Making predictions...';
        elements.predictStatus.className = 'status';
        
        // Make predictions
        const probsTensor = model.predict(processedTestData.features);
        testProbabilities = probsTensor.arraySync();
        
        // Apply threshold to get binary predictions
        const threshold = parseFloat(elements.thresholdSlider.value);
        const predictions = testProbabilities.map(prob => prob[0] >= threshold ? 1 : 0);
        
        // Store for export
        testPredictions = predictions;
        
        // Display some predictions
        let html = '<h3>Test Data Predictions (First 10 Rows)</h3>';
        html += '<table class="evaluation-table">';
        html += '<tr><th>Index</th><th>Probability</th><th>Prediction (Threshold=' + threshold + ')</th></tr>';
        
        const displayCount = Math.min(10, predictions.length);
        for (let i = 0; i < displayCount; i++) {
            html += `<tr>
                <td>${i}</td>
                <td>${testProbabilities[i][0].toFixed(4)}</td>
                <td>${predictions[i]}</td>
            </tr>`;
        }
        
        html += '</table>';
        html += `<p>Total predictions: ${predictions.length}</p>`;
        
        // Show class distribution
        const survivedCount = predictions.filter(p => p === 1).length;
        const notSurvivedCount = predictions.filter(p => p === 0).length;
        const survivalRate = (survivedCount / predictions.length * 100).toFixed(1);
        
        html += `<div class="metrics-grid">
            <div class="metric-box">
                <div>Predicted Survived</div>
                <div class="metric-value">${survivedCount}</div>
                <div>(${survivalRate}%)</div>
            </div>
            <div class="metric-box">
                <div>Predicted Not Survived</div>
                <div class="metric-value">${notSurvivedCount}</div>
                <div>(${(100 - survivalRate).toFixed(1)}%)</div>
            </div>
        </div>`;
        
        elements.predictResults.innerHTML = html;
        elements.predictStatus.textContent = `Predictions complete: ${predictions.length} samples`;
        elements.predictStatus.className = 'status success';
        
        // Enable export button
        elements.exportBtn.disabled = false;
        
    } catch (error) {
        elements.predictStatus.textContent = `Error making predictions: ${error.message}`;
        elements.predictStatus.className = 'status error';
        console.error(error);
    }
}

// Export results
function exportResults() {
    if (!testPredictions || !rawTestData) {
        alert('Please make predictions first.');
        return;
    }
    
    try {
        // Prepare submission data
        const submissionData = [];
        const probabilitiesData = [];
        
        for (let i = 0; i < rawTestData.length; i++) {
            const passengerId = rawTestData[i][ID_COLUMN] || i + 892; // Default to Kaggle test IDs
            const prediction = testPredictions[i];
            const probability = testProbabilities[i][0];
            
            // Submission CSV: PassengerId, Survived
            submissionData.push([passengerId, prediction]);
            
            // Probabilities CSV: PassengerId, Probability
            probabilitiesData.push([passengerId, probability]);
        }
        
        // Export submission CSV
        exportCSV(['PassengerId', 'Survived'], submissionData, 'submission.csv');
        
        // Export probabilities CSV
        exportCSV(['PassengerId', 'Probability'], probabilitiesData, 'probabilities.csv');
        
        elements.predictStatus.textContent += '\nCSV files exported successfully!';
        elements.predictStatus.className = 'status success';
        
    } catch (error) {
        alert(`Error exporting results: ${error.message}`);
        console.error(error);
    }
}

// CSV export function with proper quoting
function exportCSV(headers, data, filename) {
    // Create CSV content with proper quoting
    let csv = headers.map(h => `"${h}"`).join(',') + '\n';
    
    data.forEach(row => {
        const escapedRow = row.map(val => {
            // Convert to string and escape quotes
            const strVal = String(val);
            return `"${strVal.replace(/"/g, '""')}"`;
        });
        csv += escapedRow.join(',') + '\n';
    });
    
    // Create download link
    const blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
    const link = document.createElement('a');
    
    if (navigator.msSaveBlob) {
        // For IE
        navigator.msSaveBlob(blob, filename);
    } else {
        // For modern browsers
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}

// Save model
async function saveModel() {
    if (!model) {
        alert('Please create and train a model first.');
        return;
    }
    
    try {
        await model.save('downloads://titanic-tfjs-model');
        elements.predictStatus.textContent = 'Model saved successfully! Check your downloads folder.';
        elements.predictStatus.className = 'status success';
    } catch (error) {
        alert(`Error saving model: ${error.message}`);
        console.error(error);
    }
}

// Update threshold and metrics when slider changes
function updateThreshold() {
    const threshold = parseFloat(elements.thresholdSlider.value);
    elements.thresholdValue.textContent = threshold.toFixed(2);
    
    if (validationPredictions && validationLabels) {
        updateMetrics(threshold);
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    console.log('Titanic Binary Classifier initialized. Ready to load data.');
});
