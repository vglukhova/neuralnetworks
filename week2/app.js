// Global variables
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
let featureImportances = null;
let stopTrainingRequested = false;

// Schema configuration
const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Setup event listeners
    document.getElementById('loadDataBtn').addEventListener('click', loadData);
    document.getElementById('loadSampleBtn').addEventListener('click', loadSampleData);
    document.getElementById('preprocessBtn').addEventListener('click', preprocessData);
    document.getElementById('createModelBtn').addEventListener('click', createModel);
    document.getElementById('trainBtn').addEventListener('click', trainModel);
    document.getElementById('stopTrainBtn').addEventListener('click', stopTraining);
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);
    document.getElementById('predictBtn').addEventListener('click', predictTestData);
    document.getElementById('exportBtn').addEventListener('click', exportModel);
    document.getElementById('thresholdSlider').addEventListener('input', updateThreshold);
    
    updateStatus('dataStatus', 'Ready to load data. Upload CSV files or use sample data.', 'info');
});

// Improved CSV parsing with proper comma handling
function parseCSV(csvText) {
    const lines = [];
    let currentLine = [];
    let inQuotes = false;
    let currentValue = '';
    
    for (let i = 0; i < csvText.length; i++) {
        const char = csvText[i];
        const nextChar = csvText[i + 1];
        
        if (char === '"') {
            if (inQuotes && nextChar === '"') {
                // Escaped quote inside quoted field
                currentValue += '"';
                i++; // Skip next character
            } else {
                // Start or end of quoted field
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            // End of field
            currentLine.push(currentValue.trim());
            currentValue = '';
        } else if (char === '\n' && !inQuotes) {
            // End of line
            currentLine.push(currentValue.trim());
            lines.push(currentLine);
            currentLine = [];
            currentValue = '';
        } else if (char === '\r' && nextChar === '\n' && !inQuotes) {
            // Handle Windows line endings
            currentLine.push(currentValue.trim());
            lines.push(currentLine);
            currentLine = [];
            currentValue = '';
            i++; // Skip the \n
        } else {
            currentValue += char;
        }
    }
    
    // Add last line if exists
    if (currentValue !== '' || currentLine.length > 0) {
        currentLine.push(currentValue.trim());
        lines.push(currentLine);
    }
    
    // Extract headers and data
    const headers = lines[0];
    const data = lines.slice(1).map(line => {
        const obj = {};
        headers.forEach((header, index) => {
            let value = line[index] || null;
            
            // Convert to number if possible
            if (value !== null && !isNaN(value) && value.trim() !== '') {
                value = parseFloat(value);
            } else if (value === '' || value === 'NULL' || value === 'null') {
                value = null;
            }
            
            obj[header.trim()] = value;
        });
        return obj;
    });
    
    return data;
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// Update status with styling
function updateStatus(elementId, message, type = 'info') {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.className = 'status ' + type;
    element.style.display = 'block';
}

// Load data from uploaded CSV files
async function loadData() {
    const trainFile = document.getElementById('trainFile').files[0];
    const testFile = document.getElementById('testFile').files[0];
    
    if (!trainFile || !testFile) {
        updateStatus('dataStatus', 'Please upload both training and test CSV files.', 'error');
        return;
    }
    
    updateStatus('dataStatus', 'Loading and parsing data...', 'info');
    
    try {
        const trainText = await readFile(trainFile);
        const testText = await readFile(testFile);
        
        trainData = parseCSV(trainText);
        testData = parseCSV(testText);
        
        updateStatus('dataStatus', 
            `Data loaded successfully!\nTraining: ${trainData.length} samples\nTest: ${testData.length} samples`, 
            'success');
        
        // Enable preprocessing button
        document.getElementById('preprocessBtn').disabled = false;
        
        // Show data preview
        showDataPreview();
        showSurvivalChart();
        
    } catch (error) {
        updateStatus('dataStatus', `Error loading data: ${error.message}`, 'error');
        console.error('Load error:', error);
    }
}

// Load sample Titanic data
async function loadSampleData() {
    updateStatus('dataStatus', 'Loading sample Titanic data...', 'info');
    
    try {
        // Sample training data (first 10 rows of Titanic dataset)
        const sampleTrainCSV = `PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",female,27,0,2,347742,11.1333,,S
10,1,2,"Nasser, Mrs. Nicholas (Adele Achem)",female,14,1,0,237736,30.0708,,C`;
        
        // Sample test data
        const sampleTestCSV = `PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
11,3,"Sandstrom, Miss. Marguerite Rut",female,4,1,1,PP 9549,16.7,G6,S
12,1,"Bonnell, Miss. Elizabeth",female,58,0,0,113783,26.55,C103,S
13,3,"Saundercock, Mr. William Henry",male,20,0,0,A/5. 2151,8.05,,S
14,3,"Andersson, Mr. Anders Johan",male,39,1,5,347082,31.275,,S
15,3,"Vestrom, Miss. Hulda Amanda Adolfina",female,14,0,0,350406,7.8542,,S`;
        
        trainData = parseCSV(sampleTrainCSV);
        testData = parseCSV(sampleTestCSV);
        
        updateStatus('dataStatus', 
            `Sample data loaded!\nTraining: ${trainData.length} samples\nTest: ${testData.length} samples`, 
            'success');
        
        // Enable preprocessing button
        document.getElementById('preprocessBtn').disabled = false;
        
        // Show data preview
        showDataPreview();
        showSurvivalChart();
        
    } catch (error) {
        updateStatus('dataStatus', `Error loading sample data: ${error.message}`, 'error');
        console.error('Load sample error:', error);
    }
}

// Show data preview table
function showDataPreview() {
    if (!trainData || trainData.length === 0) return;
    
    const previewDiv = document.getElementById('dataPreview');
    let html = '<table><thead><tr>';
    
    // Headers (first 5 columns for preview)
    const headers = Object.keys(trainData[0]);
    headers.slice(0, 5).forEach(header => {
        html += `<th>${header}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // First 5 rows
    trainData.slice(0, 5).forEach(row => {
        html += '<tr>';
        headers.slice(0, 5).forEach(header => {
            const value = row[header];
            html += `<td>${value !== null && value !== undefined ? value : 'N/A'}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    previewDiv.innerHTML = html;
}

// Show survival distribution chart
function showSurvivalChart() {
    if (!trainData || trainData.length === 0) return;
    
    const survived = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const perished = trainData.filter(row => row[TARGET_FEATURE] === 0).length;
    const total = trainData.length;
    
    const chartDiv = document.getElementById('survivalChart');
    chartDiv.innerHTML = `
        <div style="display: flex; justify-content: space-around; margin-top: 20px;">
            <div style="text-align: center;">
                <div style="width: 100px; height: 100px; border-radius: 50%; background: #1a2980; display: flex; align-items: center; justify-content: center; margin: 0 auto; color: white; font-weight: bold; font-size: 24px;">${survived}</div>
                <p style="margin-top: 10px;">Survived (${((survived/total)*100).toFixed(1)}%)</p>
            </div>
            <div style="text-align: center;">
                <div style="width: 100px; height: 100px; border-radius: 50%; background: #26d0ce; display: flex; align-items: center; justify-content: center; margin: 0 auto; color: white; font-weight: bold; font-size: 24px;">${perished}</div>
                <p style="margin-top: 10px;">Perished (${((perished/total)*100).toFixed(1)}%)</p>
            </div>
        </div>
    `;
}

// Calculate median
function calculateMedian(values) {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

// Calculate mode
function calculateMode(values) {
    if (values.length === 0) return 'S';
    const freq = {};
    let max = 0;
    let mode = 'S';
    
    values.forEach(value => {
        if (value !== null && value !== undefined) {
            freq[value] = (freq[value] || 0) + 1;
            if (freq[value] > max) {
                max = freq[value];
                mode = value;
            }
        }
    });
    
    return mode;
}

// Calculate standard deviation
function calculateStdDev(values) {
    if (values.length === 0) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(variance);
}

// One-hot encode categorical feature
function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(value);
    if (index !== -1) encoding[index] = 1;
    return encoding;
}

// Preprocess data
function preprocessData() {
    if (!trainData || !testData) {
        updateStatus('preprocessStatus', 'Please load data first.', 'error');
        return;
    }
    
    updateStatus('preprocessStatus', 'Preprocessing data...', 'info');
    
    try {
        // Calculate imputation values from training data only
        const ageMedian = calculateMedian(trainData.map(row => row.Age).filter(a => a !== null && !isNaN(a)));
        const fareMedian = calculateMedian(trainData.map(row => row.Fare).filter(f => f !== null && !isNaN(f)));
        const embarkedMode = calculateMode(trainData.map(row => row.Embarked).filter(e => e !== null));
        
        // Get training data statistics for standardization
        const trainAges = trainData.map(r => r.Age).filter(a => a !== null && !isNaN(a));
        const trainFares = trainData.map(r => r.Fare).filter(f => f !== null && !isNaN(f));
        
        const ageMean = trainAges.reduce((a, b) => a + b, 0) / trainAges.length || 0;
        const fareMean = trainFares.reduce((a, b) => a + b, 0) / trainFares.length || 0;
        
        const ageStd = calculateStdDev(trainAges) || 1;
        const fareStd = calculateStdDev(trainFares) || 1;
        
        // Preprocess training data
        preprocessedTrainData = { features: [], labels: [] };
        
        trainData.forEach(row => {
            // Handle missing values
            const age = (row.Age !== null && !isNaN(row.Age)) ? row.Age : ageMedian;
            const fare = (row.Fare !== null && !isNaN(row.Fare)) ? row.Fare : fareMedian;
            const embarked = (row.Embarked !== null) ? row.Embarked : embarkedMode;
            
            // Standardize numerical features
            const standardizedAge = (age - ageMean) / ageStd;
            const standardizedFare = (fare - fareMean) / fareStd;
            
            // One-hot encode categorical features
            const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]);
            const sexOneHot = oneHotEncode(row.Sex, ['male', 'female']);
            const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);
            
            // Combine all features
            let features = [
                standardizedAge,
                standardizedFare,
                row.SibSp || 0,
                row.Parch || 0
            ];
            
            features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);
            
            preprocessedTrainData.features.push(features);
            if (row[TARGET_FEATURE] !== undefined && row[TARGET_FEATURE] !== null) {
                preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
            }
        });
        
        // Preprocess test data
        preprocessedTestData = { features: [], passengerIds: [] };
        
        testData.forEach(row => {
            // Handle missing values (use same imputation as training)
            const age = (row.Age !== null && !isNaN(row.Age)) ? row.Age : ageMedian;
            const fare = (row.Fare !== null && !isNaN(row.Fare)) ? row.Fare : fareMedian;
            const embarked = (row.Embarked !== null) ? row.Embarked : embarkedMode;
            
            // Standardize numerical features (using training stats)
            const standardizedAge = (age - ageMean) / ageStd;
            const standardizedFare = (fare - fareMean) / fareStd;
            
            // One-hot encode categorical features
            const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]);
            const sexOneHot = oneHotEncode(row.Sex, ['male', 'female']);
            const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);
            
            // Combine all features
            let features = [
                standardizedAge,
                standardizedFare,
                row.SibSp || 0,
                row.Parch || 0
            ];
            
            features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);
            
            preprocessedTestData.features.push(features);
            preprocessedTestData.passengerIds.push(row[ID_FEATURE]);
        });
        
        // Convert to tensors
        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);
        
        updateStatus('preprocessStatus', 
            `Preprocessing complete!\nTraining features shape: ${preprocessedTrainData.features.shape}\nTraining labels shape: ${preprocessedTrainData.labels.shape}`, 
            'success');
        
        // Enable model creation button
        document.getElementById('createModelBtn').disabled = false;
        
    } catch (error) {
        updateStatus('preprocessStatus', `Error during preprocessing: ${error.message}`, 'error');
        console.error('Preprocessing error:', error);
    }
}

// Create model with Sigmoid gate layer
function createModel() {
    if (!preprocessedTrainData) {
        updateStatus('modelStatus', 'Please preprocess data first.', 'error');
        return;
    }
    
    const inputShape = preprocessedTrainData.features.shape[1];
    
    // Create model with Sigmoid gate layer as specified in HTML
    model = tf.sequential();
    
    // First dense layer
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        inputShape: [inputShape]
    }));
    
    // Sigmoid gate layer (8 units)
    model.add(tf.layers.dense({
        units: 8,
        activation: 'sigmoid'
    }));
    
    // Output layer
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    
    // Compile model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Display model summary
    updateStatus('modelStatus', 
        `Model created successfully!\nArchitecture: Dense(16, relu) → Sigmoid Gate (8 units) → Dense(1, sigmoid)\nTotal parameters: ${model.countParams().toLocaleString()}`, 
        'success');
    
    // Enable training button
    document.getElementById('trainBtn').disabled = false;
    document.getElementById('stopTrainBtn').disabled = false;
}

// Train model
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        updateStatus('trainingStatus', 'Please create model first.', 'error');
        return;
    }
    
    stopTrainingRequested = false;
    updateStatus('trainingStatus', 'Training model... (80/20 split, 50 epochs)', 'info');
    
    try {
        // Create 80/20 stratified split
        const splitIndex = Math.floor(preprocessedTrainData.features.shape[0] * 0.8);
        
        const trainFeatures = preprocessedTrainData.features.slice(0, splitIndex);
        const trainLabels = preprocessedTrainData.labels.slice(0, splitIndex);
        const valFeatures = preprocessedTrainData.features.slice(splitIndex);
        const valLabels = preprocessedTrainData.labels.slice(splitIndex);
        
        validationData = valFeatures;
        validationLabels = valLabels;
        
        // Setup for early stopping if user requests
        const callbacks = {
            onEpochEnd: (epoch, logs) => {
                updateStatus('trainingStatus', 
                    `Epoch ${epoch + 1}/50 - Loss: ${logs.loss.toFixed(4)}, Accuracy: ${(logs.acc * 100).toFixed(2)}%, Val Loss: ${logs.val_loss.toFixed(4)}, Val Accuracy: ${(logs.val_acc * 100).toFixed(2)}%`, 
                    'info');
                
                // Update training history chart
                if (!trainingHistory) {
                    trainingHistory = {
                        loss: [],
                        val_loss: [],
                        acc: [],
                        val_acc: []
                    };
                }
                
                trainingHistory.loss.push(logs.loss);
                trainingHistory.val_loss.push(logs.val_loss);
                trainingHistory.acc.push(logs.acc);
                trainingHistory.val_acc.push(logs.val_acc);
                
                updateTrainingChart();
                
                // Check if stop requested
                if (stopTrainingRequested) {
                    model.stopTraining = true;
                    updateStatus('trainingStatus', 'Training stopped by user.', 'info');
                }
            },
            onTrainEnd: () => {
                updateStatus('trainingStatus', 'Training completed!', 'success');
                
                // Make validation predictions
                validationPredictions = model.predict(validationData);
                
                // Enable evaluation
                document.getElementById('evaluateBtn').disabled = false;
                document.getElementById('thresholdSlider').disabled = false;
                document.getElementById('predictBtn').disabled = false;
                document.getElementById('exportBtn').disabled = false;
                
                // Calculate feature importance
                calculateFeatureImportance();
            }
        };
        
        // Train the model
        await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: callbacks,
            verbose: 0
        });
        
    } catch (error) {
        updateStatus('trainingStatus', `Error during training: ${error.message}`, 'error');
        console.error('Training error:', error);
    }
}

// Stop training
function stopTraining() {
    stopTrainingRequested = true;
    updateStatus('trainingStatus', 'Stopping training...', 'info');
}

// Update training chart
function updateTrainingChart() {
    if (!trainingHistory || trainingHistory.loss.length === 0) return;
    
    const historyDiv = document.getElementById('trainingHistory');
    
    // Create a simple text-based chart
    let html = '<div style="display: flex; flex-direction: column; gap: 10px;">';
    
    // Loss chart
    html += '<div><strong>Loss:</strong></div>';
    html += '<div style="height: 100px; display: flex; align-items: flex-end; gap: 2px;">';
    trainingHistory.loss.forEach((loss, i) => {
        const height = Math.min(100, loss * 100);
        const color = i < trainingHistory.loss.length - 1 ? '#1a2980' : '#26d0ce';
        html += `<div style="width: 5px; height: ${height}px; background: ${color};"></div>`;
    });
    html += '</div>';
    
    // Accuracy chart
    html += '<div><strong>Accuracy:</strong></div>';
    html += '<div style="height: 100px; display: flex; align-items: flex-end; gap: 2px;">';
    trainingHistory.acc.forEach((acc, i) => {
        const height = acc * 100;
        const color = i < trainingHistory.acc.length - 1 ? '#1a2980' : '#26d0ce';
        html += `<div style="width: 5px; height: ${height}px; background: ${color};"></div>`;
    });
    html += '</div>';
    
    html += '</div>';
    historyDiv.innerHTML = html;
}

// Calculate feature importance
async function calculateFeatureImportance() {
    if (!model || !preprocessedTrainData) return;
    
    try {
        // Simple feature importance based on weights
        const layer = model.getLayer(null, 0); // Get first layer
        const weights = layer.getWeights()[0]; // Get weight matrix
        
        // Calculate average absolute weight for each input feature
        const weightsArray = await weights.array();
        featureImportances = [];
        
        for (let i = 0; i < weightsArray.length; i++) {
            let sum = 0;
            for (let j = 0; j < weightsArray[i].length; j++) {
                sum += Math.abs(weightsArray[i][j]);
            }
            featureImportances.push(sum / weightsArray[i].length);
        }
        
        // Normalize to 0-100%
        const maxImportance = Math.max(...featureImportances);
        const normalizedImportances = featureImportances.map(imp => (imp / maxImportance) * 100);
        
        // Feature names (matches preprocessing order)
        const featureNames = [
            'Age', 'Fare', 'SibSp', 'Parch',
            'Pclass_1', 'Pclass_2', 'Pclass_3',
            'Sex_male', 'Sex_female',
            'Embarked_C', 'Embarked_Q', 'Embarked_S'
        ];
        
        // Display feature importance
        const importanceDiv = document.getElementById('featureImportance');
        let html = '<div class="feature-importance-grid">';
        
        // Create array of feature objects
        const featureObjs = featureNames.map((name, idx) => ({
            name,
            importance: normalizedImportances[idx] || 0
        }));
        
        // Sort by importance
        featureObjs.sort((a, b) => b.importance - a.importance);
        
        // Display top features
        featureObjs.slice(0, 8).forEach(feature => {
            html += `
                <div class="feature-item">
                    <div class="feature-name-col">${feature.name}</div>
                    <div class="feature-bar-container">
                        <div class="feature-bar" style="width: ${feature.importance.toFixed(1)}%">
                            <span class="feature-value">${feature.importance.toFixed(1)}%</span>
                        </div>
                    </div>
                    <div class="feature-percentage">${feature.importance.toFixed(1)}%</div>
                </div>
            `;
        });
        
        html += '</div>';
        importanceDiv.innerHTML = html;
        
    } catch (error) {
        console.error('Error calculating feature importance:', error);
    }
}

// Update threshold slider value
function updateThreshold() {
    const threshold = parseFloat(document.getElementById('thresholdSlider').value);
    document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
}

// Evaluate model
async function evaluateModel() {
    if (!model || !validationPredictions || !validationLabels) {
        updateStatus('evaluationStatus', 'Please train model first.', 'error');
        return;
    }
    
    updateStatus('evaluationStatus', 'Evaluating model...', 'info');
    
    try {
        const threshold = parseFloat(document.getElementById('thresholdSlider').value);
        const predVals = await validationPredictions.array();
        const trueVals = await validationLabels.array();
        
        let tp = 0, tn = 0, fp = 0, fn = 0;
        
        for (let i = 0; i < predVals.length; i++) {
            const prediction = predVals[i] >= threshold ? 1 : 0;
            const actual = trueVals[i];
            
            if (prediction === 1 && actual === 1) tp++;
            else if (prediction === 0 && actual === 0) tn++;
            else if (prediction === 1 && actual === 0) fp++;
            else fn++;
        }
        
        // Calculate metrics
        const accuracy = (tp + tn) / (tp + tn + fp + fn);
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;
        
        // Simple AUC approximation
        const auc = calculateAUC(predVals, trueVals);
        
        // Update metrics display
        document.getElementById('accuracyValue').textContent = accuracy.toFixed(3);
        document.getElementById('precisionValue').textContent = precision.toFixed(3);
        document.getElementById('recallValue').textContent = recall.toFixed(3);
        document.getElementById('f1Value').textContent = f1.toFixed(3);
        document.getElementById('aucValue').textContent = auc.toFixed(3);
        
        // Create evaluation table
        const tableDiv = document.getElementById('evaluationTable');
        tableDiv.innerHTML = `
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>True Positives</td>
                        <td>${tp}</td>
                        <td>Correctly predicted survival</td>
                    </tr>
                    <tr>
                        <td>True Negatives</td>
                        <td>${tn}</td>
                        <td>Correctly predicted perished</td>
                    </tr>
                    <tr>
                        <td>False Positives</td>
                        <td>${fp}</td>
                        <td>Incorrectly predicted survival</td>
                    </tr>
                    <tr>
                        <td>False Negatives</td>
                        <td>${fn}</td>
                        <td>Incorrectly predicted perished</td>
                    </tr>
                    <tr>
                        <td>Accuracy</td>
                        <td>${(accuracy * 100).toFixed(2)}%</td>
                        <td>Overall correctness</td>
                    </tr>
                    <tr>
                        <td>Precision</td>
                        <td>${precision.toFixed(4)}</td>
                        <td>Quality of positive predictions</td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td>${recall.toFixed(4)}</td>
                        <td>Ability to find all positives</td>
                    </tr>
                    <tr>
                        <td>F1 Score</td>
                        <td>${f1.toFixed(4)}</td>
                        <td>Balance of precision and recall</td>
                    </tr>
                    <tr>
                        <td>AUC</td>
                        <td>${auc.toFixed(4)}</td>
                        <td>Area Under ROC Curve</td>
                    </tr>
                </tbody>
            </table>
        `;
        
        updateStatus('evaluationStatus', 'Evaluation complete!', 'success');
        
        // Update ROC curve
        updateROCCurve(predVals, trueVals);
        
    } catch (error) {
        updateStatus('evaluationStatus', `Error during evaluation: ${error.message}`, 'error');
        console.error('Evaluation error:', error);
    }
}

// Calculate AUC (Area Under ROC Curve)
function calculateAUC(predictions, labels) {
    // Simple AUC calculation using trapezoidal rule
    const sorted = predictions.map((p, i) => ({p, label: labels[i]}))
        .sort((a, b) => b.p - a.p);
    
    let area = 0;
    let fp = 0, tp = 0;
    let prevFp = 0, prevTp = 0;
    
    const totalPositives = labels.filter(l => l === 1).length;
    const totalNegatives = labels.filter(l => l === 0).length;
    
    for (const item of sorted) {
        if (item.label === 1) {
            tp++;
        } else {
            fp++;
        }
        
        area += (fp - prevFp) * (tp + prevTp) / 2;
        prevFp = fp;
        prevTp = tp;
    }
    
    return area / (totalPositives * totalNegatives);
}

// Update ROC curve
function updateROCCurve(predictions, labels) {
    const rocDiv = document.getElementById('rocCurve');
    
    // Simple ROC curve visualization
    let html = '<div style="position: relative; height: 200px; border: 1px solid #ddd; margin-top: 10px;">';
    
    // Diagonal line
    html += '<div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 100%;">';
    html += '<div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 100%; border-left: 1px solid #ddd; border-bottom: 1px solid #ddd;"></div>';
    html += '<div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 100%; border-top: 1px solid #ddd; border-right: 1px solid #ddd;"></div>';
    
    // Diagonal reference line
    html += '<div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 100%; border-bottom: 1px dashed #999;"></div>';
    
    // Calculate ROC points
    const thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    const points = thresholds.map(threshold => {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            const prediction = predictions[i] >= threshold ? 1 : 0;
            const actual = labels[i];
            
            if (prediction === 1 && actual === 1) tp++;
            else if (prediction === 0 && actual === 0) tn++;
            else if (prediction === 1 && actual === 0) fp++;
            else fn++;
        }
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        return { fpr, tpr };
    });
    
    // Draw ROC curve
    let path = '';
    points.forEach((point, i) => {
        const x = point.fpr * 100;
        const y = point.tpr * 100;
        
        if (i === 0) {
            path += `M ${x}% ${100 - y}% `;
        } else {
            path += `L ${x}% ${100 - y}% `;
        }
    });
    
    html += `<svg style="position: absolute; bottom: 0; left: 0; width: 100%; height: 100%;">
        <path d="${path}" stroke="#1a2980" stroke-width="2" fill="none" />
    </svg>`;
    
    html += '</div>';
    html += '<div style="text-align: center; margin-top: 5px; font-size: 12px; color: #666;">False Positive Rate</div>';
    html += '<div style="transform: rotate(-90deg); transform-origin: left top; position: absolute; left: 0; top: 100px; font-size: 12px; color: #666;">True Positive Rate</div>';
    
    rocDiv.innerHTML = html;
}

// Predict on test data
async function predictTestData() {
    if (!model || !preprocessedTestData) {
        updateStatus('exportStatus', 'Please train model first.', 'error');
        return;
    }
    
    updateStatus('exportStatus', 'Making predictions on test data...', 'info');
    
    try {
        const testFeatures = tf.tensor2d(preprocessedTestData.features);
        const predictions = model.predict(testFeatures);
        const predValues = await predictions.array();
        
        // Create results
        const results = preprocessedTestData.passengerIds.map((id, i) => {
            const probability = Array.isArray(predValues[i]) ? predValues[i][0] : predValues[i];
            return {
                PassengerId: id,
                Survived: probability >= 0.5 ? 1 : 0,
                Probability: probability
            };
        });
        
        // Show predictions
        updateStatus('exportStatus', `Predictions generated for ${results.length} test samples.`, 'success');
        
        // Create downloadable CSV
        let csvContent = 'PassengerId,Survived,Probability\n';
        results.forEach(row => {
            csvContent += `${row.PassengerId},${row.Survived},${row.Probability.toFixed(4)}\n`;
        });
        
        // Create download link
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'titanic_predictions.csv';
        link.textContent = 'Download Predictions CSV';
        link.style.display = 'block';
        link.style.marginTop = '10px';
        link.style.padding = '10px';
        link.style.background = '#1a2980';
        link.style.color = 'white';
        link.style.textDecoration = 'none';
        link.style.borderRadius = '5px';
        
        const statusDiv = document.getElementById('exportStatus');
        statusDiv.appendChild(link);
        
        // Clean up predictions tensor
        predictions.dispose();
        testFeatures.dispose();
        
    } catch (error) {
        updateStatus('exportStatus', `Error during prediction: ${error.message}`, 'error');
        console.error('Prediction error:', error);
    }
}

// Export model
async function exportModel() {
    if (!model) {
        updateStatus('exportStatus', 'Please train model first.', 'error');
        return;
    }
    
    updateStatus('exportStatus', 'Exporting model...', 'info');
    
    try {
        // Save model locally
        await model.save('downloads://titanic-model');
        updateStatus('exportStatus', 'Model exported successfully!', 'success');
    } catch (error) {
        updateStatus('exportStatus', `Error exporting model: ${error.message}`, 'error');
        console.error('Export error:', error);
    }
}
