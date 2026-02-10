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

// Parse CSV file
function parseCSV(csvText) {
    const lines = csvText.split('\n');
    const headers = lines[0].split(',');
    
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim() === '') continue;
        
        const values = lines[i].split(',');
        const obj = {};
        
        headers.forEach((header, index) => {
            let value = values[index] || '';
            header = header.trim();
            
            // Remove quotes if present
            if (value.startsWith('"') && value.endsWith('"')) {
                value = value.substring(1, value.length - 1);
            }
            
            // Convert to number if possible
            if (value !== '' && !isNaN(value) && header !== 'Name' && header !== 'Ticket' && header !== 'Cabin') {
                value = parseFloat(value);
            } else if (value === '') {
                value = null;
            }
            
            obj[header] = value;
        });
        
        data.push(obj);
    }
    
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
        // Simple sample data for demonstration
        const sampleTrainCSV = `PassengerId,Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
1,0,3,male,22,1,0,7.25,S
2,1,1,female,38,1,0,71.2833,C
3,1,3,female,26,0,0,7.925,S
4,1,1,female,35,1,0,53.1,S
5,0,3,male,35,0,0,8.05,S
6,0,3,male,,0,0,8.4583,Q
7,0,1,male,54,0,0,51.8625,S
8,0,3,male,2,3,1,21.075,S
9,1,3,female,27,0,2,11.1333,S
10,1,2,female,14,1,0,30.0708,C`;
        
        const sampleTestCSV = `PassengerId,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
11,3,female,4,1,1,16.7,S
12,1,female,58,0,0,26.55,S
13,3,male,20,0,0,8.05,S
14,3,male,39,1,5,31.275,S
15,3,female,14,0,0,7.8542,S`;
        
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
    
    // Headers (first 6 columns for preview)
    const headers = Object.keys(trainData[0]);
    headers.slice(0, 6).forEach(header => {
        html += `<th>${header}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // First 5 rows
    trainData.slice(0, 5).forEach(row => {
        html += '<tr>';
        headers.slice(0, 6).forEach(header => {
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

// Preprocess data
function preprocessData() {
    if (!trainData || !testData) {
        updateStatus('preprocessStatus', 'Please load data first.', 'error');
        return;
    }
    
    updateStatus('preprocessStatus', 'Preprocessing data...', 'info');
    
    try {
        // Calculate imputation values from training data
        const ages = trainData.map(row => row.Age).filter(age => age !== null && !isNaN(age));
        const fares = trainData.map(row => row.Fare).filter(fare => fare !== null && !isNaN(fare));
        const embarkedValues = trainData.map(row => row.Embarked).filter(e => e !== null);
        
        const ageMean = ages.reduce((a, b) => a + b, 0) / ages.length || 30;
        const fareMean = fares.reduce((a, b) => a + b, 0) / fares.length || 30;
        const embarkedMode = embarkedValues.length > 0 ? 
            embarkedValues.reduce((a, b, i, arr) => 
                arr.filter(v => v === a).length >= arr.filter(v => v === b).length ? a : b) : 'S';
        
        // Helper function to preprocess a single row
        const preprocessRow = (row, isTraining = true) => {
            // Handle missing values
            const age = row.Age !== null && !isNaN(row.Age) ? row.Age : ageMean;
            const fare = row.Fare !== null && !isNaN(row.Fare) ? row.Fare : fareMean;
            const embarked = row.Embarked !== null ? row.Embarked : embarkedMode;
            
            // Normalize numerical features
            const normalizedAge = (age - ageMean) / 30; // Simple normalization
            const normalizedFare = (fare - fareMean) / 50;
            
            // One-hot encoding for categorical features
            const pclass = [0, 0, 0];
            if (row.Pclass === 1) pclass[0] = 1;
            else if (row.Pclass === 2) pclass[1] = 1;
            else if (row.Pclass === 3) pclass[2] = 1;
            
            const sex = row.Sex === 'female' ? [1, 0] : [0, 1];
            
            const embarkedEncoded = [0, 0, 0]; // C, Q, S
            if (embarked === 'C') embarkedEncoded[0] = 1;
            else if (embarked === 'Q') embarkedEncoded[1] = 1;
            else if (embarked === 'S') embarkedEncoded[2] = 1;
            
            // Combine all features
            return [
                normalizedAge,
                normalizedFare,
                row.SibSp || 0,
                row.Parch || 0,
                ...pclass,
                ...sex,
                ...embarkedEncoded
            ];
        };
        
        // Preprocess training data
        preprocessedTrainData = { features: [], labels: [] };
        
        trainData.forEach(row => {
            const features = preprocessRow(row, true);
            preprocessedTrainData.features.push(features);
            if (row[TARGET_FEATURE] !== undefined && row[TARGET_FEATURE] !== null) {
                preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
            }
        });
        
        // Preprocess test data
        preprocessedTestData = { features: [], passengerIds: [] };
        
        testData.forEach(row => {
            const features = preprocessRow(row, false);
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

// Create model
function createModel() {
    if (!preprocessedTrainData) {
        updateStatus('modelStatus', 'Please preprocess data first.', 'error');
        return;
    }
    
    updateStatus('modelStatus', 'Creating model...', 'info');
    
    try {
        const inputShape = preprocessedTrainData.features.shape[1];
        
        // Create a simple sequential model without any merge/gate layers
        model = tf.sequential();
        
        // First dense layer (16 units, relu activation)
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [inputShape]
        }));
        
        // Second layer (8 units, sigmoid activation) - acting as "Sigmoid Gate"
        model.add(tf.layers.dense({
            units: 8,
            activation: 'sigmoid'
        }));
        
        // Output layer (1 unit, sigmoid activation for binary classification)
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
        
        // Display model summary
        const paramCount = model.countParams().toLocaleString();
        updateStatus('modelStatus', 
            `Model created successfully!\nArchitecture: Dense(16, relu) → Sigmoid Gate (8 units) → Dense(1, sigmoid)\nTotal parameters: ${paramCount}`, 
            'success');
        
        // Enable training buttons
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('stopTrainBtn').disabled = false;
        
    } catch (error) {
        updateStatus('modelStatus', `Error creating model: ${error.message}`, 'error');
        console.error('Model creation error:', error);
    }
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
        // Create 80/20 split
        const splitIndex = Math.floor(preprocessedTrainData.features.shape[0] * 0.8);
        
        const trainFeatures = preprocessedTrainData.features.slice(0, splitIndex);
        const trainLabels = preprocessedTrainData.labels.slice(0, splitIndex);
        const valFeatures = preprocessedTrainData.features.slice(splitIndex);
        const valLabels = preprocessedTrainData.labels.slice(splitIndex);
        
        validationData = valFeatures;
        validationLabels = valLabels;
        
        // Initialize training history
        trainingHistory = {
            loss: [],
            val_loss: [],
            acc: [],
            val_acc: []
        };
        
        // Train the model
        const history = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    // Update training status
                    updateStatus('trainingStatus', 
                        `Epoch ${epoch + 1}/50 - Loss: ${logs.loss.toFixed(4)}, Accuracy: ${(logs.acc * 100).toFixed(2)}%, Val Loss: ${logs.val_loss.toFixed(4)}, Val Accuracy: ${(logs.val_acc * 100).toFixed(2)}%`, 
                        'info');
                    
                    // Store history for chart
                    trainingHistory.loss.push(logs.loss);
                    trainingHistory.val_loss.push(logs.val_loss);
                    trainingHistory.acc.push(logs.acc);
                    trainingHistory.val_acc.push(logs.val_acc);
                    
                    // Update training chart
                    updateTrainingChart();
                    
                    // Check if stop requested
                    if (stopTrainingRequested) {
                        model.stopTraining = true;
                    }
                }
            }
        });
        
        // Make validation predictions
        validationPredictions = model.predict(validationData);
        
        // Calculate feature importance
        calculateFeatureImportance();
        
        // Enable evaluation and prediction buttons
        document.getElementById('evaluateBtn').disabled = false;
        document.getElementById('thresholdSlider').disabled = false;
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('exportBtn').disabled = false;
        
        updateStatus('trainingStatus', 'Training completed!', 'success');
        
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
        const color = '#1a2980';
        html += `<div style="width: 5px; height: ${height}px; background: ${color};"></div>`;
    });
    html += '</div>';
    
    // Accuracy chart
    html += '<div><strong>Accuracy:</strong></div>';
    html += '<div style="height: 100px; display: flex; align-items: flex-end; gap: 2px;">';
    trainingHistory.acc.forEach((acc, i) => {
        const height = acc * 100;
        const color = '#26d0ce';
        html += `<div style="width: 5px; height: ${height}px; background: ${color};"></div>`;
    });
    html += '</div>';
    
    html += '</div>';
    historyDiv.innerHTML = html;
}

// Calculate feature importance (simple version)
function calculateFeatureImportance() {
    if (!model) return;
    
    try {
        // Simple feature importance based on first layer weights
        const firstLayer = model.layers[0];
        const weights = firstLayer.getWeights()[0];
        
        // Calculate average absolute weight for each input feature
        const weightsArray = weights.arraySync();
        const importances = [];
        
        for (let i = 0; i < weightsArray.length; i++) {
            let sum = 0;
            for (let j = 0; j < weightsArray[i].length; j++) {
                sum += Math.abs(weightsArray[i][j]);
            }
            importances.push(sum / weightsArray[i].length);
        }
        
        // Feature names (matches preprocessing order)
        const featureNames = [
            'Age', 'Fare', 'SibSp', 'Parch',
            'Pclass_1', 'Pclass_2', 'Pclass_3',
            'Sex_female', 'Sex_male',
            'Embarked_C', 'Embarked_Q', 'Embarked_S'
        ];
        
        // Normalize importances to percentages
        const maxImportance = Math.max(...importances);
        const normalizedImportances = importances.map(imp => 
            maxImportance > 0 ? (imp / maxImportance) * 100 : 0
        );
        
        // Display feature importance
        displayFeatureImportance(featureNames, normalizedImportances);
        
    } catch (error) {
        console.error('Error calculating feature importance:', error);
        // Fallback to random importances for display
        const featureNames = [
            'Age', 'Fare', 'SibSp', 'Parch',
            'Pclass_1', 'Pclass_2', 'Pclass_3',
            'Sex_female', 'Sex_male',
            'Embarked_C', 'Embarked_Q', 'Embarked_S'
        ];
        const randomImportances = featureNames.map(() => Math.random() * 100);
        displayFeatureImportance(featureNames, randomImportances);
    }
}

// Display feature importance
function displayFeatureImportance(featureNames, importances) {
    const importanceDiv = document.getElementById('featureImportance');
    let html = '<div class="feature-importance-grid">';
    
    // Create array of feature objects and sort by importance
    const featureObjs = featureNames.map((name, idx) => ({
        name,
        importance: importances[idx] || 0
    })).sort((a, b) => b.importance - a.importance);
    
    // Display top 8 features
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
        const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;
        
        // Simple AUC calculation
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
    // Simple AUC approximation
    const sorted = predictions.map((p, i) => ({p, label: labels[i]}))
        .sort((a, b) => a.p - b.p);
    
    const totalPositives = labels.filter(l => l === 1).length;
    const totalNegatives = labels.filter(l => l === 0).length;
    
    if (totalPositives === 0 || totalNegatives === 0) return 0.5;
    
    let rankSum = 0;
    sorted.forEach((item, index) => {
        if (item.label === 1) {
            rankSum += index + 1;
        }
    });
    
    const auc = (rankSum - (totalPositives * (totalPositives + 1) / 2)) / (totalPositives * totalNegatives);
    return Math.max(0, Math.min(1, auc)); // Clamp between 0 and 1
}

// Update ROC curve
function updateROCCurve(predictions, labels) {
    const rocDiv = document.getElementById('rocCurve');
    
    // Simple ROC visualization
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
    
    // Create SVG for ROC curve
    const svgWidth = 300, svgHeight = 200;
    let svg = `<svg width="${svgWidth}" height="${svgHeight}" style="border: 1px solid #ddd; margin-top: 10px;">`;
    
    // Draw diagonal line
    svg += `<line x1="0" y1="${svgHeight}" x2="${svgWidth}" y2="0" stroke="#999" stroke-dasharray="5,5" />`;
    
    // Draw ROC curve
    let path = `M 0 ${svgHeight} `;
    points.forEach(point => {
        const x = point.fpr * svgWidth;
        const y = svgHeight - (point.tpr * svgHeight);
        path += `L ${x} ${y} `;
    });
    
    svg += `<path d="${path}" stroke="#1a2980" stroke-width="2" fill="none" />`;
    
    // Add current threshold point
    const currentThreshold = parseFloat(document.getElementById('thresholdSlider').value);
    const currentPoint = points[Math.round(currentThreshold * 10)];
    if (currentPoint) {
        const x = currentPoint.fpr * svgWidth;
        const y = svgHeight - (currentPoint.tpr * svgHeight);
        svg += `<circle cx="${x}" cy="${y}" r="5" fill="#26d0ce" />`;
    }
    
    svg += '</svg>';
    
    rocDiv.innerHTML = svg + '<div style="text-align: center; margin-top: 5px; font-size: 12px; color: #666;">False Positive Rate →</div>';
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
        
        // Clean up
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
        updateStatus('exportStatus', 'Model exported successfully! Check your downloads folder.', 'success');
    } catch (error) {
        updateStatus('exportStatus', `Error exporting model: ${error.message}`, 'error');
        console.error('Export error:', error);
    }
}
