// ============================================
// Titanic Survival Classifier with Sigmoid Gates
// TensorFlow.js - Browser-based Binary Classifier
// ============================================

// Global variables
let trainData = null;
let testData = null;
let processedTrainData = null;
let processedTestData = null;
let model = null;
let gateWeights = null;
let isTraining = false;
let trainingHistory = null;
let predictions = null;
let featureImportance = null;
let evaluationResults = null;

// Feature columns (excluding PassengerId and Survived)
const FEATURE_COLS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];
const TARGET_COL = 'Survived';
const ID_COL = 'PassengerId';

// DOM Elements
const elements = {
    trainFile: document.getElementById('trainFile'),
    testFile: document.getElementById('testFile'),
    loadDataBtn: document.getElementById('loadDataBtn'),
    inspectBtn: document.getElementById('inspectBtn'),
    preprocessBtn: document.getElementById('preprocessBtn'),
    createModelBtn: document.getElementById('createModelBtn'),
    trainBtn: document.getElementById('trainBtn'),
    stopTrainBtn: document.getElementById('stopTrainBtn'),
    evaluateBtn: document.getElementById('evaluateBtn'),
    predictBtn: document.getElementById('predictBtn'),
    exportModelBtn: document.getElementById('exportModelBtn'),
    exportPredBtn: document.getElementById('exportPredBtn'),
    exportImportanceBtn: document.getElementById('exportImportanceBtn'),
    showImportanceBtn: document.getElementById('showImportanceBtn'),
    useGateToggle: document.getElementById('useGateToggle'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue'),
    loadStatus: document.getElementById('loadStatus'),
    preprocessStatus: document.getElementById('preprocessStatus'),
    modelStatus: document.getElementById('modelStatus'),
    trainingStatus: document.getElementById('trainingStatus'),
    predictionStatus: document.getElementById('predictionStatus'),
    exportStatus: document.getElementById('exportStatus'),
    evaluationTable: document.getElementById('evaluation-table'),
    dataPreview: document.getElementById('data-preview'),
    inspectionCharts: document.getElementById('inspection-charts'),
    featureImportanceDiv: document.getElementById('feature-importance')
};

// ============================================
// 1. CSV Loading Functions
// ============================================

/**
 * Load CSV file with proper comma escaping
 */
async function loadCSV(file, isTest = false) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = async function(e) {
            try {
                const csvText = e.target.result;
                
                // Handle CSV parsing with proper escaping
                const lines = csvText.split('\n');
                const headers = lines[0].split(',');
                
                // Validate required columns
                const requiredCols = isTest ? 
                    ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'] :
                    ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'];
                
                for (const col of requiredCols) {
                    if (!headers.includes(col)) {
                        throw new Error(`Missing column: ${col}`);
                    }
                }
                
                // Parse CSV using tf.data.csv with proper configuration
                const columnConfigs = {};
                requiredCols.forEach(col => {
                    columnConfigs[col] = {
                        required: true,
                        dtype: col === 'Survived' || col === 'Pclass' || col === 'Sex' || col === 'Embarked' ? 
                               'string' : 'float32'
                    };
                });
                
                const dataset = tf.data.csv(csvText, {
                    columnConfigs,
                    configuredColumnsOnly: false,
                    hasHeader: true,
                    delimiter: ','
                });
                
                // Convert to array
                const dataArray = await dataset.toArray();
                
                // Extract features and labels
                const data = {
                    raw: dataArray,
                    features: dataArray.map(row => {
                        const features = {};
                        FEATURE_COLS.forEach(col => {
                            features[col] = row[col];
                        });
                        return features;
                    }),
                    ids: dataArray.map(row => row[ID_COL])
                };
                
                if (!isTest) {
                    data.labels = dataArray.map(row => row[TARGET_COL]);
                }
                
                showStatus('success', `${isTest ? 'Test' : 'Train'} data loaded: ${dataArray.length} rows`);
                resolve(data);
                
            } catch (error) {
                showStatus('error', `CSV parsing error: ${error.message}`);
                reject(error);
            }
        };
        
        reader.onerror = () => {
            showStatus('error', 'Error reading file');
            reject(new Error('File read error'));
        };
        
        reader.readAsText(file);
    });
}

// ============================================
// 2. Data Inspection & Preview
// ============================================

/**
 * Show data preview table
 */
function showDataPreview(data, title = 'Train Data Preview') {
    if (!data || !data.raw || data.raw.length === 0) return;
    
    const previewRows = data.raw.slice(0, 10);
    let html = `<h3>${title} (10 rows)</h3>`;
    html += '<table><thead><tr>';
    
    // Headers
    const headers = Object.keys(previewRows[0]);
    headers.forEach(header => {
        html += `<th>${header}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // Rows
    previewRows.forEach(row => {
        html += '<tr>';
        headers.forEach(header => {
            let value = row[header];
            // Truncate long values
            if (value && value.toString().length > 20) {
                value = value.toString().substring(0, 20) + '...';
            }
            html += `<td>${value !== undefined ? value : ''}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    elements.dataPreview.innerHTML = html;
}

/**
 * Inspect data statistics
 */
async function inspectData() {
    if (!trainData) return;
    
    try {
        // Calculate basic statistics
        const stats = {
            totalRows: trainData.raw.length,
            features: FEATURE_COLS,
            survivalRate: null,
            missingValues: {}
        };
        
        // Calculate survival rate
        if (trainData.labels) {
            const survived = trainData.labels.filter(l => l === 1).length;
            stats.survivalRate = (survived / trainData.labels.length * 100).toFixed(1);
        }
        
        // Calculate missing values percentage
        FEATURE_COLS.forEach(feature => {
            const missing = trainData.raw.filter(row => 
                row[feature] === undefined || row[feature] === null || row[feature] === ''
            ).length;
            stats.missingValues[feature] = ((missing / trainData.raw.length) * 100).toFixed(1);
        });
        
        // Show statistics
        let statsHTML = `<h3>Data Statistics</h3>`;
        statsHTML += `<p><strong>Total Rows:</strong> ${stats.totalRows}</p>`;
        statsHTML += `<p><strong>Features:</strong> ${stats.features.join(', ')}</p>`;
        if (stats.survivalRate) {
            statsHTML += `<p><strong>Survival Rate:</strong> ${stats.survivalRate}%</p>`;
        }
        
        statsHTML += `<h4>Missing Values (%)</h4><ul>`;
        Object.entries(stats.missingValues).forEach(([feature, percent]) => {
            statsHTML += `<li>${feature}: ${percent}%</li>`;
        });
        statsHTML += `</ul>`;
        
        elements.inspectionCharts.innerHTML = statsHTML;
        
        // Show visualizations if tfvis is available
        if (typeof tfvis !== 'undefined') {
            // Survival by Sex chart
            const sexData = {};
            trainData.raw.forEach(row => {
                if (row.Sex && row.Survived !== undefined) {
                    const sex = row.Sex;
                    if (!sexData[sex]) {
                        sexData[sex] = { survived: 0, total: 0 };
                    }
                    sexData[sex].total++;
                    if (row.Survived == 1) {
                        sexData[sex].survived++;
                    }
                }
            });
            
            const sexChartData = Object.entries(sexData).map(([sex, data]) => ({
                x: sex,
                y: (data.survived / data.total * 100).toFixed(1)
            }));
            
            tfvis.render.barchart(
                { name: 'Survival Rate by Sex (%)', tab: 'Data Inspection' },
                { values: sexChartData },
                { xLabel: 'Sex', yLabel: 'Survival %', height: 300 }
            );
            
            // Survival by Pclass chart
            const pclassData = {};
            trainData.raw.forEach(row => {
                if (row.Pclass && row.Survived !== undefined) {
                    const pclass = `Class ${row.Pclass}`;
                    if (!pclassData[pclass]) {
                        pclassData[pclass] = { survived: 0, total: 0 };
                    }
                    pclassData[pclass].total++;
                    if (row.Survived == 1) {
                        pclassData[pclass].survived++;
                    }
                }
            });
            
            const pclassChartData = Object.entries(pclassData).map(([pclass, data]) => ({
                x: pclass,
                y: (data.survived / data.total * 100).toFixed(1)
            }));
            
            tfvis.render.barchart(
                { name: 'Survival Rate by Passenger Class (%)', tab: 'Data Inspection' },
                { values: pclassChartData },
                { xLabel: 'Passenger Class', yLabel: 'Survival %', height: 300 }
            );
        }
        
    } catch (error) {
        showStatus('error', `Inspection error: ${error.message}`);
    }
}

// ============================================
// 3. Preprocessing Functions
// ============================================

/**
 * Preprocess data: imputation, standardization, one-hot encoding
 */
function preprocessData(data, isTest = false) {
    return tf.tidy(() => {
        const rawData = data.raw;
        
        // Calculate statistics from training data
        const ageValues = rawData.map(row => parseFloat(row.Age)).filter(val => !isNaN(val));
        const fareValues = rawData.map(row => parseFloat(row.Fare)).filter(val => !isNaN(val));
        
        const ageMedian = ageValues.length > 0 ? 
            ageValues.sort((a, b) => a - b)[Math.floor(ageValues.length / 2)] : 30;
        const fareMedian = fareValues.length > 0 ? 
            fareValues.sort((a, b) => a - b)[Math.floor(fareValues.length / 2)] : 32.2;
        
        // Calculate fare mean and std for standardization
        const fareMean = fareValues.reduce((sum, val) => sum + val, 0) / fareValues.length || 0;
        const fareStd = Math.sqrt(
            fareValues.reduce((sum, val) => sum + Math.pow(val - fareMean, 2), 0) / fareValues.length
        ) || 1;
        
        // Calculate age mean and std for standardization
        const ageMean = ageValues.reduce((sum, val) => sum + val, 0) / ageValues.length || 0;
        const ageStd = Math.sqrt(
            ageValues.reduce((sum, val) => sum + Math.pow(val - ageMean, 2), 0) / ageValues.length
        ) || 1;
        
        // Find mode for Embarked
        const embarkedCounts = {};
        rawData.forEach(row => {
            if (row.Embarked && row.Embarked.trim() !== '') {
                embarkedCounts[row.Embarked] = (embarkedCounts[row.Embarked] || 0) + 1;
            }
        });
        const embarkedMode = Object.keys(embarkedCounts).length > 0 ?
            Object.keys(embarkedCounts).reduce((a, b) => embarkedCounts[a] > embarkedCounts[b] ? a : b) : 'S';
        
        // Preprocess each row
        const processedFeatures = [];
        const processedLabels = !isTest ? [] : null;
        
        rawData.forEach(row => {
            // Handle missing values
            const age = !isNaN(parseFloat(row.Age)) ? parseFloat(row.Age) : ageMedian;
            const fare = !isNaN(parseFloat(row.Fare)) ? parseFloat(row.Fare) : fareMedian;
            const embarked = row.Embarked && row.Embarked.trim() !== '' ? row.Embarked : embarkedMode;
            const sex = row.Sex || 'male';
            const pclass = row.Pclass || '3';
            
            // Standardize Age and Fare
            const ageStdized = (age - ageMean) / ageStd;
            const fareStdized = (fare - fareMean) / fareStd;
            
            // One-hot encoding for categorical variables
            const sexMale = sex === 'male' ? 1 : 0;
            const sexFemale = sex === 'female' ? 1 : 0;
            
            const pclass1 = pclass === '1' ? 1 : 0;
            const pclass2 = pclass === '2' ? 1 : 0;
            const pclass3 = pclass === '3' ? 1 : 0;
            
            const embarkedC = embarked === 'C' ? 1 : 0;
            const embarkedQ = embarked === 'Q' ? 1 : 0;
            const embarkedS = embarked === 'S' ? 1 : 0;
            
            // Create feature vector (7 original features)
            const featureVector = [
                parseInt(pclass),      // Pclass (1,2,3)
                sex === 'female' ? 1 : 0, // Sex (0=male, 1=female)
                ageStdized,            // Standardized Age
                parseInt(row.SibSp || 0), // SibSp
                parseInt(row.Parch || 0), // Parch
                fareStdized,           // Standardized Fare
                ['C', 'Q', 'S'].indexOf(embarked) // Embarked (0=C, 1=Q, 2=S)
            ];
            
            processedFeatures.push(featureVector);
            
            if (!isTest && row.Survived !== undefined) {
                processedLabels.push(parseInt(row.Survived));
            }
        });
        
        // Convert to tensors
        const featuresTensor = tf.tensor2d(processedFeatures);
        const labelsTensor = !isTest ? tf.tensor1d(processedLabels) : null;
        
        return {
            features: featuresTensor,
            labels: labelsTensor,
            ids: data.ids,
            preprocessingInfo: {
                ageMean, ageStd, ageMedian,
                fareMean, fareStd, fareMedian,
                embarkedMode
            }
        };
    });
}

// ============================================
// 4. Model Creation with Sigmoid Gates
// ============================================

/**
 * Create model with Sigmoid Feature Gates
 */
function createModel() {
    // Clear previous model
    if (model) {
        model.dispose();
    }
    
    const useGate = elements.useGateToggle.checked;
    
    if (useGate) {
        // Create Sigmoid gate weights variable
        gateWeights = tf.variable(tf.randomUniform([7, 1], -0.1, 0.1));
        
        // Create Lambda layer for sigmoid gate
        const sigmoidGateLayer = tf.layers.lambda({
            function: (x, gate) => {
                return tf.mul(x, tf.sigmoid(gate));
            },
            functionArgs: { gate: gateWeights }
        });
        
        // Build sequential model with sigmoid gate
        model = tf.sequential();
        model.add(sigmoidGateLayer);
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            kernelInitializer: 'glorotNormal'
        }));
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
        
    } else {
        // Create standard model without gate
        model = tf.sequential();
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [7],
            kernelInitializer: 'glorotNormal'
        }));
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
    }
    
    // Compile model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Show model summary
    model.summary();
    
    showStatus('success', `Model created ${useGate ? 'with' : 'without'} Sigmoid Feature Gates`);
    elements.trainBtn.disabled = false;
    
    return model;
}

// ============================================
// 5. Training Functions
// ============================================

/**
 * Train the model with early stopping
 */
async function trainModel() {
    if (!processedTrainData || !model) {
        showStatus('error', 'No data or model to train');
        return;
    }
    
    isTraining = true;
    elements.trainBtn.disabled = true;
    elements.stopTrainBtn.disabled = false;
    elements.trainingStatus.style.display = 'block';
    
    try {
        // Create training and validation sets (80/20 split)
        const dataSize = processedTrainData.features.shape[0];
        const trainSize = Math.floor(dataSize * 0.8);
        
        const indices = tf.range(0, dataSize);
        const shuffledIndices = indices.gather(tf.util.createShuffledIndices(dataSize));
        
        const trainIndices = shuffledIndices.slice([0], [trainSize]);
        const valIndices = shuffledIndices.slice([trainSize], [dataSize - trainSize]);
        
        const trainFeatures = processedTrainData.features.gather(trainIndices);
        const trainLabels = processedTrainData.labels.gather(trainIndices);
        const valFeatures = processedTrainData.features.gather(valIndices);
        const valLabels = processedTrainData.labels.gather(valIndices);
        
        // Training parameters
        const epochs = 50;
        const batchSize = 32;
        
        // Early stopping variables
        let bestValLoss = Infinity;
        let patience = 10;
        let patienceCounter = 0;
        
        trainingHistory = {
            loss: [],
            val_loss: [],
            accuracy: [],
            val_accuracy: []
        };
        
        // Training loop
        for (let epoch = 0; epoch < epochs && isTraining; epoch++) {
            const history = await model.fit(trainFeatures, trainLabels, {
                epochs: 1,
                batchSize: batchSize,
                validationData: [valFeatures, valLabels],
                shuffle: true,
                verbose: 0
            });
            
            const loss = history.history.loss[0];
            const acc = history.history.acc ? history.history.acc[0] : history.history.accuracy[0];
            const valLoss = history.history.val_loss[0];
            const valAcc = history.history.val_acc ? history.history.val_acc[0] : history.history.val_accuracy[0];
            
            trainingHistory.loss.push(loss);
            trainingHistory.accuracy.push(acc);
            trainingHistory.val_loss.push(valLoss);
            trainingHistory.val_accuracy.push(valAcc);
            
            // Update status
            elements.trainingStatus.innerHTML = 
                `Epoch ${epoch + 1}/${epochs} - Loss: ${loss.toFixed(4)}, Acc: ${acc.toFixed(4)}, ` +
                `Val Loss: ${valLoss.toFixed(4)}, Val Acc: ${valAcc.toFixed(4)}`;
            
            // Early stopping check
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                patienceCounter = 0;
            } else {
                patienceCounter++;
                if (patienceCounter >= patience) {
                    showStatus('success', `Early stopping at epoch ${epoch + 1}`);
                    break;
                }
            }
            
            // Visualize training progress
            if (typeof tfvis !== 'undefined') {
                tfvis.show.history(
                    { name: 'Training History', tab: 'Training' },
                    trainingHistory,
                    ['loss', 'val_loss', 'accuracy', 'val_accuracy']
                );
            }
            
            // Allow UI updates
            await tf.nextFrame();
        }
        
        if (isTraining) {
            showStatus('success', 'Training completed successfully');
            elements.evaluateBtn.disabled = false;
            elements.predictBtn.disabled = false;
            elements.showImportanceBtn.disabled = false;
        } else {
            showStatus('info', 'Training stopped by user');
        }
        
    } catch (error) {
        showStatus('
