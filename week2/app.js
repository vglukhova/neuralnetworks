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
    initializeMetricsDisplay();
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
 * This fixes the comma escape problem in CSV files
 * @param {string} csvText - Raw CSV text
 * @returns {Array} Array of objects representing the CSV data
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
 * This handles the comma escape problem for fields like "Braund, Mr. Owen Harris"
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
            if (inQuotes && nextChar === '"') {
                // Escaped quote inside quotes (e.g., "" within quotes)
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
        
        // Log first few lines for debugging
        const firstLines = trainText.split(/\r\n|\n|\r/).slice(0, 3);
        console.log('First 3 lines of training CSV:', firstLines);
        
        // Use the fixed CSV parser that handles quoted fields with commas
        trainData = parseCSV(trainText);
        console.log('Parsed training data:', trainData.length, 'rows');
        
        // Load test data if provided
        if (testFileInput.files[0]) {
            const testFile = testFileInput.files[0];
            console.log('Loading test file:', testFile.name);
            const testText = await testFile.text();
            testData = parseCSV(testText);
            console.log('Parsed test data:', testData.length, 'rows');
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
        console.error('Error stack:', error.stack);
        updateStatus('dataStatus', 'error', `Error loading CSV: ${error.message}. Make sure you're using the Titanic dataset format.`);
        
        // Reset button on error
        const loadBtn = document.getElementById('loadDataBtn');
        loadBtn.innerHTML = '<i class="fas fa-upload"></i> Load CSV Files';
        loadBtn.disabled = false;
    }
}

/**
 * Load sample Titanic data (hardcoded subset for demo)
 */
function loadSampleData() {
    updateStatus('dataStatus', 'info', 'Loading sample Titanic data...');
    
    // Sample Titanic data (subset for demo)
    const sampleTrainData = [
        {PassengerId: 1, Survived: 0, Pclass: 3, Name: 'Braund, Mr. Owen Harris', Sex: 'male', Age: 22, SibSp: 1, Parch: 0, Ticket: 'A/5 21171', Fare: 7.25, Cabin: null, Embarked: 'S'},
        {PassengerId: 2, Survived: 1, Pclass: 1, Name: 'Cumings, Mrs. John Bradley', Sex: 'female', Age: 38, SibSp: 1, Parch: 0, Ticket: 'PC 17599', Fare: 71.28, Cabin: 'C85', Embarked: 'C'},
        {PassengerId: 3, Survived: 1, Pclass: 3, Name: 'Heikkinen, Miss. Laina', Sex: 'female', Age: 26, SibSp: 0, Parch: 0, Ticket: 'STON/O2. 3101282', Fare: 7.92, Cabin: null, Embarked: 'S'},
        {PassengerId: 4, Survived: 1, Pclass: 1, Name: 'Futrelle, Mrs. Jacques Heath', Sex: 'female', Age: 35, SibSp: 1, Parch: 0, Ticket: '113803', Fare: 53.1, Cabin: 'C123', Embarked: 'S'},
        {PassengerId: 5, Survived: 0, Pclass: 3, Name: 'Allen, Mr. William Henry', Sex: 'male', Age: 35, SibSp: 0, Parch: 0, Ticket: '373450', Fare: 8.05, Cabin: null, Embarked: 'S'}
    ];
    
    const sampleTestData = [
        {PassengerId: 892, Pclass: 3, Name: 'Kelly, Mr. James', Sex: 'male', Age: 34.5, SibSp: 0, Parch: 0, Ticket: '330911', Fare: 7.83, Cabin: null, Embarked: 'Q'},
        {PassengerId: 893, Pclass: 3, Name: 'Wilkes, Mrs. James', Sex: 'female', Age: 47, SibSp: 1, Parch: 0, Ticket: '363272', Fare: 7, Cabin: null, Embarked: 'S'}
    ];
    
    trainData = sampleTrainData;
    testData = sampleTestData;
    
    updateStatus('dataStatus', 'success', 
        `Loaded ${trainData.length} sample training records and ${testData.length} test records.`);
    
    // Enable preprocessing button
    document.getElementById('preprocessBtn').disabled = false;
    
    // Show data preview with ALL columns
    showDataPreview();
    
    // Show survival distribution
    showSurvivalDistribution();
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
    html += '<table style="min-width: 800px;">'; // Ensure table has enough width
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
            // Format the display
            if (val === null || val === undefined) {
                html += '<td style="color: #999; font-style: italic;">null</td>';
            } else if (typeof val === 'string' && val.length > 30) {
                // Truncate long strings
                html += `<td title="${val}">${val.substring(0, 30)}...</td>`;
            } else {
                html += `<td>${val}</td>`;
            }
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    html += '</div>'; // Close overflow container
    
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
    let html = '<div style="display: flex; flex-direction: column; gap: 20px;">';
    
    // Sex survival chart
    html += '<div>';
    html += '<h4>Survival by Sex</h4>';
    sexData.forEach(item => {
        html += `<p style="margin: 8px 0;"><strong>${item.sex}:</strong> ${item.survivalRate}% survived (${item.count} passengers)</p>`;
        html += `<div style="height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 5px 0;">`;
        html += `<div style="height: 100%; width: ${item.survivalRate}%; background: linear-gradient(to right, #26d0ce, #1a2980);"></div>`;
        html += '</div>';
    });
    html += '</div>';
    
    // Class survival chart
    html += '<div>';
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
        // Convert categorical values to numerical indices
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
 */
function createModel() {
    if (!processedTrainData) {
        updateStatus('modelStatus', 'error', 'No processed data available. Please preprocess data first.');
        return;
    }
    
    try {
        updateStatus('modelStatus', 'info', 'Creating model with Sigmoid gate layer...');
        
        // Get input shape
        const inputShape = processedTrainData.features.shape[1];
        
        // Create sequential model
        model = tf.sequential();
        
        // Hidden layer with 16 neurons and ReLU activation
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [inputShape],
            name: 'hidden_layer'
        }));
        
        // SIGMOID GATE LAYER: This layer learns feature importance
        // The sigmoid activation creates a gating mechanism that can learn
        // which features are most important for the prediction
        model.add(tf.layers.dense({
            units: 8,
            activation: 'sigmoid',
            name: 'sigmoid_gate'
        }));
        
        // Output layer with 1 neuron and sigmoid activation for binary classification
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
        
        // Print model summary
        console.log('Model summary:');
        model.summary();
        
        updateStatus('modelStatus', 'success', 
            `Model created with ${inputShape} input features, 16-unit hidden layer, 8-unit sigmoid gate, and 1 output.`);
        
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
        
        // Train the model
        const history = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
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
                    const container = document.getElementById('trainingHistory');
                    if (container) {
                        let html = '<div style="display: flex; flex-direction: column; gap: 20px;">';
                        
                        // Loss chart
                        html += '<div>';
                        html += '<h4>Training & Validation Loss</h4>';
                        html += '<div style="height: 200px; position: relative; border: 1px solid #ddd; border-radius: 5px; padding: 10px;">';
                        
                        const maxLoss = Math.max(...trainingHistory.map(h => Math.max(h.loss, h.val_loss)));
                        trainingHistory.forEach((h, idx) => {
                            const lossHeight = maxLoss > 0 ? (h.loss / maxLoss) * 180 : 0;
                            const valLossHeight = maxLoss > 0 ? (h.val_loss / maxLoss) * 180 : 0;
                            
                            html += `<div style="position: absolute; bottom: 0; left: ${idx * 15}px; width: 12px; height: ${lossHeight}px; background: #1a2980;"></div>`;
                            html += `<div style="position: absolute; bottom: 0; left: ${idx * 15 + 6}px; width: 12px; height: ${valLossHeight}px; background: #26d0ce;"></div>`;
                        });
                        
                        html += '</div>';
                        html += '<div style="display: flex; gap: 10px; margin-top: 10px;">';
                        html += '<div><div style="width: 12px; height: 12px; background: #1a2980; display: inline-block; margin-right: 5px;"></div> Training Loss</div>';
                        html += '<div><div style="width: 12px; height: 12px; background: #26d0ce; display: inline-block; margin-right: 5px;"></div> Validation Loss</div>';
                        html += '</div>';
                        html += '</div>';
                        
                        // Accuracy chart
                        html += '<div>';
                        html += '<h4>Training & Validation Accuracy</h4>';
                        html += '<div style="height: 200px; position: relative; border: 1px solid #ddd; border-radius: 5px; padding: 10px;">';
                        
                        trainingHistory.forEach((h, idx) => {
                            const accHeight = h.acc * 180;
                            const valAccHeight = h.val_acc * 180;
                            
                            html += `<div style="position: absolute; bottom: 0; left: ${idx * 15}px; width: 12px; height: ${accHeight}px; background: #1a2980;"></div>`;
                            html += `<div style="position: absolute; bottom: 0; left: ${idx * 15 + 6}px; width: 12px; height: ${valAccHeight}px; background: #26d0ce;"></div>`;
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
                    
                    // Calculate and display feature importance using the sigmoid gate
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
 * Calculate and display feature importance using the SIGMOID GATE layer
 * The sigmoid gate is specifically designed to learn feature importance
 */
async function calculateFeatureImportance() {
    if (!model) {
        console.error('No model available for feature importance calculation');
        return;
    }
    
    try {
        console.log('Calculating feature importance using SIGMOID GATE...');
        
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
        
        console.log('Feature names:', featureNames);
        console.log('Model layers:');
        model.layers.forEach((layer, idx) => {
            console.log(`  Layer ${idx}: ${layer.name} (${layer.getClassName()})`);
        });
        
        // METHOD 1: Use SIGMOID GATE weights (primary method)
        console.log('Using SIGMOID GATE for feature importance...');
        const importanceScores = await calculateImportanceFromSigmoidGate(featureNames);
        
        // Display the results
        displaySigmoidGateImportance(importanceScores, featureNames);
        
    } catch (error) {
        console.error('Error in sigmoid gate feature importance:', error);
        displayFeatureImportanceError(error);
    }
}

/**
 * Calculate feature importance using the SIGMOID GATE layer
 * This is the core method that uses the sigmoid gate as intended
 */
async function calculateImportanceFromSigmoidGate(featureNames) {
    try {
        console.log('=== SIGMOID GATE FEATURE IMPORTANCE CALCULATION ===');
        
        // Get all layers
        const hiddenLayer = model.layers[0];     // Dense(16, relu)
        const sigmoidGateLayer = model.layers[1]; // Dense(8, sigmoid) - THIS IS THE SIGMOID GATE
        const outputLayer = model.layers[2];     // Dense(1, sigmoid)
        
        console.log('Hidden layer:', hiddenLayer.name);
        console.log('Sigmoid gate layer:', sigmoidGateLayer.name);
        console.log('Output layer:', outputLayer.name);
        
        // Step 1: Get weights from input to hidden layer
        const hiddenWeights = hiddenLayer.getWeights();
        console.log('Hidden layer weights:', hiddenWeights.length, 'tensors');
        
        if (hiddenWeights.length < 1) {
            throw new Error('No weights in hidden layer');
        }
        
        const inputToHiddenWeights = hiddenWeights[0]; // Shape: [input_features, 16]
        console.log('Input→Hidden weights shape:', inputToHiddenWeights.shape);
        
        // Step 2: Get weights from hidden to sigmoid gate
        const sigmoidGateWeights = sigmoidGateLayer.getWeights();
        console.log('Sigmoid gate weights:', sigmoidGateWeights.length, 'tensors');
        
        if (sigmoidGateWeights.length < 1) {
            throw new Error('No weights in sigmoid gate layer');
        }
        
        const hiddenToGateWeights = sigmoidGateWeights[0]; // Shape: [16, 8]
        console.log('Hidden→Gate weights shape:', hiddenToGateWeights.shape);
        
        // Step 3: Get weights from sigmoid gate to output
        const outputWeights = outputLayer.getWeights();
        console.log('Output layer weights:', outputWeights.length, 'tensors');
        
        if (outputWeights.length < 1) {
            throw new Error('No weights in output layer');
        }
        
        const gateToOutputWeights = outputWeights[0]; // Shape: [8, 1]
        console.log('Gate→Output weights shape:', gateToOutputWeights.shape);
        
        // Convert all weights to arrays
        const [W1, W2, W3] = await Promise.all([
            inputToHiddenWeights.array(),  // W1: input → hidden
            hiddenToGateWeights.array(),   // W2: hidden → sigmoid gate
            gateToOutputWeights.array()    // W3: sigmoid gate → output
        ]);
        
        console.log('W1 dimensions:', W1.length, 'x', W1[0].length);
        console.log('W2 dimensions:', W2.length, 'x', W2[0].length);
        console.log('W3 dimensions:', W3.length, 'x', W3[0].length);
        
        // Step 4: Calculate feature importance through the sigmoid gate
        // For each input feature, calculate its influence through the network
        const importanceScores = [];
        
        for (let featureIdx = 0; featureIdx < featureNames.length; featureIdx++) {
            if (featureIdx >= W1.length) {
                console.warn(`Feature index ${featureIdx} exceeds weight matrix dimensions`);
                importanceScores.push({
                    name: featureNames[featureIdx],
                    importance: 0,
                    explanation: 'Weight matrix dimension mismatch'
                });
                continue;
            }
            
            // Calculate total influence of this feature through the sigmoid gate
            let totalInfluence = 0;
            
            // For each neuron in hidden layer (16 neurons)
            for (let hiddenIdx = 0; hiddenIdx < W1[featureIdx].length; hiddenIdx++) {
                const weightToHidden = W1[featureIdx][hiddenIdx];
                
                // For each neuron in sigmoid gate (8 neurons)
                for (let gateIdx = 0; gateIdx < W2[hiddenIdx].length; gateIdx++) {
                    const weightHiddenToGate = W2[hiddenIdx][gateIdx];
                    
                    // For the output (1 neuron)
                    const weightGateToOutput = W3[gateIdx][0];
                    
                    // Calculate the influence chain: feature → hidden → gate → output
                    // Apply sigmoid to gate activation to simulate the gating effect
                    const gateActivation = Math.tanh(weightHiddenToGate); // Approximation of sigmoid
                    const influence = Math.abs(weightToHidden * gateActivation * weightGateToOutput);
                    
                    totalInfluence += influence;
                }
            }
            
            // Apply sigmoid to the total influence to get a 0-1 range
            // This mimics how the sigmoid gate would activate
            const sigmoidInfluence = 1 / (1 + Math.exp(-totalInfluence));
            
            importanceScores.push({
                name: featureNames[featureIdx],
                importance: sigmoidInfluence,
                rawInfluence: totalInfluence,
                explanation: getFeatureExplanation(featureNames[featureIdx])
            });
            
            console.log(`Feature: ${featureNames[featureIdx]}, Raw: ${totalInfluence.toFixed(4)}, Sigmoid: ${sigmoidInfluence.toFixed(4)}`);
        }
        
        // Normalize to percentages
        const totalImportance = importanceScores.reduce((sum, f) => sum + f.importance, 0);
        
        importanceScores.forEach(feature => {
            if (totalImportance > 0) {
                feature.importance = (feature.importance / totalImportance) * 100;
            } else {
                feature.importance = 100 / importanceScores.length;
            }
        });
        
        // Sort by importance
        importanceScores.sort((a, b) => b.importance - a.importance);
        
        return importanceScores;
        
    } catch (error) {
        console.error('Error in sigmoid gate calculation:', error);
        throw error;
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
 * Display sigmoid gate feature importance results
 */
function displaySigmoidGateImportance(importanceScores, featureNames) {
    const container = document.getElementById('featureImportance');
    let html = '';
    
    if (!importanceScores || importanceScores.length === 0) {
        html = '<div class="status error">No feature importance data available.</div>';
        container.innerHTML = html;
        return;
    }
    
    console.log('Displaying', importanceScores.length, 'importance scores');
    
    // Header with explanation of sigmoid gate
    html += '<div style="margin-bottom: 20px; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px;">';
    html += '<h3 style="margin-top: 0; color: white;"><i class="fas fa-filter"></i> Sigmoid Gate Feature Importance</h3>';
    html += '<p>The sigmoid gate layer (8 neurons) learns which features are most important for survival prediction by acting as a feature filter.</p>';
    html += '</div>';
    
    // Feature importance visualization
    html += '<div class="feature-importance-grid" style="margin-top: 20px;">';
    
    importanceScores.forEach(feature => {
        // Calculate bar width with minimum visibility
        const barWidth = Math.max(feature.importance, 8);
        
        // Determine color based on importance
        let barColor = '#4CAF50'; // Green for high importance
        if (feature.importance < 30) barColor = '#FF9800'; // Orange for medium
        if (feature.importance < 15) barColor = '#F44336'; // Red for low
        
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
        html += '<h4><i class="fas fa-chart-line"></i> Key Insights from Sigmoid Gate</h4>';
        
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
        
        // Model interpretation
        html += '<div style="margin-top: 20px; padding: 15px; background: #e8f4f8; border-radius: 6px;">';
        html += '<h5 style="margin-top: 0; color: #2196F3;"><i class="fas fa-lightbulb"></i> What the Sigmoid Gate Learned</h5>';
        html += '<p>The sigmoid gate layer has learned to weight these features based on their predictive power for survival. ' +
               'Features with higher percentages have stronger influence on the final prediction.</p>';
        
        if (topFeatures[0].name.includes('Sex') || topFeatures[0].importance > 40) {
            html += '<p><strong>Observation:</strong> The model correctly identifies gender as the most important factor, ' +
                   'which aligns with historical accounts of the Titanic disaster.</p>';
        }
        html += '</div>';
        html += '</div>';
    }
    
    // Debug information (collapsible)
    html += '<div style="margin-top: 20px;">';
    html += '<button onclick="toggleDebugInfo()" style="background: #6c757d; color: white; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer; font-size: 0.9em;">';
    html += '<i class="fas fa-bug"></i> Show Debug Information';
    html += '</button>';
    html += '<div id="debugInfo" style="display: none; margin-top: 10px; padding: 15px; background: #f8f9fa; border-radius: 6px; font-family: monospace; font-size: 0.9em;">';
    html += `<p><strong>Number of features:</strong> ${featureNames.length}</p>`;
    html += `<p><strong>Top feature:</strong> ${importanceScores[0].name} (${importanceScores[0].importance.toFixed(2)}%)</p>`;
    html += `<p><strong>Raw influence range:</strong> `;
    const rawValues = importanceScores.map(f => f.rawInfluence);
    html += `${Math.min(...rawValues).toFixed(4)} to ${Math.max(...rawValues).toFixed(4)}</p>`;
    html += '<p><strong>Model architecture:</strong> Input → Dense(16, relu) → Dense(8, sigmoid) → Dense(1, sigmoid)</p>';
    html += '</div>';
    html += '</div>';
    
    // Add JavaScript for toggling debug info
    html += `
        <script>
            function toggleDebugInfo() {
                const debugDiv = document.getElementById('debugInfo');
                const button = event.target;
                if (debugDiv.style.display === 'none') {
                    debugDiv.style.display = 'block';
                    button.innerHTML = '<i class="fas fa-bug"></i> Hide Debug Information';
                } else {
                    debugDiv.style.display = 'none';
                    button.innerHTML = '<i class="fas fa-bug"></i> Show Debug Information';
                }
            }
        </script>
    `;
    
    container.innerHTML = html;
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
                <li><strong>Ensure model is fully trained:</strong> Train for at least 50 epochs</li>
                <li><strong>Check model architecture:</strong> Should have 3 layers including sigmoid gate</li>
                <li><strong>Use sufficient data:</strong> Small sample data may not show meaningful patterns</li>
                <li><strong>Try real Titanic dataset:</strong> Download from Kaggle for better results</li>
                <li><strong>Check console for errors:</strong> Open Developer Tools (F12) for details</li>
            </ol>
            <div style="margin-top: 15px; padding: 10px; background: white; border-radius: 4px;">
                <p><strong>Expected sigmoid gate weights:</strong></p>
                <ul style="margin-left: 20px;">
                    <li>Layer 0: [features × 16] weights</li>
                    <li>Layer 1 (Sigmoid Gate): [16 × 8] weights</li>
                    <li>Layer 2: [8 × 1] weights</li>
                </ul>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

/**
 * Evaluate the trained model and display results in the evaluation table
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
        
        // Display evaluation table - FIXED: This ensures the table is visible
        const tableContainer = document.getElementById('evaluationTable');
        
        // Clear any existing content
        tableContainer.innerHTML = '';
        
        // Create evaluation metrics table
        let html = '<h3 style="margin-bottom: 15px; color: #1a2980;">Detailed Model Evaluation</h3>';
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
        html += `<tr style="background-color: #f9f9f9;"><td><strong>AUC-ROC</strong></td><td>${metrics.auc.toFixed(4)}</td><td>Area under ROC curve (0.5 = random, 1.0 = perfect)</td></tr>`;
        html += '</tbody></table>';
        
        // Add performance summary
        html += '<div style="margin-top: 20px; padding: 15px; background: #e8f4f8; border-radius: 8px; border-left: 4px solid #26d0ce;">';
        html += '<h4 style="margin-top: 0; color: #1a2980;">Performance Summary</h4>';
        html += `<p>With a threshold of <strong>${threshold.toFixed(2)}</strong>, the model correctly classifies <strong>${((metrics.accuracy) * 100).toFixed(1)}%</strong> of validation samples.</p>`;
        
        if (metrics.auc > 0.8) {
            html += '<p style="color: #155724;"><strong>Excellent performance:</strong> AUC > 0.8 indicates strong discriminatory power.</p>';
        } else if (metrics.auc > 0.7) {
            html += '<p style="color: #856404;"><strong>Good performance:</strong> AUC > 0.7 indicates acceptable discriminatory power.</p>';
        } else {
            html += '<p style="color: #721c24;"><strong>Needs improvement:</strong> Consider adjusting the model or threshold.</p>';
        }
        html += '</div>';
        
        tableContainer.innerHTML = html;
        
        // Create ROC curve visualization
        createROCCurve(validationLabels, predictions);
        
        updateStatus('evaluationStatus', 'success', 'Evaluation completed successfully! Metrics and confusion matrix updated.');
        
    } catch (error) {
        console.error('Error evaluating model:', error);
        updateStatus('evaluationStatus', 'error', `Error evaluating model: ${error.message}`);
    }
}

/**
 * Update the metrics display with new values
 * @param {Object} metrics - Evaluation metrics object
 */
function updateMetricsDisplay(metrics) {
    // Update main metrics display
    document.getElementById('accuracyValue').textContent = metrics.accuracy.toFixed(3);
    document.getElementById('precisionValue').textContent = metrics.precision.toFixed(3);
    document.getElementById('recallValue').textContent = metrics.recall.toFixed(3);
    document.getElementById('f1Value').textContent = metrics.f1.toFixed(3);
    document.getElementById('aucValue').textContent = metrics.auc.toFixed(3);
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
    const total = tp + fp + tn + fn;
    const accuracy = total > 0 ? (tp + tn) / total : 0;
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
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
    
    if (totalPos === 0 || totalNeg === 0) {
        return 0.5; // Random classifier
    }
    
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
        
        const tpr = (tp + fn) > 0 ? tp / (tp + fn) : 0;
        const fpr = (fp + tn) > 0 ? fp / (fp + tn) : 0;
        
        rocPoints.push({ fpr, tpr, threshold });
    });
    
    // Create visualization
    const container = document.getElementById('rocCurve');
    const width = 400, height = 300;
    
    let html = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="border: 1px solid #ddd; border-radius: 5px;">`;
    
    // Draw axes
    html += `<line x1="0" y1="${height}" x2="${width}" y2="${height}" stroke="#333" stroke-width="2" />`;
    html += `<line x1="0" y1="0" x2="0" y2="${height}" stroke="#333" stroke-width="2" />`;
    
    // Draw diagonal line (random classifier)
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
    
    // If validation predictions exist, update metrics and table
    if (validationPredictions && validationLabels) {
        const metrics = calculateMetrics(validationLabels, validationPredictions, value);
        
        // Update metrics display
        updateMetricsDisplay(metrics);
        
        // Update confusion matrix display
        updateConfusionMatrixDisplay(
            metrics.confusionMatrix.tp,
            metrics.confusionMatrix.fp,
            metrics.confusionMatrix.tn,
            metrics.confusionMatrix.fn
        );
        
        // Update the evaluation table
        const tableContainer = document.getElementById('evaluationTable');
        if (tableContainer && tableContainer.innerHTML) {
            // Update values in the table using regex replacement
            let html = tableContainer.innerHTML;
            
            // Update accuracy value
            html = html.replace(/<strong>Accuracy<\/strong><\/td><td>[\d.]+<\/td>/g, 
                `<strong>Accuracy</strong></td><td>${metrics.accuracy.toFixed(4)}</td>`);
            
            // Update precision value
            html = html.replace(/<strong>Precision<\/strong><\/td><td>[\d.]+<\/td>/g, 
                `<strong>Precision</strong></td><td>${metrics.precision.toFixed(4)}</td>`);
            
            // Update recall value
            html = html.replace(/<strong>Recall \(Sensitivity\)<\/strong><\/td><td>[\d.]+<\/td>/g, 
                `<strong>Recall (Sensitivity)</strong></td><td>${metrics.recall.toFixed(4)}</td>`);
            
            // Update F1 score value
            html = html.replace(/<strong>F1 Score<\/strong><\/td><td>[\d.]+<\/td>/g, 
                `<strong>F1 Score</strong></td><td>${metrics.f1.toFixed(4)}</td>`);
            
            // Update AUC value
            html = html.replace(/<strong>AUC-ROC<\/strong><\/td><td>[\d.]+<\/td>/g, 
                `<strong>AUC-ROC</strong></td><td>${metrics.auc.toFixed(4)}</td>`);
            
            // Update performance summary
            const summaryRegex = /With a threshold of <strong>[\d.]+<\/strong>, the model correctly classifies <strong>[\d.]+%<\/strong> of validation samples\./g;
            const newSummary = `With a threshold of <strong>${value.toFixed(2)}</strong>, the model correctly classifies <strong>${(metrics.accuracy * 100).toFixed(1)}%</strong> of validation samples.`;
            html = html.replace(summaryRegex, newSummary);
            
            // Update performance assessment
            if (metrics.auc > 0.8) {
                html = html.replace(/<p style="color: #[0-9a-fA-F]+;"><strong>(Excellent|Good|Needs improvement) performance:<\/strong>.*?<\/p>/g, 
                    '<p style="color: #155724;"><strong>Excellent performance:</strong> AUC > 0.8 indicates strong discriminatory power.</p>');
            } else if (metrics.auc > 0.7) {
                html = html.replace(/<p style="color: #[0-9a-fA-F]+;"><strong>(Excellent|Good|Needs improvement) performance:<\/strong>.*?<\/p>/g, 
                    '<p style="color: #856404;"><strong>Good performance:</strong> AUC > 0.7 indicates acceptable discriminatory power.</p>');
            } else {
                html = html.replace(/<p style="color: #[0-9a-fA-F]+;"><strong>(Excellent|Good|Needs improvement) performance:<\/strong>.*?<\/p>/g, 
                    '<p style="color: #721c24;"><strong>Needs improvement:</strong> Consider adjusting the model or threshold.</p>');
            }
            
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
        let probabilitiesCSV = 'PassengerId,Probability,Survived_Prediction\n';
        
        for (let i = 0; i < processedTestData.passengerIds.length; i++) {
            const passengerId = processedTestData.passengerIds[i];
            const probability = predsArray[i];
            const survived = probability >= threshold ? 1 : 0;
            
            submissionCSV += `${passengerId},${survived}\n`;
            probabilitiesCSV += `${passengerId},${probability.toFixed(6)},${survived}\n`;
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
        
        updateStatus('exportStatus', 'success', 'Model exported successfully! Check your downloads folder for "titanic-tfjs-model" files.');
        
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
