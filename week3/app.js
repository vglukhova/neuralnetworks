// app.js - The Gradient Puzzle
// TensorFlow.js demo: baseline MSE vs student model with custom loss + architecture

// --- Global state ---
let xInput;                 // fixed noise input [1,16,16,1]
let targetRamp;             // perfect gradient target [1,16,16,1]
let baselineModel;
let studentModel;
let optimizer;
let stepCount = 0;
let autoTraining = false;
let currentArch = 'compression'; // matches radio button

// Hyperparameters for student loss (can be tuned)
const LAMBDA_TV = 0.1;      // smoothness weight
const LAMBDA_DIR = 0.01;    // direction weight

// --- Utility functions ---
function log(message, isError = false) {
    const logDiv = document.getElementById('logArea');
    const className = isError ? 'error' : 'info';
    logDiv.innerHTML += `<span class="${className}">> ${message}</span><br>`;
    logDiv.scrollTop = logDiv.scrollHeight;
}

function clearLog() {
    document.getElementById('logArea').innerHTML = '';
}

// Create target gradient: from 0 (left) to 1 (right)
function createTargetRamp() {
    return tf.tidy(() => {
        const colVals = tf.linspace(0, 1, 16); // shape [16]
        const rows = tf.ones([16, 1]).mul(colVals); // [16,16] each row identical
        return rows.reshape([1, 16, 16, 1]);
    });
}

// Fixed random input (keep same across resets)
function createFixedNoise() {
    return tf.randomUniform([1, 16, 16, 1], 0, 1);
}

// --- Loss components ---
function mse(yTrue, yPred) {
    return tf.losses.meanSquaredError(yTrue, yPred).mean(); // scalar
}

// Smoothness: total variation (squared differences between adjacent pixels)
function smoothness(yPred) {
    return tf.tidy(() => {
        // yPred shape [1,16,16,1]
        const rightDiff = yPred.slice([0,0,0,0], [1,16,15,1]).sub(yPred.slice([0,0,1,0], [1,16,15,1]));
        const downDiff = yPred.slice([0,0,0,0], [1,15,16,1]).sub(yPred.slice([0,1,0,0], [1,15,16,1]));
        const tv = tf.square(rightDiff).sum().add(tf.square(downDiff).sum());
        return tv;
    });
}

// Direction: encourage correlation with target ramp (negative sign for minimization)
function direction(yPred) {
    return tf.tidy(() => {
        // Ldir = -mean(yPred * targetRamp)
        const prod = yPred.mul(targetRamp).mean().neg();
        return prod;
    });
}

// --- Baseline loss (fixed MSE) ---
function baselineLoss(yTrue, yPred) {
    return mse(yTrue, yPred);
}

// --- Student loss (custom: MSE + smoothness + direction) ---
function studentLoss(yTrue, yPred) {
    // Combined loss: encourage gradient structure
    const mseVal = mse(yTrue, yPred);
    const tvVal = smoothness(yPred);
    const dirVal = direction(yPred);
    // Return scalar (adding scalars yields scalar)
    return mseVal.add(LAMBDA_TV * tvVal).add(LAMBDA_DIR * dirVal);
}

// --- Model creators ---
function createBaselineModel() {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [16, 16, 1] }));
    // Baseline uses a simple compression architecture (fixed)
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
    return model;
}

// Create student model with selectable architecture
function createStudentModel(archType) {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [16, 16, 1] }));

    if (archType === 'compression') {
        model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    } else if (archType === 'transformation') {
        // Hidden layer same size as flattened input (256)
        model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    } else if (archType === 'expansion') {
        // Larger hidden layer
        model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    } else {
        throw new Error(`Unknown architecture: ${archType}`);
    }

    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
    return model;
}

// --- Training step for both models using tf.variableGrads (correct API) ---
function trainStep() {
    tf.tidy(() => {
        try {
            // ---- Baseline gradients ----
            const baseLossFn = () => baselineLoss(targetRamp, baselineModel.apply(xInput, { training: true }));
            const { grads: baseGrads } = tf.variableGrads(baseLossFn);
            optimizer.applyGradients(baseGrads);  // Direct use of {value, grads} object

            // ---- Student gradients ----
            const studentLossFn = () => studentLoss(targetRamp, studentModel.apply(xInput, { training: true }));
            const { grads: studentGrads } = tf.variableGrads(studentLossFn);
            optimizer.applyGradients(studentGrads);  // Direct use

            // Update step count and log
            stepCount++;
            const predBaseline = baselineModel.predict(xInput);
            const predStudent = studentModel.predict(xInput);
            const lossBaseline = baselineLoss(targetRamp, predBaseline);
            const lossStudent = studentLoss(targetRamp, predStudent);

            log(`Step ${stepCount} | Baseline loss: ${lossBaseline.dataSync()[0].toFixed(4)} | Student loss: ${lossStudent.dataSync()[0].toFixed(4)}`);
            updateCanvases(predBaseline, predStudent);
        } catch (e) {
            log(`Error in training step: ${e.message}`, true);
            stopAutoTrain();
        }
    });
}

// --- Canvas rendering ---
function updateCanvases(predBaseline, predStudent) {
    // Input canvas
    const inputData = xInput.dataSync();
    drawCanvas('canvasInput', inputData);

    // Baseline output
    const baseData = predBaseline.dataSync();
    drawCanvas('canvasBaseline', baseData);

    // Student output
    const studentData = predStudent.dataSync();
    drawCanvas('canvasStudent', studentData);

    // Clean up tensors passed in (they are from predict, need disposal)
    predBaseline.dispose();
    predStudent.dispose();
}

function drawCanvas(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(16, 16);
    for (let i = 0; i < 256; i++) {
        const val = Math.floor(data[i] * 255);
        imageData.data[i*4] = val;
        imageData.data[i*4+1] = val;
        imageData.data[i*4+2] = val;
        imageData.data[i*4+3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

// --- Reset everything ---
function resetModels() {
    tf.tidy(() => {
        baselineModel?.dispose();
        studentModel?.dispose();
        optimizer?.dispose();

        baselineModel = createBaselineModel();
        studentModel = createStudentModel(currentArch);
        optimizer = tf.train.adam(0.01);
        stepCount = 0;

        // Initial prediction for display
        const predBase = baselineModel.predict(xInput);
        const predStudent = studentModel.predict(xInput);
        updateCanvases(predBase, predStudent);

        clearLog();
        log('Weights reset. Ready.');
    });
}

// --- Auto training ---
function stepAutoTrain() {
    if (!autoTraining) return;
    for (let i = 0; i < 5; i++) { // multiple steps per frame for speed
        trainStep();
    }
    requestAnimationFrame(stepAutoTrain);
}

function startAutoTrain() {
    autoTraining = true;
    document.getElementById('autoTrain').textContent = '⏸ Stop';
    stepAutoTrain();
}

function stopAutoTrain() {
    autoTraining = false;
    document.getElementById('autoTrain').textContent = '▶ Auto Train';
}

// --- Initialization ---
async function init() {
    xInput = createFixedNoise();
    targetRamp = createTargetRamp();

    baselineModel = createBaselineModel();
    studentModel = createStudentModel(currentArch);
    optimizer = tf.train.adam(0.01);

    // Initial render
    const predBase = baselineModel.predict(xInput);
    const predStudent = studentModel.predict(xInput);
    updateCanvases(predBase, predStudent);

    log('Models ready. Step 0 | Baseline loss: — | Student loss: —');

    // --- Event listeners ---
    document.getElementById('trainStep').addEventListener('click', () => {
        if (autoTraining) stopAutoTrain();
        trainStep();
    });

    document.getElementById('autoTrain').addEventListener('click', () => {
        if (autoTraining) {
            stopAutoTrain();
        } else {
            startAutoTrain();
        }
    });

    document.getElementById('reset').addEventListener('click', () => {
        stopAutoTrain();
        resetModels();
    });

    document.querySelectorAll('input[name="arch"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            stopAutoTrain();
            currentArch = e.target.value;
            // Recreate student model with new architecture
            const newStudent = createStudentModel(currentArch);
            studentModel.dispose();
            studentModel = newStudent;
            log(`Switched to ${currentArch} architecture.`);
            // Re-render with new model's predictions
            const predStudent = studentModel.predict(xInput);
            const predBase = baselineModel.predict(xInput);
            updateCanvases(predBase, predStudent);
        });
    });
}

// Start everything when the page loads
window.addEventListener('load', init);
