// app.js - The Gradient Puzzle (FIXED)
// TensorFlow.js demo: baseline MSE vs student model with custom loss + architecture

// --- Global state ---
let xInput;
let targetRamp;
let baselineModel;
let studentModel;
let baselineOptimizer;
let studentOptimizer;
let stepCount = 0;
let autoTraining = false;
let currentArch = 'compression';

// Hyperparameters for student loss
const LAMBDA_TV = 0.1;
const LAMBDA_DIR = 0.01;

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

function createTargetRamp() {
    return tf.tidy(() => {
        const colVals = tf.linspace(0, 1, 16);
        const rows = tf.ones([16, 1]).mul(colVals);
        return rows.reshape([1, 16, 16, 1]);
    });
}

function createFixedNoise() {
    return tf.randomUniform([1, 16, 16, 1], 0, 1);
}

// --- FIXED Loss functions (return scalars directly) ---
function mse(yTrue, yPred) {
    return tf.losses.meanSquaredError(yTrue, yPred);
}

function smoothness(yPred) {
    return tf.tidy(() => {
        const rightDiff = yPred.slice([0,0,0,0], [1,16,15,1]).sub(yPred.slice([0,0,1,0], [1,16,15,1]));
        const downDiff = yPred.slice([0,0,0,0], [1,15,16,1]).sub(yPred.slice([0,1,0,0], [1,15,16,1]));
        return tf.square(rightDiff).sum().add(tf.square(downDiff).sum());
    });
}

function direction(yPred) {
    return tf.tidy(() => {
        return yPred.mul(targetRamp).mean().neg();
    });
}

function baselineLoss(yTrue, yPred) {
    return mse(yTrue, yPred);
}

function studentLoss(yTrue, yPred) {
    const mseVal = mse(yTrue, yPred);
    const tvVal = smoothness(yPred);
    const dirVal = direction(yPred);
    return mseVal.add(tf.scalar(LAMBDA_TV).mul(tvVal)).add(tf.scalar(LAMBDA_DIR).mul(dirVal));
}

// --- Model creators ---
function createBaselineModel() {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [16, 16, 1] }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
    return model;
}

function createStudentModel(archType) {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [16, 16, 1] }));

    if (archType === 'compression') {
        model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    } else if (archType === 'transformation') {
        model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    } else if (archType === 'expansion') {
        model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    }

    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
    return model;
}

// --- CRITICAL FIX: Correct tf.variableGrads usage ---
function trainBaseline() {
    const lossFn = () => {
        const pred = baselineModel.apply(xInput, { training: true });
        return baselineLoss(targetRamp, pred);
    };
    
    const { value, grads } = tf.variableGrads(lossFn, baselineModel.trainableVariables);
    baselineOptimizer.applyGradients(zipGradients(grads, baselineModel.trainableVariables));
    value.dispose();
}

function trainStudent() {
    const lossFn = () => {
        const pred = studentModel.apply(xInput, { training: true });
        return studentLoss(targetRamp, pred);
    };
    
    const { value, grads } = tf.variableGrads(lossFn, studentModel.trainableVariables);
    studentOptimizer.applyGradients(zipGradients(grads, studentModel.trainableVariables));
    value.dispose();
}

function zipGradients(grads, variables) {
    return variables.map((v, i) => ({ grads: grads[i], variable: v }));
}

// --- Main training step ---
function trainStep() {
    tf.tidy(() => {
        try {
            trainBaseline();
            trainStudent();
            
            stepCount++;
            const predBaseline = baselineModel.predict(xInput);
            const predStudent = studentModel.predict(xInput);
            const lossBaseline = baselineLoss(targetRamp, predBaseline);
            const lossStudent = baselineLoss(targetRamp, predStudent); // MSE for comparison

            log(`Step ${stepCount} | Baseline: ${lossBaseline.dataSync()[0].toFixed(4)} | Student: ${lossStudent.dataSync()[0].toFixed(4)}`);
            updateCanvases(predBaseline, predStudent);
        } catch (e) {
            log(`Error: ${e.message}`, true);
        }
    });
}

// --- Canvas rendering ---
function updateCanvases(predBaseline, predStudent) {
    const inputData = xInput.dataSync();
    drawCanvas('canvasInput', inputData);
    drawCanvas('canvasBaseline', predBaseline.dataSync());
    drawCanvas('canvasStudent', predStudent.dataSync());
    
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

// --- Reset ---
function resetModels() {
    tf.tidy(() => {
        baselineModel?.dispose();
        studentModel?.dispose();
        baselineOptimizer?.dispose();
        studentOptimizer?.dispose();

        baselineModel = createBaselineModel();
        studentModel = createStudentModel(currentArch);
        baselineOptimizer = tf.train.adam(0.01);
        studentOptimizer = tf.train.adam(0.01);
        stepCount = 0;

        const predBase = baselineModel.predict(xInput);
        const predStudent = studentModel.predict(xInput);
        updateCanvases(predBase, predStudent);

        clearLog();
        log('Models reset. Ready.');
    });
}

// --- Auto training ---
function stepAutoTrain() {
    if (!autoTraining) return;
    for (let i = 0; i < 5; i++) {
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
    await tf.ready();
    xInput = createFixedNoise();
    targetRamp = createTargetRamp();

    baselineModel = createBaselineModel();
    studentModel = createStudentModel(currentArch);
    baselineOptimizer = tf.train.adam(0.01);
    studentOptimizer = tf.train.adam(0.01);

    const predBase = baselineModel.predict(xInput);
    const predStudent = studentModel.predict(xInput);
    updateCanvases(predBase, predStudent);

    log('Models ready. Step or Auto Train to begin.');

    // Event listeners
    document.getElementById('trainStep').addEventListener('click', () => {
        if (autoTraining) stopAutoTrain();
        trainStep();
    });

    document.getElementById('autoTrain').addEventListener('click', () => {
        if (autoTraining) stopAutoTrain();
        else startAutoTrain();
    });

    document.getElementById('reset').addEventListener('click', () => {
        stopAutoTrain();
        resetModels();
    });

    document.querySelectorAll('input[name="arch"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            stopAutoTrain();
            currentArch = e.target.value;
            studentModel.dispose();
            studentModel = createStudentModel(currentArch);
            log(`Switched to ${currentArch}`);
            const predStudent = studentModel.predict(xInput);
            const predBase = baselineModel.predict(xInput);
            updateCanvases(predBase, predStudent);
        });
    });
}

// Export for HTML
window.initGradientPuzzle = init;


// Start everything when the page loads
window.addEventListener('load', init);
