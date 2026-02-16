// app.js - The Gradient Puzzle (COMPLETE WORKING VERSION)
// TensorFlow.js demo: baseline MSE vs student model with custom loss + architecture

// --- Global state ---
let xInput;
let targetRamp;
let baselineModel;
let studentModel;
let optimizer;
let stepCount = 0;
let autoTraining = false;
let currentArch = 'compression';

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

// --- Loss components ---
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
    } else {
        throw new Error(`Unknown architecture: ${archType}`);
    }

    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
    return model;
}

// --- FIXED Training step (handles tf.variableGrads correctly) ---
function trainStep() {
    tf.tidy(() => {
        try {
            // Helper function to train a model
            const trainModel = (model, lossFn) => {
                const lossFunction = () => lossFn(targetRamp, model.apply(xInput, { training: true }));
                const { value, grads } = tf.variableGrads(lossFunction);
                
                // Convert grads object to array of {grads: Tensor, variable: Variable}
                const gradVars = Object.keys(grads).map(key => ({
                    grads: grads[key].grad,
                    variable: grads[key].originalVariable
                }));
                
                optimizer.applyGradients(gradVars);
                value.dispose();
                
                // Dispose individual grad tensors
                Object.values(grads).forEach(g => g.grad.dispose());
            };

            // Train both models
            trainModel(baselineModel, baselineLoss);
            trainModel(studentModel, studentLoss);

            stepCount++;

            const predBaseline = baselineModel.predict(xInput);
            const predStudent = studentModel.predict(xInput);
            const lossBaseline = baselineLoss(targetRamp, predBaseline);
            const lossStudent = baselineLoss(targetRamp, predStudent); // MSE for comparison

            log(`Step ${stepCount} | Baseline: ${lossBaseline.dataSync()[0].toFixed(4)} | Student: ${lossStudent.dataSync()[0].toFixed(4)}`);
            updateCanvases(predBaseline, predStudent);
            
            lossBaseline.dispose();
            lossStudent.dispose();
            
        } catch (e) {
            log(`Error: ${e.message}`, true);
            console.error(e);
            stopAutoTrain();
        }
    });
}

// --- Canvas rendering ---
function updateCanvases(predBaseline, predStudent) {
    const inputData = xInput.dataSync();
    drawCanvas('canvasInput', inputData);
    
    const baseData = predBaseline.dataSync();
    drawCanvas('canvasBaseline', baseData);
    
    const studentData = predStudent.dataSync();
    drawCanvas('canvasStudent', studentData);
    
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
    await tf.ready(); // Wait for TF.js to initialize
    
    xInput = createFixedNoise();
    targetRamp = createTargetRamp();

    baselineModel = createBaselineModel();
    studentModel = createStudentModel(currentArch);
    optimizer = tf.train.adam(0.01);

    const predBase = baselineModel.predict(xInput);
    const predStudent = studentModel.predict(xInput);
    updateCanvases(predBase, predStudent);

    log('Models ready. Click Step Train or Auto Train!');

    // Event listeners
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
            studentModel.dispose();
            studentModel = createStudentModel(currentArch);
            log(`Switched to ${currentArch} architecture.`);
            const predStudent = studentModel.predict(xInput);
            const predBase = baselineModel.predict(xInput);
            updateCanvases(predBase, predStudent);
        });
    });
}

// Start when page loads
window.addEventListener('load', init);

