// app.js - The Gradient Puzzle (FINAL FIX)
// Uses model.fit() - no manual gradients, no GradientTape

// --- Global state ---
let xInput, targetRamp;
let baselineModel, studentModel;
let stepCount = 0;
let autoTraining = false;
let currentArch = 'compression';

const LAMBDA_TV = 0.1;
const LAMBDA_DIR = 0.01;

// --- Utils ---
function log(message, isError = false) {
    const logDiv = document.getElementById('logArea');
    const className = isError ? 'error' : 'info';
    logDiv.innerHTML += `<span class="${className}">> ${message}</span><br>`;
    logDiv.scrollTop = logDiv.scrollHeight;
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

// --- Canvas rendering ---
function updateCanvases() {
    const inputData = xInput.dataSync();
    const predBaseline = baselineModel.predict(xInput).dataSync();
    const predStudent = studentModel.predict(xInput).dataSync();
    
    drawCanvas('canvasInput', inputData);
    drawCanvas('canvasBaseline', predBaseline);
    drawCanvas('canvasStudent', predStudent);
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

// --- Custom loss for student (works with model.fit) ---
function studentLoss(yTrue, yPred) {
    const mseLoss = tf.losses.meanSquaredError(yTrue, yPred);
    
    // Smoothness (total variation)
    const rightDiff = yPred.slice([0,0,0,0], [1,16,15,1]).sub(yPred.slice([0,0,1,0], [1,16,15,1]));
    const downDiff = yPred.slice([0,0,0,0], [1,15,16,1]).sub(yPred.slice([0,1,0,0], [1,15,16,1]));
    const tvLoss = tf.square(rightDiff).sum().add(tf.square(downDiff).sum());
    
    // Direction (correlation with target)
    const dirLoss = yPred.mul(targetRamp).mean().neg();
    
    return mseLoss.add(tf.scalar(LAMBDA_TV).mul(tvLoss)).add(tf.scalar(LAMBDA_DIR).mul(dirLoss));
}

// --- Models ---
function createBaselineModel() {
    const model = tf.sequential();
    model.add(tf.layers.flatten({inputShape: [16, 16, 1]}));
    model.add(tf.layers.dense({units: 32, activation: 'relu'}));
    model.add(tf.layers.dense({units: 256, activation: 'sigmoid'}));
    model.add(tf.layers.reshape({targetShape: [16, 16, 1]}));
    model.compile({optimizer: 'adam', loss: 'meanSquaredError'});
    return model;
}

function createStudentModel(arch) {
    const model = tf.sequential();
    model.add(tf.layers.flatten({inputShape: [16, 16, 1]}));
    
    if (arch === 'compression') model.add(tf.layers.dense({units: 32, activation: 'relu'}));
    else if (arch === 'transformation') model.add(tf.layers.dense({units: 256, activation: 'relu'}));
    else model.add(tf.layers.dense({units: 512, activation: 'relu'}));
    
    model.add(tf.layers.dense({units: 256, activation: 'sigmoid'}));
    model.add(tf.layers.reshape({targetShape: [16, 16, 1]}));
    model.compile({optimizer: 'adam', loss: studentLoss});
    return model;
}

// --- Training step using model.fit (SIMPLE & WORKING) ---
async function trainStep() {
    try {
        // Train 1 step (batchSize=1)
        await baselineModel.fit(xInput, targetRamp, {
            epochs: 1, batchSize: 1, verbose: 0
        });
        
        await studentModel.fit(xInput, targetRamp, {
            epochs: 1, batchSize: 1, verbose: 0
        });
        
        stepCount++;
        updateCanvases();
        
        // Log MSE loss for comparison
        const predB = baselineModel.predict(xInput);
        const predS = studentModel.predict(xInput);
        const mseB = tf.losses.meanSquaredError(targetRamp, predB).dataSync()[0];
        const mseS = tf.losses.meanSquaredError(targetRamp, predS).dataSync()[0];
        
        log(`Step ${stepCount} | Baseline: ${mseB.toFixed(4)} | Student: ${mseS.toFixed(4)}`);
        
        predB.dispose();
        predS.dispose();
    } catch (e) {
        log(`Error: ${e.message}`, true);
    }
}

// --- Reset ---
async function resetModels() {
    await tf.disposeVariables();
    baselineModel?.dispose();
    studentModel?.dispose();
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel(currentArch);
    stepCount = 0;
    
    xInput = createFixedNoise();
    targetRamp = createTargetRamp();
    
    updateCanvases();
    clearLog();
    log('Models reset. Ready.');
}

// --- Auto train ---
let autoId;
async function startAutoTrain() {
    autoTraining = true;
    document.getElementById('autoTrain').textContent = '⏸ Stop';
    autoId = setInterval(trainStep, 50);
}

function stopAutoTrain() {
    autoTraining = false;
    document.getElementById('autoTrain').textContent = '▶ Auto Train';
    if (autoId) clearInterval(autoId);
}

// --- Init ---
async function init() {
    await tf.ready();
    xInput = createFixedNoise();
    targetRamp = createTargetRamp();
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel(currentArch);
    
    updateCanvases();
    log('Models ready - NO GRADIENTTAPE!');
    
    // Events
    document.getElementById('trainStep').onclick = () => {
        if (autoTraining) stopAutoTrain();
        trainStep();
    };
    
    document.getElementById('autoTrain').onclick = () => {
        if (autoTraining) stopAutoTrain();
        else startAutoTrain();
    };
    
    document.getElementById('reset').onclick = () => {
        stopAutoTrain();
        resetModels();
    };
    
    document.querySelectorAll('input[name="arch"]').forEach(radio => {
        radio.onchange = (e) => {
            stopAutoTrain();
            currentArch = e.target.value;
            studentModel.dispose();
            studentModel = createStudentModel(currentArch);
            log(`Switched to ${currentArch}`);
            updateCanvases();
        };
    });
}

window.addEventListener('load', init);
