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

// Hyperparameters for student loss (students will adjust these)
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

// --- Student loss (TO BE MODIFIED BY STUDENTS) ---
// TODO-B: Implement custom loss by combining MSE, smoothness, and direction.
// The current version is just MSE (identical to baseline). Uncomment and modify the lines below.
function studentLoss(yTrue, yPred) {
    // ===== STUDENT TODO: Uncomment and adjust the combined loss =====
    // const mseVal = mse(yTrue, yPred);
    // const tvVal = smoothness(yPred);
    // const dirVal = direction(yPred);
    // return mseVal + LAMBDA_TV * tvVal + LAMBDA_DIR * dirVal;
    // ===============================================================
    
    // Default: MSE only (to match baseline initially)
    return mse(yTrue, yPred);
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

// TODO-A: Implement different projection types for the student model.
function createStudentModel(archType) {
    // ===== STUDENT TODO: Implement 'transformation' and 'expansion' architectures =====
    // Hint: use different hidden layer sizes.
    // - compression: small hidden (e.g., 32 units)
    // - transformation: same as input dimension (256 units) — already partially implemented
    // - expansion: larger hidden (e.g., 512 units)
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [16, 16, 1] }));

    if (archType === 'compression') {
        model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    } else if (archType === 'transformation') {
        // TODO: replace with correct implementation (e.g., units = 256)
        throw new Error('Transformation architecture not implemented yet (TODO-A)');
    } else if (archType === 'expansion') {
        // TODO: replace with correct implementation (e.g., units = 512)
        throw new Error('Expansion architecture not implemented yet (TODO-A)');
    } else {
        throw new Error(`Unknown architecture: ${archType}`);
    }

    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
    return model;
}

// --- Training step for both models ---
function trainStep() {
    tf.tidy(() => {
        try {
            // Ensure models exist
            if (!baselineModel || !studentModel) return;

            // Get predictions
            const predBaseline = baselineModel.apply(xInput, { training: true });
            const predStudent = studentModel.apply(xInput, { training: true });

            // Compute losses
            const lossBaseline = baselineLoss(targetRamp, predBaseline);
            const lossStudent = studentLoss(targetRamp, predStudent); // student may throw

            // Compute gradients for baseline
            const gradsBaseline = tf.grads(() => lossBaseline);
            const [gradsB] = gradsBaseline([predBaseline], 0); // 0 = yPred? Actually we need vars
            // Better: use tf.variableGrads
            const { value, grads } = tf.variableGrads(() => lossBaseline, baselineModel.trainableVariables);
            optimizer.applyGradients(grads);

            // Compute gradients for student
            const { value: sVal, grads: sGrads } = tf.variableGrads(() => lossStudent, studentModel.trainableVariables);
            optimizer.applyGradients(sGrads);

            // Update log and step count
            stepCount++;
            log(`Step ${stepCount} | Baseline loss: ${lossBaseline.dataSync()[0].toFixed(4)} | Student loss: ${lossStudent.dataSync()[0].toFixed(4)}`);

            // Render
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
        try {
            studentModel = createStudentModel(currentArch);
        } catch (e) {
            log(`Failed to create student model: ${e.message}`, true);
            // Fallback to a dummy model to keep UI working
            studentModel = createBaselineModel(); // temporary
        }
        optimizer = tf.train.adam(0.01);
        stepCount = 0;

        // Initial prediction for display
        const predBase = baselineModel.predict(xInput);
        const predStudent = studentModel.predict(xInput);
        updateCanvases(predBase, predStudent);
        predBase.dispose();
        predStudent.dispose();

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
    predBase.dispose();
    predStudent.dispose();

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
            try {
                // Recreate student model with new architecture
                const newStudent = createStudentModel(currentArch);
                studentModel.dispose();
                studentModel = newStudent;
                log(`Switched to ${currentArch} architecture.`);
                // Re-render with new model's predictions
                const predStudent = studentModel.predict(xInput);
                const predBase = baselineModel.predict(xInput);
                updateCanvases(predBase, predStudent);
                predBase.dispose();
                predStudent.dispose();
            } catch (err) {
                log(`Error: ${err.message}`, true);
            }
        });
    });
}

// Start everything when the page loads
window.addEventListener('load', init);
