// app.js - Neural Network Design: The Gradient Puzzle
// Student task: Modify TODO blocks to create emergent gradient patterns!

let xInput, yTarget;
let baselineModel, studentModel;
let optimizer;
let stepCount = 0;
let autoTrainInterval = null;
let currentArch = 'compression';

const canvases = {
    input: document.getElementById('inputCanvas'),
    baseline: document.getElementById('baselineCanvas'),
    student: document.getElementById('studentCanvas')
};
const ctx = {
    input: canvases.input.getContext('2d'),
    baseline: canvases.baseline.getContext('2d'),
    student: canvases.student.getContext('2d')
};

const statusEl = document.getElementById('status');

// === HELPER FUNCTIONS (IMPLEMENTED - USE THESE!) ===
function mse(yTrue, yPred) {
    return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

function smoothness(yPred) {
    // Ltv = SUM(Pi - Pi+1)^2 (total variation loss)
    const [batch, h, w, c] = yPred.shape;
    const pixels = yPred.reshape([h * w]);
    const diffs = tf.sub(pixels.slice([0]), pixels.slice([1]));
    return tf.mean(tf.square(diffs));
}

function directionX(yPred) {
    // Ldir = -Mean(Output * Task) where Task encourages left-dark/right-bright
    const [batch, h, w, c] = yPred.shape;
    const xCoord = tf.linspace(0, 1, w).expandDims(0).expandDims(0).expandDims(-1);
    const task = tf.sub(tf.onesLike(xCoord), xCoord); // 1-left, 0-right
    const weighted = tf.mul(yPred, task);
    return tf.neg(tf.mean(weighted));
}

// === CANVAS RENDERING ===
function renderTensor(canvasCtx, tensor, scale = 10) {
    const data = tensor.dataSync();
    const imgData = ctx.input.createImageData(canvasCtx.canvas.width, canvasCtx.canvas.height);
    
    for (let i = 0; i < data.length; i++) {
        const val = Math.max(0, Math.min(1, data[i])) * 255;
        const idx = i * 4;
        imgData.data[idx] = val;
        imgData.data[idx + 1] = val;
        imgData.data[idx + 2] = val;
        imgData.data[idx + 3] = 255;
    }
    
    canvasCtx.putImageData(imgData, 0, 0);
}

function logStatus(message) {
    stepCount++;
    statusEl.innerHTML += `[Step ${stepCount}] ${message}\n`;
    statusEl.scrollTop = statusEl.scrollHeight;
}

// === BASELINE MODEL (MSE ONLY - FIXED) ===
function createBaselineModel() {
    const model = tf.sequential({
        layers: [
            tf.layers.conv2d({inputShape: [16, 16, 1], filters: 32, kernelSize: 3, padding: 'same', activation: 'relu'}),
            tf.layers.conv2d({filters: 16, kernelSize: 3, padding: 'same', activation: 'relu'}),
            tf.layers.conv2d({filters: 1, kernelSize: 3, padding: 'same', activation: 'sigmoid'})
        ]
    });
    model.compile({optimizer: 'adam', loss: 'meanSquaredError'});
    return model;
}

// === STUDENT MODEL TODO-A: ARCHITECTURE ===
function createStudentModel(archType) {
    // TODO-A: Implement three projection types for student model
    // Baseline is COMPRESSION (16x16 -> 16x16 with bottleneck)
    // TRANSFORMATION: rotate/flip the input pattern
    // EXPANSION: 16x16 -> 32x32 upsampling then back to 16x16
    
    if (archType === 'compression') {
        // Implemented baseline compression (same size with bottleneck)
        return tf.sequential({
            layers: [
                tf.layers.conv2d({inputShape: [16, 16, 1], filters: 32, kernelSize: 3, padding: 'same', activation: 'relu'}),
                tf.layers.conv2d({filters: 8, kernelSize: 3, padding: 'same', activation: 'relu'}),
                tf.layers.conv2d({filters: 1, kernelSize: 3, padding: 'same', activation: 'sigmoid'})
            ]
        });
    } else if (archType === 'transformation') {
        throw new Error('TODO-A: Implement transformation architecture (rotate/flip patterns)');
    } else if (archType === 'expansion') {
        throw new Error('TODO-A: Implement expansion architecture (upsample 16x16 -> 32x32 -> 16x16)');
    }
    throw new Error(`Unknown architecture: ${archType}`);
}

// === TRAINING STEP ===
async function trainStep() {
    const baselinePred = baselineModel.predict(xInput);
    const studentPred = studentModel.predict(xInput);
    
    // Baseline loss (MSE only)
    const baselineLoss = await mse(yTarget, baselinePred).dataSync()[0];
    
    // TODO-B: Student custom loss
    // Start with: return mse(yTarget, yPred)
    // Add: + lambda1 * smoothness(yPred) + lambda2 * directionX(yPred)
    // Try lambda1=0.1, lambda2=0.5 initially
    const studentLossFn = (yTrue, yPred) => {
        // TODO-B: Replace this MSE with Ltotal = Lsortedmse + lambda1*Ltv + lambda2*Ldir
        return mse(yTrue, yPred);
    };
    const studentLoss = await studentLossFn(yTarget, studentPred).dataSync()[0];
    
    // Backprop for baseline
    await baselineModel.fit(xInput, yTarget, {
        epochs: 1,
        verbose: 0,
        callbacks: {
            onBatchEnd: () => baselinePred.dispose()
        }
    });
    
    // Backprop for student (manual loss)
    const grads = tf.variableGrads(studentLossFn(yTarget, studentPred));
    optimizer.applyGradients(grads.grads);
    tf.dispose(grads);
    
    // Render
    renderTensor(ctx.baseline, baselinePred);
    renderTensor(ctx.student, studentPred);
    
    baselinePred.dispose();
    studentPred.dispose();
    
    logStatus(`Baseline: ${baselineLoss.toFixed(4)}, Student: ${studentLoss.toFixed(4)}`);
}

// === INIT ===
async function init() {
    // Fixed random noise input/target
    xInput = tf.tidy(() => tf.randomUniform([1, 16, 16, 1], 0, 1));
    yTarget = tf.tidy(() => tf.randomUniform([1, 16, 16, 1], 0, 1));
    
    optimizer = tf.train.adam(0.01);
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel(currentArch);
    
    // Initial render
    renderTensor(ctx.input, xInput.squeeze());
    await trainStep();
    
    logStatus('Initialized! Modify TODO-A/B in app.js to solve the gradient puzzle.');
}

// === UI EVENT HANDLERS ===
document.getElementById('trainStep').addEventListener('click', trainStep);
document.getElementById('resetWeights').addEventListener('click', async () => {
    baselineModel.dispose();
    studentModel.dispose();
    await init();
    stepCount = 0;
    statusEl.innerHTML = 'Reset!';
});

document.getElementById('autoTrain').addEventListener('click', () => {
    const btn = document.getElementById('autoTrain');
    if (autoTrainInterval) {
        clearInterval(autoTrainInterval);
        autoTrainInterval = null;
        btn.textContent = 'Auto Train (Start)';
    } else {
        autoTrainInterval = setInterval(trainStep, 100);
        btn.textContent = 'Auto Train (Stop)';
    }
});

// Architecture selector (student model only)
document.querySelectorAll('input[name="arch"]').forEach(radio => {
    radio.addEventListener('change', async (e) => {
        currentArch = e.target.value;
        studentModel.dispose();
        studentModel = createStudentModel(currentArch);
        logStatus(`Switched to ${currentArch} architecture`);
    });
});

init();

