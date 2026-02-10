// app.js - Neural Network Design: The Gradient Puzzle
// Students: Modify TODO sections in this file

// ====================
// Global State & Config
// ====================
let stepCount = 0;
let autoTraining = false;
let animationFrameId = null;

// Fixed input tensor (1 batch, 16x16, 1 channel)
const xInput = tf.tidy(() => 
    tf.randomUniform([1, 16, 16, 1], -1, 1).clone()
);

// Generated coefficients for custom loss - USE THESE EXACT VALUES
// TODO-B: Implement custom loss using these coefficients
const smoothnessCoeff = 0.18; // Between 0.05 and 0.3
const directionCoeff = 0.09;  // Between 0.02 and 0.15

// ====================
// Model Creation
// ====================

/**
 * Creates the baseline model (fixed architecture, MSE loss)
 * Architecture: 16x16 → 8x8 (compression) → 16x16
 */
function createBaselineModel() {
    const model = tf.sequential();
    
    // Encoder
    model.add(tf.layers.conv2d({
        inputShape: [16, 16, 1],
        filters: 4,
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
    }));
    
    // Bottleneck
    model.add(tf.layers.conv2d({
        filters: 8,
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
    }));
    
    // Decoder
    model.add(tf.layers.conv2dTranspose({
        filters: 4,
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
    }));
    
    model.add(tf.layers.conv2dTranspose({
        filters: 1,
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        activation: 'tanh',
        kernelInitializer: 'glorotNormal'
    }));
    
    return model;
}

/**
 * Creates the student model with selectable architecture.
 * TODO-A: Implement transformation and expansion architectures.
 * Currently only compression works.
 */
function createStudentModel(archType = 'compression') {
    // TODO-A: Complete this function to handle three architecture types
    // compression: reduce spatial dimensions (as baseline)
    // transformation: same input/output size but with feature transformation
    // expansion: increase spatial dimensions
    
    if (archType === 'compression') {
        // Current implementation (same as baseline)
        const model = tf.sequential();
        model.add(tf.layers.conv2d({
            inputShape: [16, 16, 1],
            filters: 4,
            kernelSize: 3,
            strides: 2,
            padding: 'same',
            activation: 'relu',
            kernelInitializer: 'glorotNormal'
        }));
        model.add(tf.layers.conv2d({
            filters: 8,
            kernelSize: 3,
            strides: 2,
            padding: 'same',
            activation: 'relu',
            kernelInitializer: 'glorotNormal'
        }));
        model.add(tf.layers.conv2dTranspose({
            filters: 4,
            kernelSize: 3,
            strides: 2,
            padding: 'same',
            activation: 'relu',
            kernelInitializer: 'glorotNormal'
        }));
        model.add(tf.layers.conv2dTranspose({
            filters: 1,
            kernelSize: 3,
            strides: 2,
            padding: 'same',
            activation: 'tanh',
            kernelInitializer: 'glorotNormal'
        }));
        return model;
    }
    
    // TODO-A: Add transformation architecture (same input/output size)
    if (archType === 'transformation') {
        // Student task: design a network that transforms without compression
        // Example: series of conv layers with stride 1, same padding
        throw new Error('Transformation architecture not implemented yet - see TODO-A');
    }
    
    // TODO-A: Add expansion architecture (output larger than input)
    if (archType === 'expansion') {
        // Student task: design a network that outputs larger than 16x16
        // Example: use transpose conv with stride > 1
        throw new Error('Expansion architecture not implemented yet - see TODO-A');
    }
    
    throw new Error(`Unknown architecture type: ${archType}`);
}

// ====================
// Loss Functions
// ====================

/** Mean squared error */
function mse(yTrue, yPred) {
    return tf.tidy(() => yTrue.sub(yPred).square().mean());
}

/** Smoothness penalty (total variation style) */
function smoothness(yPred) {
    return tf.tidy(() => {
        // Calculate differences between neighboring pixels
        const diffX = yPred.slice([0, 0, 0, 0], [1, 15, 16, 1])
                         .sub(yPred.slice([0, 1, 0, 0], [1, 15, 16, 1]));
        const diffY = yPred.slice([0, 0, 0, 0], [1, 16, 15, 1])
                         .sub(yPred.slice([0, 0, 1, 0], [1, 16, 15, 1]));
        
        return diffX.square().mean().add(diffY.square().mean());
    });
}

/** Direction penalty (encourage left-dark / right-bright gradient) */
function directionX(yPred) {
    return tf.tidy(() => {
        const shape = yPred.shape;
        const width = shape[2];
        
        // Create coordinate tensor: -1 at left, +1 at right
        const coords = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]);
        const targetGradient = coords; // Ideal pattern
        
        // Compare each pixel's value to its ideal position-based value
        const meanVal = yPred.mean(); // Center around mean
        const centered = yPred.sub(meanVal);
        
        return centered.sub(targetGradient).square().mean();
    });
}

/**
 * Student loss function - modify this!
 * TODO-B: Start from MSE and add smoothness + direction penalties
 * Use the exact coefficients: smoothnessCoeff = 0.18, directionCoeff = 0.09
 */
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const mseLoss = mse(yTrue, yPred);
        
        // TODO-B: Implement custom loss with smoothness and direction penalties
        // Currently only MSE - add the two regularization terms
        // const smoothLoss = smoothness(yPred);
        // const dirLoss = directionX(yPred);
        // return mseLoss.add(smoothLoss.mul(smoothnessCoeff))
        //              .add(dirLoss.mul(directionCoeff));
        
        return mseLoss; // Replace with custom loss
    });
}

// ====================
// Training Setup
// ====================
const baselineModel = createBaselineModel();
let studentModel = createStudentModel('compression');
const optimizer = tf.train.adam(0.01);

// ====================
// UI Elements
// ====================
const inputCanvas = document.getElementById('inputCanvas');
const baselineCanvas = document.getElementById('baselineCanvas');
const studentCanvas = document.getElementById('studentCanvas');
const stepCountElement = document.getElementById('stepCount');
const baselineLossElement = document.getElementById('baselineLoss');
const studentLossElement = document.getElementById('studentLoss');
const archDisplayElement = document.getElementById('archDisplay');
const errorLogElement = document.getElementById('errorLog');
const trainStepButton = document.getElementById('trainStep');
const autoTrainButton = document.getElementById('autoTrain');
const resetButton = document.getElementById('reset');
const archRadios = document.querySelectorAll('input[name="arch"]');

// ====================
// Visualization
// ====================

/** Render a tensor to canvas */
function renderTensorToCanvas(tensor, canvas) {
    tf.tidy(() => {
        const [batch, height, width, channels] = tensor.shape;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(width, height);
        
        // Normalize to 0-255
        const normalized = tensor.add(1).div(2).mul(255);
        const data = normalized.dataSync();
        
        for (let i = 0; i < height * width; i++) {
            const val = Math.min(255, Math.max(0, data[i]));
            imageData.data[i * 4] = val;     // R
            imageData.data[i * 4 + 1] = val; // G
            imageData.data[i * 4 + 2] = val; // B
            imageData.data[i * 4 + 3] = 255; // A
        }
        
        // Scale up for display
        ctx.putImageData(imageData, 0, 0);
    });
}

/** Update all visualizations */
function updateVisualizations() {
    tf.tidy(() => {
        const baselinePred = baselineModel.predict(xInput);
        const studentPred = studentModel.predict(xInput);
        
        renderTensorToCanvas(xInput, inputCanvas);
        renderTensorToCanvas(baselinePred, baselineCanvas);
        renderTensorToCanvas(studentPred, studentCanvas);
        
        // TODO-C: Compare losses and observe differences
        const baselineLossValue = mse(xInput, baselinePred).dataSync()[0];
        const studentLossValue = studentLoss(xInput, studentPred).dataSync()[0];
        
        baselineLossElement.textContent = baselineLossValue.toFixed(6);
        studentLossElement.textContent = studentLossValue.toFixed(6);
        stepCountElement.textContent = stepCount;
    });
}

// ====================
// Training Loop
// ====================

/** Single training step */
function trainStep() {
    try {
        tf.tidy(() => {
            // Track variables for gradient computation
            const baselineVars = baselineModel.trainableWeights;
            const studentVars = studentModel.trainableWeights;
            
            // Baseline training (MSE only)
            optimizer.minimize(() => {
                const pred = baselineModel.predict(xInput);
                return mse(xInput, pred);
            }, true, baselineVars);
            
            // Student training (custom loss)
            optimizer.minimize(() => {
                const pred = studentModel.predict(xInput);
                return studentLoss(xInput, pred);
            }, true, studentVars);
        });
        
        stepCount++;
        updateVisualizations();
        errorLogElement.textContent = ''; // Clear errors on success
    } catch (error) {
        errorLogElement.innerHTML = `<span class="error">Error in trainStep: ${error.message}</span>`;
        console.error(error);
    }
}

/** Auto-training loop */
function autoTrainStep() {
    if (!autoTraining) return;
    
    // Limit steps per frame for performance
    for (let i = 0; i < 3; i++) {
        trainStep();
    }
    
    animationFrameId = requestAnimationFrame(autoTrainStep);
}

/** Reset models to initial weights */
function resetModels() {
    // Dispose old models
    studentModel.dispose();
    
    // Create new models with fresh weights
    const archType = document.querySelector('input[name="arch"]:checked').value;
    studentModel = createStudentModel(archType);
    
    stepCount = 0;
    updateVisualizations();
    errorLogElement.textContent = 'Models reset to initial weights.';
}

// ====================
// Event Listeners
// ====================

trainStepButton.addEventListener('click', trainStep);

autoTrainButton.addEventListener('click', () => {
    autoTraining = !autoTraining;
    
    if (autoTraining) {
        autoTrainButton.textContent = 'Stop Auto Train';
        autoTrainButton.classList.add('running');
        autoTrainStep();
    } else {
        autoTrainButton.textContent = 'Auto Train';
        autoTrainButton.classList.remove('running');
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
        }
    }
});

resetButton.addEventListener('click', resetModels);

archRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
        const archType = e.target.value;
        archDisplayElement.textContent = archType;
        
        try {
            // Recreate student model with new architecture
            studentModel.dispose();
            studentModel = createStudentModel(archType);
            updateVisualizations();
            errorLogElement.textContent = `Switched to ${archType} architecture.`;
        } catch (error) {
            errorLogElement.innerHTML = `<span class="error">Architecture error: ${error.message}</span>`;
        }
    });
});

// ====================
// Initialization
// ====================

// Initial visualization
updateVisualizations();

// Log initial status
errorLogElement.textContent = 'Ready. Modify app.js to implement TODO sections.';
console.log('=== The Gradient Puzzle ===');
console.log('Student tasks:');
console.log('1. TODO-A: Implement transformation/expansion architectures in createStudentModel()');
console.log('2. TODO-B: Implement custom loss with smoothness (coeff=0.18) + direction (coeff=0.09)');
console.log('3. TODO-C: Observe how custom loss affects output structure vs baseline');

// Memory cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (autoTraining) {
        cancelAnimationFrame(animationFrameId);
    }
    tf.dispose([baselineModel, studentModel, xInput]);
});
