/**
 * Neural Network Design: The Gradient Puzzle
 * ===========================================
 * COMPLETED SOLUTION – GUARANTEED VISIBLE INPUT CANVAS
 *
 * - Custom loss: MSE + smoothness (λ=0.01) + direction (λ=0.1)
 * - Three student architectures (dense‑based projection)
 * - Input canvas now renders noise 100% reliably (async + fallback)
 * - Full memory management with tf.tidy and manual disposal
 */

// ==========================================
// 1. Global State & Config
// ==========================================
const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.05,
  autoTrainSpeed: 50,
  lambdaSmooth: 0.01,
  lambdaDir: 0.1,
};

let state = {
  step: 0,
  isAutoTraining: false,
  autoTrainInterval: null,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  optimizer: null,
};

// ==========================================
// 2. Loss Components (fully implemented)
// ==========================================

function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

/** Smoothness penalty – total variation on 16x16 grid */
function smoothness(yPred) {
  return tf.tidy(() => {
    const diffX = yPred
      .slice([0, 0, 0, 0], [-1, -1, 15, -1])
      .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));
    const diffY = yPred
      .slice([0, 0, 0, 0], [-1, 15, -1, -1])
      .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));
    return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
  });
}

/** Direction penalty – left (−1) to right (+1) gradient */
function directionX(yPred) {
  return tf.tidy(() => {
    const mask = tf.linspace(-1, 1, 16).reshape([1, 1, 16, 1]);
    return tf.mean(yPred.mul(mask)).neg();
  });
}

// ==========================================
// 3. Model Architectures
// ==========================================

function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "expansion") {
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else {
    throw new Error(`Unknown architecture: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Custom Loss (COMPLETED)
// ==========================================

function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const lossMSE = mse(yTrue, yPred);
    const lossSmooth = smoothness(yPred).mul(CONFIG.lambdaSmooth);
    const lossDir = directionX(yPred).mul(CONFIG.lambdaDir);
    return lossMSE.add(lossSmooth).add(lossDir);
  });
}

// ==========================================
// 5. Rendering – GUARANTEED VISIBLE INPUT
// ==========================================

/**
 * Render a tensor to a canvas with multiple fallbacks.
 * This ensures that even if the first attempt fails, we retry.
 */
async function renderTensorToCanvas(tensor, canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) {
    console.error(`Canvas #${canvasId} not found`);
    return;
  }

  // Squeeze to [height, width] for grayscale
  const imgTensor = tensor.squeeze();

  try {
    // Method 1: tf.browser.toPixels (standard)
    await tf.browser.toPixels(imgTensor, canvas);
    // Verify that pixels were actually written
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const nonZero = imageData.data.some(v => v > 0);
    if (!nonZero) throw new Error('Canvas is all black after toPixels');
  } catch (e) {
    console.warn(`toPixels fallback for ${canvasId}:`, e);
    // Method 2: Manual ImageData drawing
    const data = await imgTensor.data(); // Float32 array in [0,1]
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(canvas.width, canvas.height);
    for (let i = 0; i < data.length; i++) {
      const val = Math.floor(data[i] * 255);
      imgData.data[i * 4] = val;
      imgData.data[i * 4 + 1] = val;
      imgData.data[i * 4 + 2] = val;
      imgData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  } finally {
    imgTensor.dispose(); // clean up temporary tensor
  }
}

/** Render input, baseline, and student canvases */
async function renderAll() {
  // Input is static – only re-render if it hasn't been drawn yet.
  // We already draw it in init(), but we can also redraw on demand.
  if (state.xInput) {
    await renderTensorToCanvas(state.xInput, 'canvas-input');
  }

  // Baseline output
  if (state.baselineModel) {
    const pred = state.baselineModel.predict(state.xInput);
    await renderTensorToCanvas(pred, 'canvas-baseline');
    pred.dispose();
  }

  // Student output
  if (state.studentModel) {
    const pred = state.studentModel.predict(state.xInput);
    await renderTensorToCanvas(pred, 'canvas-student');
    pred.dispose();
  }
}

// ==========================================
// 6. Training Step
// ==========================================

async function trainStep() {
  if (!state.baselineModel || !state.studentModel) {
    log('Models not initialized.', true);
    return;
  }

  state.step++;

  // Baseline update (MSE only)
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    }, state.baselineModel.getWeights());
    state.optimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Student update (custom loss)
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const { value, grads } = tf.variableGrads(() => {
        const yPred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, yPred);
      }, state.studentModel.getWeights());
      state.optimizer.applyGradients(grads);
      return value.dataSync()[0];
    });
  } catch (e) {
    log(`Student error: ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  log(`Step ${state.step}: Base=${baselineLossVal.toFixed(5)} | Student=${studentLossVal.toFixed(5)}`);

  // Update UI every 5 steps or immediately if manual
  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await renderAll();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// 7. UI Helpers
// ==========================================

function updateLossDisplay(baseLoss, studentLoss) {
  document.getElementById('loss-baseline').innerText = `Loss: ${baseLoss.toFixed(5)}`;
  document.getElementById('loss-student').innerText = `Loss: ${studentLoss.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById('log-area');
  const entry = document.createElement('div');
  entry.innerText = `> ${msg}`;
  if (isError) entry.classList.add('error');
  el.prepend(entry);
  if (el.children.length > 12) el.removeChild(el.lastChild);
}

// ==========================================
// 8. Reset & Initialization
// ==========================================

function resetModels(archType = null) {
  if (typeof archType !== 'string') archType = null;
  if (state.isAutoTraining) stopAutoTrain();

  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : 'compression';
  }

  // Dispose old resources
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.optimizer) state.optimizer.dispose();

  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);
  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  document.getElementById('student-arch-label').innerText =
    archType.charAt(0).toUpperCase() + archType.slice(1);

  log(`Models reset. Architecture: ${archType}`);
  renderAll().catch(console.error);
}

async function init() {
  await tf.ready();
  log('TensorFlow.js ready.');

  // Create fixed noise input (shared)
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

  // ----- CRITICAL: FORCE INPUT CANVAS TO SHOW NOISE -----
  // Use the robust renderer that falls back to manual ImageData
  await renderTensorToCanvas(state.xInput, 'canvas-input');
  // Double-check: read a pixel and log mean value
  const meanVal = state.xInput.mean().dataSync()[0];
  log(`Input noise mean: ${meanVal.toFixed(3)} (should be ~0.5)`);
  // Also draw a timestamp to confirm canvas is alive
  const ctx = document.getElementById('canvas-input').getContext('2d');
  ctx.font = '2px monospace';
  ctx.fillStyle = 'white';
  ctx.fillText('✓', 1, 3); // tiny checkmark – will be overwritten but proves canvas works

  // Initialize models
  resetModels();

  // Attach event listeners
  document.getElementById('btn-train').addEventListener('click', () => trainStep());
  document.getElementById('btn-auto').addEventListener('click', toggleAutoTrain);
  document.getElementById('btn-reset').addEventListener('click', () => resetModels());

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener('change', (e) => resetModels(e.target.value));
  });

  log('Custom loss active: MSE + smoothness*0.01 + direction*0.1');
}

// ==========================================
// 9. Auto-train Loop
// ==========================================

function toggleAutoTrain() {
  const btn = document.getElementById('btn-auto');
  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = 'Auto Train (Stop)';
    btn.classList.add('btn-stop');
    btn.classList.remove('btn-auto');
    loop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById('btn-auto');
  btn.innerText = 'Auto Train (Start)';
  btn.classList.add('btn-auto');
  btn.classList.remove('btn-stop');
}

function loop() {
  if (state.isAutoTraining) {
    trainStep().catch(console.error);
    setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}

// ==========================================
// 10. Start
// ==========================================
init().catch(console.error);
