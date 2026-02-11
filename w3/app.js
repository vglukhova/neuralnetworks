/**
 * Neural Network Design: The Gradient Puzzle
 * ===========================================
 * COMPLETED SOLUTION
 * - Custom loss: MSE + smoothness (λ=0.01) + direction (λ=0.1)
 * - Three student architectures: Compression, Transformation, Expansion
 * - Input canvas now renders visible noise (async/await fixed)
 * - Memory leaks minimized via tidy/dispose
 */

// ==========================================
// 1. Global State & Config
// ==========================================
const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.05,
  autoTrainSpeed: 50,
  // Custom loss coefficients
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
  optimizer: null, // shared optimizer (works because variables are distinct)
};

// ==========================================
// 2. Loss Components (fully implemented)
// ==========================================

function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

/**
 * Smoothness penalty – Total Variation style.
 * Encourages local spatial smoothness.
 */
function smoothness(yPred) {
  return tf.tidy(() => {
    // Differences in X direction: pixel[i,j] - pixel[i,j+1]
    const diffX = yPred
      .slice([0, 0, 0, 0], [-1, -1, 15, -1])
      .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));
    // Differences in Y direction: pixel[i,j] - pixel[i+1,j]
    const diffY = yPred
      .slice([0, 0, 0, 0], [-1, 15, -1, -1])
      .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));
    // Return mean of squared differences (scalar)
    return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
  });
}

/**
 * Direction penalty – encourages left(-1) to right(+1) gradient.
 * L_dir = - mean(yPred * mask)  (negative sign to minimize => maximize correlation)
 */
function directionX(yPred) {
  return tf.tidy(() => {
    const mask = tf.linspace(-1, 1, 16).reshape([1, 1, 16, 1]); // [1,1,16,1]
    return tf.mean(yPred.mul(mask)).neg();
  });
}

// ==========================================
// 3. Model Architectures
// ==========================================

/** Baseline: fixed compression autoencoder (MSE only) */
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" })); // bottleneck
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

/** Student: three projection architectures – fully implemented */
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    // Bottleneck: compress to 64, then expand
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    // Same dimension mapping: hidden size equals input size (256)
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "expansion") {
    // Overcomplete: expand to 512, then project back
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else {
    throw new Error(`Unknown architecture: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Custom Loss Function (COMPLETED)
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
// 5. Training Step
// ==========================================

async function trainStep() {
  // Safety checks
  if (!state.baselineModel || !state.studentModel) {
    log("Models not initialized. Call resetModels() first.", true);
    return;
  }

  state.step++;

  // ----- Baseline update (MSE only) -----
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    }, state.baselineModel.getWeights());

    state.optimizer.applyGradients(grads);
    return value.dataSync()[0]; // scalar
  });

  // ----- Student update (custom loss) -----
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
    log(`Student training error: ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  // Logging
  log(
    `Step ${state.step}: Baseline loss = ${baselineLossVal.toFixed(5)} | Student loss = ${studentLossVal.toFixed(5)}`
  );

  // Update UI every 5 steps or immediately if manual step
  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// 6. Rendering & UI Helpers
// ==========================================

/** Render all three canvases (input is static, rendered once in init) */
async function render() {
  // Baseline output
  const basePred = state.baselineModel.predict(state.xInput);
  await tf.browser.toPixels(
    basePred.squeeze(),
    document.getElementById("canvas-baseline")
  );
  basePred.dispose();

  // Student output
  const studPred = state.studentModel.predict(state.xInput);
  await tf.browser.toPixels(
    studPred.squeeze(),
    document.getElementById("canvas-student")
  );
  studPred.dispose();
}

function updateLossDisplay(baseLoss, studentLoss) {
  document.getElementById("loss-baseline").innerText = `Loss: ${baseLoss.toFixed(5)}`;
  document.getElementById("loss-student").innerText = `Loss: ${studentLoss.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
  const entry = document.createElement("div");
  entry.innerText = `> ${msg}`;
  if (isError) entry.classList.add("error");
  el.prepend(entry);
  // Keep log area from growing too large
  if (el.children.length > 12) el.removeChild(el.lastChild);
}

// ==========================================
// 7. Reset & Initialization
// ==========================================

function resetModels(archType = null) {
  // If called from event, archType may be Event object -> normalize
  if (typeof archType !== "string") archType = null;

  // Stop auto-training to avoid race conditions
  if (state.isAutoTraining) stopAutoTrain();

  // Get current selected architecture if not specified
  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  // Dispose old resources
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.optimizer) state.optimizer.dispose();

  // Create fresh models
  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);
  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  // Update UI label
  document.getElementById("student-arch-label").innerText =
    archType.charAt(0).toUpperCase() + archType.slice(1);

  log(`Models reset. Student architecture: ${archType}`);
  // Re-render (input is already drawn, now baseline/student will update)
  render().catch(console.error);
}

async function init() {
  // Ensure TensorFlow.js is ready
  await tf.ready();
  log("TensorFlow.js ready.");

  // Fixed noise input (shared for all models)
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

  // --- CRITICAL FIX: Await the pixel rendering so input appears ---
  await tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input")
  );
  log("Input noise rendered.");

  // Initialize models
  resetModels();

  // Attach event listeners
  document.getElementById("btn-train").addEventListener("click", () => trainStep());
  document.getElementById("btn-auto").addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", () => resetModels());

  // Architecture radio buttons
  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      resetModels(e.target.value);
    });
  });

  log("Custom loss active: MSE + smoothness*0.01 + direction*0.1");
}

// ==========================================
// 8. Auto‑train Loop
// ==========================================

function toggleAutoTrain() {
  const btn = document.getElementById("btn-auto");
  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = "Auto Train (Stop)";
    btn.classList.add("btn-stop");
    btn.classList.remove("btn-auto");
    loop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById("btn-auto");
  btn.innerText = "Auto Train (Start)";
  btn.classList.add("btn-auto");
  btn.classList.remove("btn-stop");
}

function loop() {
  if (state.isAutoTraining) {
    trainStep().catch(console.error);
    setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}

// ==========================================
// 9. Start Everything
// ==========================================
init().catch(console.error);
