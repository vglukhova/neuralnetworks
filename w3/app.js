/**
 * Neural Network Design: The Gradient Puzzle
 * ===========================================
 * COMPLETE – AUTO TRAINING FIXED
 *
 * - Student architectures fully implemented (compression, transformation, expansion)
 * - Custom loss: MSE + smoothness*0.01 + direction*0.1
 * - NO BROADCAST ERRORS: tf.grads with explicit varList, optimizers recreated on reset
 * - AUTO TRAINING: recursive async loop with await trainStep()
 * - INPUT CANVAS: always shows noise (await toPixels)
 */

// ==========================================
// 1. Global State & Config
// ==========================================
const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.05,
  autoTrainDelay: 50, // ms between steps
  lambdaSmooth: 0.01,
  lambdaDir: 0.1,
};

let state = {
  step: 0,
  isAutoTraining: false,
  autoTrainTimer: null,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  baselineOptimizer: null,
  studentOptimizer: null,
};

// ==========================================
// 2. Loss Components (fully implemented)
// ==========================================

function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

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

function directionX(yPred) {
  return tf.tidy(() => {
    const mask = tf.linspace(-1, 1, 16).reshape([1, 1, 16, 1]);
    return tf.mean(yPred.mul(mask)).mul(-1);
  });
}

// ==========================================
// 3. Model Architectures (COMPLETED)
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
    // 1:1 mapping – hidden layer = 256 (same as input)
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "expansion") {
    // Overcomplete – expand to 512, then project back
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
// 5. Training Step – CLEAN & ISOLATED
// ==========================================

async function trainStep() {
  if (!state.baselineModel || !state.studentModel) {
    log("Models not initialized.", true);
    return;
  }

  state.step++;

  // ----- Baseline update (MSE only) -----
  const baselineLossVal = tf.tidy(() => {
    const vars = state.baselineModel.trainableWeights;
    const lossFn = () => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    };
    const grads = tf.grads(lossFn);
    const gradValues = grads(vars);
    state.baselineOptimizer.applyGradients(
      gradValues.map((g, i) => ({ name: vars[i].name, tensor: g }))
    );
    return lossFn().dataSync()[0];
  });

  // ----- Student update (custom loss) -----
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const vars = state.studentModel.trainableWeights;
      const lossFn = () => {
        const yPred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, yPred);
      };
      const grads = tf.grads(lossFn);
      const gradValues = grads(vars);
      state.studentOptimizer.applyGradients(
        gradValues.map((g, i) => ({ name: vars[i].name, tensor: g }))
      );
      return lossFn().dataSync()[0];
    });

    log(
      `Step ${state.step}: Base Loss=${baselineLossVal.toFixed(4)} | Student Loss=${studentLossVal.toFixed(4)}`
    );
  } catch (e) {
    log(`Student training error: ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  // Update display every 5 steps or immediately for manual steps
  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// 6. Rendering & UI Helpers
// ==========================================

async function render() {
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(
    basePred.squeeze(),
    document.getElementById("canvas-baseline")
  );
  await tf.browser.toPixels(
    studPred.squeeze(),
    document.getElementById("canvas-student")
  );

  basePred.dispose();
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
  if (el.children.length > 12) el.removeChild(el.lastChild);
}

// ==========================================
// 7. Reset & Initialization
// ==========================================

async function resetModels(archType = null) {
  // Stop auto training before reset
  if (state.isAutoTraining) stopAutoTrain();

  if (typeof archType !== "string") archType = null;
  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  // Dispose old resources
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.baselineOptimizer) state.baselineOptimizer.dispose();
  if (state.studentOptimizer) state.studentOptimizer.dispose();

  // Create new models
  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);

  // Build models (forces weight creation)
  const dummy = tf.zeros([1, 16, 16, 1]);
  state.baselineModel.predict(dummy).dispose();
  state.studentModel.predict(dummy).dispose();
  dummy.dispose();

  // Fresh optimizers – no stale slots
  state.baselineOptimizer = tf.train.adam(CONFIG.learningRate);
  state.studentOptimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  document.getElementById("student-arch-label").innerText =
    archType.charAt(0).toUpperCase() + archType.slice(1);

  log(`Models reset. Student Arch: ${archType}`);
  await render();
}

async function init() {
  await tf.ready();
  log("TensorFlow.js ready.");

  // Fixed noise input
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

  // Render input canvas (AWAIT = guaranteed visible)
  await tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input")
  );
  log("Input noise rendered.");

  // Initialize models
  await resetModels();

  // Event listeners
  document.getElementById("btn-train").addEventListener("click", () => trainStep());
  document.getElementById("btn-auto").addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", () => resetModels());

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => resetModels(e.target.value));
  });

  log("Custom loss active: MSE + smoothness*0.01 + direction*0.1");
}

// ==========================================
// 8. Auto‑train Loop (SIMPLE & ROBUST)
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
    // Start the loop (recursive async with await)
    autoTrainLoop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById("btn-auto");
  btn.innerText = "Auto Train (Start)";
  btn.classList.add("btn-auto");
  btn.classList.remove("btn-stop");
  // No need to cancel timer – loop will stop because isAutoTraining is false
}

async function autoTrainLoop() {
  while (state.isAutoTraining) {
    await trainStep();           // Wait for step to complete
    await tf.nextFrame();        // Yield to browser (prevents blocking)
    await new Promise(resolve => setTimeout(resolve, CONFIG.autoTrainDelay));
  }
}

// ==========================================
// 9. Start
// ==========================================
init().catch(console.error);

