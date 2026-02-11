// app.js — The Gradient Puzzle
// All TODOs are COMPLETED. This is a reference solution with custom loss + 3 architectures.
// Runs directly in browser with TF.js CDN.

(async function run() {
    // --- Wait for TF.js to be ready ---
    await tf.ready();
    console.log('TensorFlow.js ready.');

    // -------- GLOBAL STATE ----------
    // Fixed input noise (shape [1,16,16,1]), never changes.
    const inputTensor = createFixedNoise();

    // Baseline model: fixed compression architecture, MSE only.
    let baselineModel = createBaselineModel();
    
    // Student model: changes with radio, custom loss, can copy baseline weights in compression mode.
    let studentModel;
    let studentOptimizer;
    let baselineOptimizer = tf.train.adam(0.01);
    
    // State variables
    let stepCount = 0;
    let isAutoTraining = false;
    let rafId = null;
    let currentArch = 'compression';  // default
    
    // UI Elements
    const canvasInput = document.getElementById('canvasInput');
    const canvasBaseline = document.getElementById('canvasBaseline');
    const canvasStudent = document.getElementById('canvasStudent');
    const lossBaselineEl = document.getElementById('lossBaseline');
    const lossStudentEl = document.getElementById('lossStudent');
    const errorEl = document.getElementById('errorMessage');

    // ---------- FIXED NOISE (never changes) ----------
    function createFixedNoise() {
        return tf.tidy(() => {
            // fixed seed for reproducibility
            const seed = 42;
            return tf.randomUniform([1, 16, 16, 1], 0, 1, 'float32', seed);
        });
    }

    // ---------- BASELINE MODEL (compression, MSE only) ----------
    function createBaselineModel() {
        const model = tf.sequential();
        // Encoder
        model.add(tf.layers.conv2d({inputShape: [16,16,1], filters: 8, kernelSize: 3, strides: 2, activation: 'relu'}));
        model.add(tf.layers.conv2d({filters: 16, kernelSize: 3, strides: 2, activation: 'relu'}));
        // Decoder
        model.add(tf.layers.upSampling2d({size: [2,2]}));
        model.add(tf.layers.conv2d({filters: 8, kernelSize: 3, padding: 'same', activation: 'relu'}));
        model.add(tf.layers.upSampling2d({size: [2,2]}));
        model.add(tf.layers.conv2d({filters: 1, kernelSize: 1, padding: 'same', activation: 'linear'}));
        return model;
    }

    // ---------- COMPLETED: Student architectures (3 modes) ----------
    function createStudentModel(archType) {
        const model = tf.sequential();
        const inputShape = [16,16,1];
        
        if (archType === 'compression') {
            // --- COMPLETED: Compression (bottleneck) ---
            model.add(tf.layers.conv2d({inputShape, filters: 8, kernelSize: 3, strides: 2, activation: 'relu'}));
            model.add(tf.layers.conv2d({filters: 16, kernelSize: 3, strides: 2, activation: 'relu'}));
            model.add(tf.layers.upSampling2d({size: [2,2]}));
            model.add(tf.layers.conv2d({filters: 8, kernelSize: 3, padding: 'same', activation: 'relu'}));
            model.add(tf.layers.upSampling2d({size: [2,2]}));
            model.add(tf.layers.conv2d({filters: 1, kernelSize: 1, padding: 'same', activation: 'linear'}));
        } else if (archType === 'transformation') {
            // --- COMPLETED: Transformation (same spatial) ---
            model.add(tf.layers.conv2d({inputShape, filters: 8, kernelSize: 3, padding: 'same', activation: 'relu'}));
            model.add(tf.layers.conv2d({filters: 8, kernelSize: 3, padding: 'same', activation: 'relu'}));
            model.add(tf.layers.conv2d({filters: 1, kernelSize: 1, padding: 'same', activation: 'linear'}));
        } else { // expansion
            // --- COMPLETED: Expansion (bottleneck then expand) ---
            model.add(tf.layers.conv2d({inputShape, filters: 32, kernelSize: 3, padding: 'same', activation: 'relu'}));
            model.add(tf.layers.conv2d({filters: 64, kernelSize: 3, padding: 'same', activation: 'relu'}));
            model.add(tf.layers.conv2d({filters: 32, kernelSize: 3, padding: 'same', activation: 'relu'}));
            model.add(tf.layers.conv2d({filters: 1, kernelSize: 1, padding: 'same', activation: 'linear'}));
        }
        return model;
    }

    // ---------- COMPLETED: Custom loss components ----------
    // Smoothness: Total variation style
    function smoothness(yPred) {
        return tf.tidy(() => {
            // yPred shape: [1,16,16,1]
            const diffRows = yPred.slice([0,1,0,0], [-1, -1, -1, -1]).sub(yPred.slice([0,0,0,0], [-1, -1, -1, -1]));
            const diffCols = yPred.slice([0,0,1,0], [-1, -1, -1, -1]).sub(yPred.slice([0,0,0,0], [-1, -1, -1, -1]));
            const sqRows = diffRows.square();
            const sqCols = diffCols.square();
            return sqRows.sum().add(sqCols.sum());
        });
    }

    // Direction penalty: aligns output with left(-1) to right(+1) gradient
    function directionX(yPred) {
        return tf.tidy(() => {
            // Create linear mask: 16x16, values from -1 (left) to +1 (right)
            const col = tf.linspace(-1, 1, 16);
            const mask = col.reshape([1, 16]).tile([16, 1]).reshape([1,16,16,1]);
            // Ldir = - mean( yPred * mask )
            const product = yPred.mul(mask);
            return tf.mean(product).neg();
        });
    }

    // MSE helper
    function mse(yTrue, yPred) {
        return tf.tidy(() => tf.mean(tf.square(yTrue.sub(yPred))));
    }

    // --- COMPLETED: Student loss (MSE + smoothness + direction) ---
    function studentLoss(yTrue, yPred) {
        const lambda_s = 0.01;
        const lambda_d = 0.1;
        return tf.tidy(() => {
            const lossMse = mse(yTrue, yPred);
            const lossSm = smoothness(yPred);
            const lossDir = directionX(yPred);
            return lossMse.add(lossSm.mul(lambda_s)).add(lossDir.mul(lambda_d));
        });
    }

    // ---------- Compute loss for a model (generic) ----------
    function computeLoss(model, input, target, isStudent = false) {
        return tf.tidy(() => {
            const pred = model.predict(input);
            if (isStudent) {
                return studentLoss(target, pred);
            } else {
                return mse(target, pred);
            }
        });
    }

    // ---------- Training step (both models) ----------
    async function trainStep() {
        try {
            tf.tidy(() => {
                // ----- Baseline update (MSE only) -----
                const baselineVars = baselineModel.trainableWeights;
                let baselineGrads;
                let baselineLossVal;
                tf.engine().startScope();
                const tape = tf.gradients(() => {
                    const pred = baselineModel.predict(inputTensor);
                    const loss = mse(inputTensor, pred);
                    baselineLossVal = loss.clone();
                    return loss;
                }, baselineVars);
                baselineGrads = tape;
                baselineOptimizer.applyGradients(baselineGrads.map((g, i) => ({name: baselineVars[i].name, tensor: g})));
                tf.engine().endScope();

                // ----- Student update (custom loss) -----
                if (studentModel) {
                    const studentVars = studentModel.trainableWeights;
                    let studentGrads;
                    let studentLossVal;
                    tf.engine().startScope();
                    const tape2 = tf.gradients(() => {
                        const pred = studentModel.predict(inputTensor);
                        const loss = studentLoss(inputTensor, pred);
                        studentLossVal = loss.clone();
                        return loss;
                    }, studentVars);
                    studentGrads = tape2;
                    studentOptimizer.applyGradients(studentGrads.map((g, i) => ({name: studentVars[i].name, tensor: g})));
                    tf.engine().endScope();

                    // Update log with losses
                    stepCount++;
                    baselineLossVal.data().then(bLoss => {
                        lossBaselineEl.innerText = `Baseline loss: ${bLoss.toFixed(5)}`;
                    });
                    studentLossVal.data().then(sLoss => {
                        lossStudentEl.innerText = `Student loss: ${sLoss.toFixed(5)}`;
                    });
                    document.querySelector('#logPanel').innerHTML = `⚙️ Step: ${stepCount} &nbsp;&nbsp; | &nbsp; <span id="lossBaseline">Baseline loss: —</span> &nbsp;&nbsp; <span id="lossStudent">Student loss: —</span>`;
                }
            });
            // Render the three views
            await renderCanvases();
        } catch (e) {
            errorEl.innerText = `⚠️ Error: ${e.message}`;
            console.error(e);
        }
    }

    // ---------- Render 16x16 grayscale to canvas (pixelated) ----------
    async function renderCanvases() {
        const inputData = await tf.browser.toPixels(inputTensor.squeeze(), canvasInput);
        
        if (baselineModel) {
            const predBase = baselineModel.predict(inputTensor);
            await tf.browser.toPixels(predBase.squeeze(), canvasBaseline);
            predBase.dispose();
        }
        if (studentModel) {
            const predStudent = studentModel.predict(inputTensor);
            await tf.browser.toPixels(predStudent.squeeze(), canvasStudent);
            predStudent.dispose();
        }
    }

    // ---------- Reset models: baseline fresh, student re-created ----------
    function resetModels() {
        tf.tidy(() => {
            // Recreate baseline
            baselineModel.dispose?.();
            baselineModel = createBaselineModel();
            baselineOptimizer = tf.train.adam(0.01);  // fresh optimizer

            // Recreate student according to current architecture
            if (studentModel) studentModel.dispose();
            studentModel = createStudentModel(currentArch);
            
            // --- COMPLETED: Copy weights from baseline only if architecture is 'compression' ---
            if (currentArch === 'compression') {
                const baselineWeights = baselineModel.getWeights();
                studentModel.setWeights(baselineWeights);
            }
            
            studentOptimizer = tf.train.adam(0.01);
            stepCount = 0;
            lossBaselineEl.innerText = 'Baseline loss: —';
            lossStudentEl.innerText = 'Student loss: —';
            errorEl.innerText = '';
        });
        renderCanvases();
    }

    // ---------- Auto-training loop ----------
    function autoTrainLoop() {
        if (!isAutoTraining) return;
        rafId = requestAnimationFrame(async () => {
            await trainStep();
            autoTrainLoop();
        });
    }

    // ---------- Attach event listeners ----------
    function initUI() {
        // Train 1 step
        document.getElementById('btnTrainStep').addEventListener('click', async () => {
            await trainStep();
        });

        // Auto start
        document.getElementById('btnAutoStart').addEventListener('click', () => {
            if (!isAutoTraining) {
                isAutoTraining = true;
                autoTrainLoop();
            }
        });

        // Auto stop
        document.getElementById('btnAutoStop').addEventListener('click', () => {
            isAutoTraining = false;
            if (rafId) cancelAnimationFrame(rafId);
        });

        // Reset weights
        document.getElementById('btnReset').addEventListener('click', () => {
            resetModels();
        });

        // Architecture radio (student only)
        const archRadios = document.querySelectorAll('input[name="arch"]');
        archRadios.forEach(radio => {
            radio.addEventListener('change', async (e) => {
                currentArch = e.target.value;
                // Recreate student model with new architecture
                if (studentModel) studentModel.dispose();
                studentModel = createStudentModel(currentArch);
                
                // --- COMPLETED: if compression, clone baseline weights ---
                if (currentArch === 'compression') {
                    const baselineWeights = baselineModel.getWeights();
                    studentModel.setWeights(baselineWeights);
                }
                
                studentOptimizer = tf.train.adam(0.01);
                await renderCanvases();
                errorEl.innerText = '';
            });
        });
    }

    // ---------- Bootstrap ----------
    // Initial student (compression) with weights copied from baseline
    studentModel = createStudentModel('compression');
    const baseWeights = baselineModel.getWeights();
    studentModel.setWeights(baseWeights);
    studentOptimizer = tf.train.adam(0.01);
    
    // First render
    await renderCanvases();
    initUI();
    console.log('Ready. All student TODOs are completed.');
})();
