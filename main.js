// UI Config
const PADDING = 10;
const TITLE_TEXT_SIZE = 24;
const DIGIT_UI_SIZE = 400;
var DIGIT_UI = null;
var PREDICTION_UI = null;
var PROGRESS_UI = null;

// ML Config
const EPOCHS = 1;
const BATCH_SIZE = 320;
const VALIDATION_SPLIT = 0.15;
var MODEL = null;
var DATA = null;

/*************** MACHINE LEARNING  ***********/

/**
 * Loads MNIST data and parses into Tensors
 */
async function loadData() {
  PROGRESS_UI.setStatus(`Loading...`);
  DATA = new MnistData();
  await DATA.load();
}

/**
 * Create a Convolutional Neural Network
 */
function createConvModel() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_H, IMAGE_W, 1],
    kernelSize: 3,
    filters: 16,
    activation: "relu"
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" }));
  model.add(tf.layers.flatten({}));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));
  return model;
}

/**
 * Create a Dense Neural Network
 */

function createDenseModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [IMAGE_H, IMAGE_W, 1] }));
  model.add(tf.layers.dense({ units: 42, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));
  return model;
}

/**
 * Train the model with the training data
 */
async function trainModel() {
  MODEL = createConvModel();
  MODEL.compile({
    optimizer: "rmsprop",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  const { xs, labels } = DATA.getTrainData();
  console.log("Training...");

  let trainBatchCount = 0;
  const totalNumBatches = Math.ceil((xs.shape[0] * (1 - VALIDATION_SPLIT)) / BATCH_SIZE) * EPOCHS;

  await MODEL.fit(xs, labels, { 
    batchSize: BATCH_SIZE, 
    validationSplit: VALIDATION_SPLIT, 
    epochs: EPOCHS,
    callbacks: {
      onBatchEnd: async (batch, { acc }) => {
        trainBatchCount++;
        let percentComplete = ((trainBatchCount / totalNumBatches) * 100).toFixed(1);
        PROGRESS_UI.setProgress(percentComplete);
        PROGRESS_UI.setStatus(`ACC ${(acc * 100).toFixed(1)}%`);
        console.log(`Training... (${percentComplete}% complete)`);
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, { val_acc }) => {
        console.log(`Accuracy: ${val_acc}`);
        PROGRESS_UI.setStatus(`ACC ${(val_acc * 100).toFixed(1)}%`);
        await tf.nextFrame();
      }
    }
  });
  console.log("Training Finished!");
  
  const {xs: testXs, labels: testLabels } = DATA.getTestData();
  const testResult = MODEL.evaluate(testXs, testLabels);
  const testAccPercent = testResult[1].dataSync()[0] * 100;
  console.log(`Final test accuracy: ${testAccPercent.toFixed(1)}%`);

}

function inferModel(data) {
  let inputs = tf.tensor4d(data, [1, 28, 28, 1]);
  const output = MODEL.predict(inputs);
  output.print();
  const distribution = output.dataSync();
  const axis = 1;
  const prediction = Array.from(output.argMax(axis).dataSync())[0];
  inputs.dispose();
  output.dispose();
  return { prediction, distribution };

}

async function loadAndTrain() {
  await loadData();
  await trainModel();
}

/*************** USER INTERFACE ***********/
async function setup() {
  console.log("👉 Setup");
  setupCanvas();
}

function resetDigitCanvas() {
  DIGIT_UI.background(0);
  DIGIT_UI.fill(255);
  DIGIT_UI.stroke(255);
  PREDICTION_UI.reset();
}

function setupCanvas() {
  textFont("Neucha", 16);
  frameRate(60);
  createCanvas(windowWidth, windowHeight);
  // Handling issues with retina screens, forcce pixel density to 1
  pixelDensity(1);

  PREDICTION_UI = new Preditions();
  PROGRESS_UI = new ProgressBar(110);

  // This is a place to store where the user is drawing
  DIGIT_UI = createGraphics(DIGIT_UI_SIZE, DIGIT_UI_SIZE);
  resetDigitCanvas();

  // Setup the buttons
  var trainBtn = createButton("Train");
  trainBtn.class("btn-secondary btn-small");
  trainBtn.position(100, 5);
  trainBtn.mousePressed(loadAndTrain);

  var checkBtn = createButton("Check");
  checkBtn.class("btn-success btn-small");
  checkBtn.position(100 + 60, 5);
  checkBtn.mousePressed(predictDigit);

  var resetBtn = createButton("Reset");
  resetBtn.class("btn-danger btn-small");
  resetBtn.position(100 + 127, 5);
  resetBtn.mousePressed(resetDigitCanvas);
}

function draw() {
  background(50);

  let Y = PADDING;

  // Draw Title
  fill(255)
    .strokeWeight(0)
    .textSize(16)
    .textFont("Neucha", 24);
  text("MNIST", PADDING + 5, Y + 24);

  // Draw Progress
  push();
  translate(300, Y);
  PROGRESS_UI.draw();
  pop();

  Y = PADDING + 24 + PADDING * 2;

  // Draw Digit
  image(DIGIT_UI, PADDING, Y);
  Y = Y + DIGIT_UI_SIZE + PADDING;

  // Draw predictions chart
  push();
  translate(0, Y);
  PREDICTION_UI.draw();
  pop();
}

// When the mouse is dragged, draw onto the user pixels
function touchMoved() {
  // Only if the user drags within the user pixels area

  const x = PADDING;
  const y = PADDING + 24 + PADDING;
  const w = DIGIT_UI_SIZE;

  if (mouseX > x && mouseY > y && mouseX < x + w && mouseY < y + w) {
    // Draw a white circle
    DIGIT_UI.ellipse(mouseX - x, mouseY - y, 24, 24);
  }
}

/**
 * Takes the image data that you drew and tries to predict the
 */
async function predictDigit() {
  // Copy the digit canvas into a 28 by 28 image
  let inputs = [];
  const smaller = createImage(28, 28, RGB);
  const img = DIGIT_UI.get();
  DIGIT_UI.width;
  smaller.copy(
    img,
    0,
    0,
    DIGIT_UI.width,
    DIGIT_UI.height,

    0,
    0,
    smaller.width,
    smaller.height
  );
  // Get an array representing the smaller image
  smaller.loadPixels();
  for (var i = 0; i < smaller.pixels.length; i += 4) {
    // Just using the red channel since it's a greyscale image
    // Not so great to use inputs of 0 so smallest value is 0.01
    inputs[i / 4] = map(smaller.pixels[i], 0, 255, 0, 0.99) + 0.01;
  }
  // Get predictions based on that image
  let data = inferModel(inputs);
  PREDICTION_UI.setData(data);
  console.log(data);
}
