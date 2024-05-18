/*
Hand Tracking + KNN Classifier

The example lets you train a knn algorithm to classify handposes

Built with handPose model from tf.js, knn-classifier from ml5js
*/

let model, video, keypoints,canvas,selectedType,  predictions = [];
// Create a KNN classifier
const classifier = knnClassifier.create();
const LABELS_MAP = {
  'Closed': 0,
  'Open': 1,
}

function preload() {
  video = createCapture(VIDEO, () => {
    loadHandTrackingModel();
  });
  video.hide();
  // Create the UI buttons
  createButtons();
}

function setup() {
  canvas = createCanvas(600, 400);
  canvas.parent('canvasContainer');
}

async function loadHandTrackingModel() {
  // Load the MediaPipe handpose model.
  model = await handpose.load();
  select('#status').html('Hand Tracking Model Loaded')
  predictHand();
}

function draw() {
  background(255);
  if (model) image(video, 0, 0);
  if (predictions.length > 0) {
    drawKeypoints();
    drawSkeleton();

    textSize(156);
    if(selectedType === "Closed"){
      text("✊", 0,height/3);
    } else if (selectedType === "Open"){
      text("🖐", 0,height/3);
    }
  }
}

async function predictHand() {
  // Pass in a video stream (or an image, canvas, or 3D tensor) to obtain a hand prediction from the MediaPipe graph.
  predictions = await model.estimateHands(video.elt);

  setTimeout(() => predictHand(), 100);
}

// Add the current hand tracking data to the classifier
function addExample(label) {
  if (predictions.length > 0) {
    const features = predictions[0].landmarks;
    const tensors = tf.tensor(features)
    // Add an example with a label to the classifier
    classifier.addExample(tensors, label);
    updateCounts();
  } else {
    console.log('No gesture is detected')
  }
}

// Predict the current frame.
async function classify() {
  // Get the total number of labels from classifier
  const numLabels = classifier.getNumClasses();
  if (numLabels <= 0) {
    console.error('There is no examples in any label');
    return;
  }
  if (predictions.length > 0) {
    const results = await classifier.predictClass(tf.tensor(predictions[0].landmarks));
    if (results.confidences) {
      const confidences = results.confidences;
      // result.label is the label that has the highest confidence
      if (results.label) {
        selectedType = results.label
        select('#result').html(results.label);
        select('#confidence').html(`${confidences[results.label] * 100} %`);
        
      }

      select('#confidenceClosed').html(`${confidences['Closed'] ? confidences['Closed'] * 100 : 0} %`);
      select('#confidenceOpen').html(`${confidences['Open'] ? confidences['Open'] * 100 : 0} %`);
    }
    classify();
  } else {
    setTimeout(() => classify(), 1000);
  }
}

// Update the example count for each label	
function updateCounts() {
  const counts = classifier.getClassExampleCount();

  select('#exampleClosed').html(counts['Closed'] || 0);
  select('#exampleOpen').html(counts['Open'] || 0);
}

// Clear the examples in one label
function clearLabel(label) {
  classifier.clearClass(label);
  updateCounts();
}

// Clear all the examples in all labels
function clearAllLabels() {
  classifier.clearAllClasses();
  updateCounts();
}

// A util function to create UI buttons
function createButtons() {
  // When the A button is pressed, add the current frame
  // from the video with a label of "closed" to the classifier
  buttonA = select('#addClassClosed');
  buttonA.mousePressed(function() {
    addExample('Closed');
  });

  // When the B button is pressed, add the current frame
  // from the video with a label of "open" to the classifier
  buttonB = select('#addClassOpen');
  buttonB.mousePressed(function() {
    addExample('Open');
  });

  // Reset buttons
  resetBtnA = select('#resetClosed');
  resetBtnA.mousePressed(function() {
    clearLabel('Closed');
  });

  resetBtnB = select('#resetOpen');
  resetBtnB.mousePressed(function() {
    clearLabel('Open');
  });


  // Predict button
  buttonPredict = select('#buttonPredict');
  buttonPredict.mousePressed(classify);

  // Clear all classes button
  buttonClearAll = select('#clearAll');
  buttonClearAll.mousePressed(clearAllLabels);
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  let prediction = predictions[0];
  for (let j = 0; j < prediction.landmarks.length; j++) {
    let keypoint = prediction.landmarks[j];
    fill(255, 0, 0);
    noStroke();
    ellipse(keypoint[0], keypoint[1], 10, 10);
  }
}

// A function to draw the skeletons
function drawSkeleton() {
  let annotations = predictions[0].annotations;
  stroke(255, 0, 0);
  for (let j = 0; j < annotations.thumb.length - 1; j++) {
    line(annotations.thumb[j][0], annotations.thumb[j][1], annotations.thumb[j + 1][0], annotations.thumb[j + 1][1]);
  }
  for (let j = 0; j < annotations.indexFinger.length - 1; j++) {
    line(annotations.indexFinger[j][0], annotations.indexFinger[j][1], annotations.indexFinger[j + 1][0], annotations.indexFinger[j + 1][1]);
  }
  for (let j = 0; j < annotations.middleFinger.length - 1; j++) {
    line(annotations.middleFinger[j][0], annotations.middleFinger[j][1], annotations.middleFinger[j + 1][0], annotations.middleFinger[j + 1][1]);
  }
  for (let j = 0; j < annotations.ringFinger.length - 1; j++) {
    line(annotations.ringFinger[j][0], annotations.ringFinger[j][1], annotations.ringFinger[j + 1][0], annotations.ringFinger[j + 1][1]);
  }
  for (let j = 0; j < annotations.pinky.length - 1; j++) {
    line(annotations.pinky[j][0], annotations.pinky[j][1], annotations.pinky[j + 1][0], annotations.pinky[j + 1][1]);
  }

  line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.thumb[0][0], annotations.thumb[0][1]);
  line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.indexFinger[0][0], annotations.indexFinger[0][1]);
  line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.middleFinger[0][0], annotations.middleFinger[0][1]);
  line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.ringFinger[0][0], annotations.ringFinger[0][1]);
  line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.pinky[0][0], annotations.pinky[0][1]);
}