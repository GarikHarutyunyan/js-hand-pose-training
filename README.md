# Hand Pose Training with TensorFlow.js and ml5.js

This project demonstrates a simple example of hand pose recognition using TensorFlow.js and ml5.js. The application utilizes TensorFlow's HandPose model to detect hand movements through the device's camera. Users can train the model to recognize open and closed hand poses using the KNN classifier algorithm from ml5.js.

---

🚀 **Hosted Application:** [Hand Pose Training](https://garikharutyunyan.github.io/js-hand-pose-training/)

---

## Usage

1. Clone the repository to your local machine:
git clone https://github.com/GarikHarutyunyan/js-hand-pose-training.git


2. Open `index.html` in your web browser.

3. Allow access to your device's camera when prompted.

4. Choose between the "Open Hand" and "Closed Hand" options to train the model by clicking on the respective buttons and performing the corresponding hand poses.

5. After training, click on the "Start Prediction" button to begin hand pose recognition.

6. Hold your hand in a pose, and the system will predict whether your hand is open or closed based on the trained model.

## Files

- `index.html`: The HTML file containing the interface and functionality.
- `script.js`: The JavaScript file implementing the logic for hand pose recognition and training.

## Requirements

- Web browser with camera access support.
- Internet connection for loading TensorFlow.js and ml5.js libraries.

## Technologies Used

- TensorFlow.js: A JavaScript library for training and deploying machine learning models in the browser.
- ml5.js: A friendly machine learning library for the web, built on top of TensorFlow.js.
- HTML, and JavaScript: Standard web technologies for building user interfaces and implementing functionality.

## Credits

- This project utilizes the HandPose model from TensorFlow.js and the KNN classifier algorithm from ml5.js.
- Developed by Garik Harutyunyan.