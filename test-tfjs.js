const tf = require('@tensorflow/tfjs');
// Load the tfjs-node binding with TensorFlow C++ to speed things up dramatically
//require('@tensorflow/tfjs-node');  // Use '@tensorflow/tfjs-node-gpu'
const loadFrozenModel = require('@tensorflow/tfjs-converter').loadFrozenModel;

global.fetch = require('node-fetch');

const MODEL_URL = 'https://raw.githubusercontent.com/AlbertZheng/quickdraw-cnn/master/web-model/tensorflowjs_model.pb';
const WEIGHTS_URL = 'https://raw.githubusercontent.com/AlbertZheng/quickdraw-cnn/master/web-model/weights_manifest.json';


async function loadModelAndPredict() {
  try {
    console.log("### Loading model... ###");
    let model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL);
    console.log("### Model loaded. ###");

    let ones = tf.ones([64, 28, 28, 1]);
    console.log("### Predicting... ###");
    // Refer to https://js.tensorflow.org/api/0.13.0/#tf.Model.predict
    //
    //model.predict(ones, {batchSize: 15, verbose: true}).print(true);
    // Or
    model.predict(ones).print(true);
    // Or
    //model.execute(ones).print(true);

    /*< Expected output will look like as below if the number of the trained categories is 10 (rank):
          Tensor
            dtype: float32
            rank: 2
            shape: [1,10]
            values:
               [[0.110344, 0.0668225, 0.0469237, 0.1852526, 0.0312816, 0.2039736, 0.1091306, 0.0362919, 0.0720466, 0.1379328],]
     */
  } catch(err) {
    console.log(err);
  }
}

loadModelAndPredict();
