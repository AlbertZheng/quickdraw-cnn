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
    model.predict(ones, {batchSize: 15, verbose: true}).print(true);
    // Or
    //model.predict(ones).print(true);
    // Or
    //model.execute(ones).print(true);

    /*< Expected output will like as below if the number of trained Categories is 2 (rank):
        Tensor
          dtype: float32
          rank: 2
          shape: [128,2]
          values:
            [[0.5204441, 0.4795559],
             [0.5204441, 0.4795559],
             [0.5204441, 0.4795559],
             ...,
             [0.5204441, 0.4795559],
             [0.5204441, 0.4795559],
             [0.5204441, 0.4795559]]
     */
  } catch(err) {
    console.log(err);
  }
}

loadModelAndPredict();
