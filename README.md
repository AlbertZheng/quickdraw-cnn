# quickdraw-cnn

## Project info

**This is a "You draw, AI guess" MVP (Minimum Viable Product) that can recognize the drawing on web using a web friendly CNN model.**

This project utilizes [Google's "The Quick Draw" dataset](https://github.com/googlecreativelab/quickdraw-dataset). The Quick Draw dataset is a collection of **50 million drawings** across **345 categories**, and if training the network model using the full dataset, the needed computation resource of GPU will be very huge for me. So as for demonstrating how to develop a AI enabled Web App by using TensorFlow full stack, I **just sampled 10 categories and 20000 pictures per category (i.e. total 200000 pictures)** from this dataset to train the CNN model, and achieved almost 94% accuracy after training 16 epoch, then published the model as a web friendly model for ```TensorFlow.js``` based Web App.


## Technical details

### Technical stack
- Server side: **```TensorFlow + TensorLayer + Python```**
- Client side: **```TensorFlow.js + ES6```**
- Model UT: **```TensorFlow.js + Node.js```**


### Codes style

Codes style of this MVP:
- More TF & TL (1.x) style: use more recent and decent TF APIs.
- More Pythonic: fully leverage the power of python.
- Readability (over efficiency): Since it's for instruction purposes, we prefer readability over others.
- Understandability (over everything): Understanding DL key concepts is the main goal of this code.


## Live demo

To play the live demo [https://ai.kyletiger.com/quickdraw-cnn](https://ai.kyletiger.com/quickdraw-cnn), you need to use a modern web browser (e.g. **```Chrome```, ```Safari```, or ```Wechat``` on either web or mobile devices**) that supports ES6 runtime. **IE (or IE kernel based browsers) isn't supported!**
