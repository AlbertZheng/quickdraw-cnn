# quickdraw-cnn

## Project info

**This is a "You draw, AI guesses" MVP (Minimum Viable Product) that can recognize the sketch drawing on web canvas using a TensorFlow.js friendly CNN model.**

This project utilizes [Google's "The Quick Draw" dataset](https://github.com/googlecreativelab/quickdraw-dataset). The Quick Draw dataset is a collection of **50 million drawings** across **345 categories**, and if training the network model using the full dataset, the needed computation resource of GPU will be very huge for me. So as for demonstrating how to develop a AI enabled Web App by using TensorFlow full stack, I **just sampled 10 categories and tens of thousands pictures per category** from this dataset to train the CNN model, and **achieved 94.87% accuracy** after training 25 epochs on Google colab, then published the model as a web friendly model for ```TensorFlow.js``` based Web App.


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


## Demo live

To play the demo live [https://ai.kyletiger.com/quickdraw-cnn](https://ai.kyletiger.com/quickdraw-cnn), you need to use a modern web browser (e.g. **```Chrome```, ```Safari```, or ```Wechat``` on either web or mobile devices**) that supports ES6 runtime. **IE (or IE kernel based browsers) isn't supported!**


## License

Copyright (C) 2000-2018 Lisong Zheng, 郑立松

The binaries and source code of this project can be used according to the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0.html).
