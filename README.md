# Assignment 3: Image Captioning with Vanilla RNNs and LSTMs, Neural Net Visualization, Style Transfer, Generative Adversarial Networks

## Goals

In this assignment, you will implement recurrent neural networks and apply them to image captioning on the Microsoft COCO data. You will also explore methods for visualizing the features of a pretrained model on ImageNet, and use this model to implement Style Transfer. Finally, you will train a Generative Adversarial Network to generate images that look like a training dataset!

The goals of this assignment are as follows:

* Understand the architecture of recurrent neural networks (RNNs) and how they operate on sequences by sharing weights over time.
* Understand and implement both Vanilla RNNs and Long-Short Term Memory (LSTM) networks.
* Understand how to combine convolutional neural nets and recurrent nets to implement an image captioning system.
* Explore various applications of image gradients, including saliency maps, fooling images, class visualizations.
* Understand and implement techniques for image style transfer.
* Understand how to train and implement a Generative Adversarial Network (GAN) to produce images that resemble samples from a dataset.

## Q1: Image Captioning with Vanilla RNNs

The notebook [`RNN_Captioning.ipynb`](https://github.com/eemrys/cs231n/tree/assignment_3/assignment3/RNN_Captioning.ipynb) will walk you through the implementation of an image captioning system on MS-COCO using vanilla recurrent networks.

## Q2: Image Captioning with LSTMs

The notebook [`LSTM_Captioning.ipynb`](https://github.com/eemrys/cs231n/tree/assignment_3/assignment3/LSTM_Captioning.ipynb) will walk you through the implementation of Long-Short Term Memory (LSTM) RNNs, and apply them to image captioning on MS-COCO.

## Q3: Network Visualization: Saliency maps, Class Visualization, and Fooling Images

The notebooks [`NetworkVisualization-TensorFlow.ipynb`](https://github.com/eemrys/cs231n/tree/assignment_3/assignment3/NetworkVisualization-TensorFlow.ipynb), and `NetworkVisualization-PyTorch.ipynb` will introduce the pretrained SqueezeNet model, compute gradients with respect to images, and use them to produce saliency maps and fooling images. Please complete only one of the notebooks (<ins>TensorFlow</ins> or PyTorch). No extra credit will be awardeded if you complete both notebooks.

## Q4: Style Transfer

In thenotebooks [`StyleTransfer-TensorFlow.ipynb`](https://github.com/eemrys/cs231n/tree/assignment_3/assignment3/StyleTransfer-TensorFlow.ipynb) or `StyleTransfer-PyTorch.ipynb` you will learn how to create images with the content of one image but the style of another. Please complete only one of the notebooks (<ins>TensorFlow</ins> or PyTorch). No extra credit will be awardeded if you complete both notebooks.

## Q5: Generative Adversarial Networks

In the notebooks [`GANS-TensorFlow.ipynb`](https://github.com/eemrys/cs231n/tree/assignment_3/assignment3/Generative_Adversarial_Networks_TF.ipynb) or `GANS-PyTorch.ipynb` you will learn how to generate images that match a training dataset, and use these models to improve classifier performance when training on a large amount of unlabeled data and a small amount of labeled data. Please complete only one of the notebooks (<ins>TensorFlow</ins> or PyTorch). No extra credit will be awarded if you complete both notebooks.
