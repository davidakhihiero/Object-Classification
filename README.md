# Object-Classification
Using Deep Neural Networks for Object Classification <br>
![Cat and Dog Image](https://github.com/davidakhihiero/Object-Classification/blob/main/images/Cat%20and%20Dog.jpg?raw=true)

In this project, I built and trained 3 neural network models for object classification (cats and dogs images); a simple multilayer perceptron
model, a convolutional neural network and a simple residual neural network.

Dataset from Kaggle: <a href="https://www.kaggle.com/competitions/dogs-vs-cats/data" target="_blank">Cats and Dogs Dataset</a> <br>
The dataset of 25,000 labelled cats and dogs images was split into a 20,000-5,000 training-validation set. The training data was augmented by flipping
the images horizontally to increase the training images to 40,000.

The model input size was (64, 64, 1) so the training/dev images were read in as GRAYSCALE and resized to 64 x 64.

The MLP had 4 hidden layers, about 10.4 million trainable parameters and fit the training set to about 90% accuracy after 30 epochs but performed 
very poorly on the dev set with an accuracy of about 61.5%. 

The CNN which had three convolution-maxpool blocks and about 4.6 million trainable parameters performed much better on the dev set with an accuracy of
85.5% after fitting the training data to about 96% accuracy after only 10 epochs.

The ResNet had four residual-pooling blocks, immediately after a convolution-maxpool block, with each residual block containing two convolutional 
layers. The model had about 3.9 million trainable parameters. The model trained for 10 epochs to fit the training data to an accuracy of 96% and had an
accuracy of 90.4% on the dev set. <br>

![Dog Prediction](https://github.com/davidakhihiero/Object-Classification/blob/main/images/prediction_1.png?raw=true)
![Cat Prediction](https://github.com/davidakhihiero/Object-Classification/blob/main/images/prediction_2.png?raw=true)
