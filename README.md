# Object-Classification
Using Deep Neural Networks for Object Classification <br>
![Cat and Dog Image](https://github.com/davidakhihiero/Object-Classification/blob/main/images/Cat%20and%20Dog.jpg?raw=true)

In this project, I built and trained 3 neural network models for object classification (cats and dogs images); a simple multilayer perceptron
model, a convolutional neural network and a simple residual neural network.

Dataset from Kaggle: [go](http://stackoverflow.com){:target="_blank"}. <a href="https://www.kaggle.com/competitions/dogs-vs-cats/data" target="_blank">Cats and Dogs Dataset</a>

The MLP had 4 hidden layers and fit the training set to about 90% accuracy after 30 epochs but performed very poorly on the dev set with
an accuracy of about 61.5%. 

The CNN which had three convolution-maxpool blocks performed much better on the dev set with an accuracy of 85.5% after fitting the 
training data to almost 97% accuracy after only 10 epochs.

The ResNet had three residual-pooling blocks, immediately after a convolution-maxpool block, with each residual block containing two convolutional 
layers. 
