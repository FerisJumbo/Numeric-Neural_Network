# Numeric Neural Network

Still a work in progress, but will be able to tell the user which single digit number they wrote in a 32x32 pixel canvas.

## The Neural Network

##### Just a couple of specs (and yes I'm listing these because it makes me sound smarter than I am).

The network uses the Adam Optimizer (Adaptive Momentum) as it is one, most commonly used, and two, lowers total loss better than the other ones I know.

Their are 1,024 (32x32) input nodes that pass through 2 hidden layers with sizes of 768 and 512 neurons respectively and finally 10 output nuerons (0, 1, ..., 9).
(Since the Neural Net is 2 hidden layers deep, technically it is classified as 'Deep Learning' AI, so thats cool I guess...)

The first and second hidden layers each use the Rectified Linear Unit (ReLU) Activation Function as it is simpler and all you really need for hidden layers.
The output layer uses the Softmax Activation Function as it gives how confident the Neuron is in its answer.

The Network uses Categorical Cross-Entropy Loss, which is just a fancy way of saying "Your Network sucks." Essentially the closer to zero the loss is, the better
your Network is at predicting values.
Loss is calculated by data_loss + regularization_loss (I'd explain this more if I actually understood the calculus behind it).

## How it works

I'll give a small explanation on what goes on in a Neural Network during training as some probably dont know.

1. Forward Propagate
1. Calculate Loss
1. Backward Propagate
1. Repeat 200,000 Times

I know its a little wordy, but I think you get the just of it.

## Training

The latest results from training the neural network are...

```
Training-
    epochs: 2,500
    accuracy: 1.0
    total loss: 1.08587
        data loss:0.00129
        regularization loss: 1.08457
    end learning rate: 0.00024

Valadation-
    number used: 0
    accuracy: 1.0
    total loss: 1.36353
    chosen number: 0
    confidence: 75.55%
```
