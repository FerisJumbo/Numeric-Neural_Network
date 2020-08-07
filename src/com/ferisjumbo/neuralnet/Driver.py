import numpy as np
import X_traindata.Training_Data as traindata
import Optimizers as op
import NetworkUtils as nU
from tqdm import tqdm
import math

np.random.seed(0)

X_train, y_train = traindata.get_training_data()

dense1 = nU.Layer_Dense(1024, 512, weight_regularizer_l2=1e-3, bias_regularizer_l2=1e-3)
activation1 = nU.ReLU_Activation()
dense2 = nU.Layer_Dense(512, 384, weight_regularizer_l2=1e-3, bias_regularizer_l2=1e-3)
activation2 = nU.ReLU_Activation()
dense3 = nU.Layer_Dense(384, 10)
activation3 = nU.Softmax_Activation()

loss_function = nU.Loss_CategoricalCrossentropy()
optimizer = op.Optimizer_Adam(learning_rate=5e-3, decay=1e-8)

epochs = 1000
loop = tqdm(total=epochs, position=0, leave=False)

for epoch in range(epochs + 1):
    # Forward Propagation
    dense1.forward_propagation(X_train)
    activation1.forward_propagation(dense1.output)

    dense2.forward_propagation(activation1.output)
    activation2.forward_propagation(dense2.output)

    dense3.forward_propagation(activation2.output)
    activation3.forward_propagation(dense3.output)

    # Calculate Loss and Accuracy
    data_loss = loss_function.forward_propagation(activation3.output, y_train)
    regularization_loss = (
        loss_function.regularization_loss(dense1)
        + loss_function.regularization_loss(dense2)
        + loss_function.regularization_loss(dense3)
    )
    loss = data_loss + regularization_loss

    predictions = np.argmax(activation3.output, axis=1)
    accuracy = np.mean(predictions == y_train)

    if not epoch % (epochs / 10):
        print(
            "\n",
            "epoch:",
            epoch,
            "acc:",
            accuracy,
            "loss:",
            loss,
            "(data_loss: ",
            data_loss,
            " regul loss: ",
            regularization_loss,
            ")" "\n\tlr: ",
            optimizer.current_learning_rate,
        )

    # Backward Propagation
    loss_function.backward_propagation(activation3.output, y_train)
    activation3.backward_propagation(loss_function.dvalues)
    dense3.backward_propagation(activation3.dvalues)
    activation2.backward_propagation(dense3.dvalues)
    dense2.backward_propagation(activation2.dvalues)
    activation1.backward_propagation(dense2.dvalues)
    dense1.backward_propagation(activation1.dvalues)

    # Optimize Weights
    optimizer.pre_optimize()

    optimizer.optimize(dense1)
    optimizer.optimize(dense2)
    optimizer.optimize(dense3)

    optimizer.post_optimize()

    # Visual Progress bar
    loop.set_description("Learning:".format(epochs))
    loop.update(1)

loop.close()

X_test, y_test = traindata.get_testing_data("three", 1)

dense1.forward_propagation(X_test)
activation1.forward_propagation(dense1.output)

dense2.forward_propagation(activation1.output)
activation2.forward_propagation(dense2.output)

dense3.forward_propagation(activation2.output)
activation3.forward_propagation(dense3.output)

data_loss = loss_function.forward_propagation(activation3.output, y_test)
regularization_loss = (
    loss_function.regularization_loss(dense1)
    + loss_function.regularization_loss(dense2)
    + loss_function.regularization_loss(dense3)
)
loss = data_loss + regularization_loss

predictions = np.argmax(activation3.output, axis=1)
accuracy = np.mean(predictions == y_test)

print(f"\nvalidation check:\nacc:{accuracy:3f} loss:{loss:3f}")
print(activation3.output)
