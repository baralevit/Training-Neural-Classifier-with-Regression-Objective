# Training-Neural-Classifier-with-Regression-Objective
Modified an existing convolutional network to output vectors in <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^9"> instead of probability distributions. A point in $\mathbb{R}^9$ was then picked to represent each class, and the objective used to train the network was the MSE between the outputs and the target class representations
Through analysis of the results of the initial model, discovered a modification to the network that improved the results
