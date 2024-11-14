from scipy.io import loadmat
import numpy as np
from Model import neural_network
from RandInitialise import initialise
from Prediction import predict
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Sigmoid function definition
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loading mat file
data = loadmat('mnist-original.mat')

# Extracting features from mat file
X = data['data']
X = X.transpose()

# Normalizing the data
X = X / 255

# Extracting labels from mat file
y = data['label']
y = y.flatten()

# Splitting data into training set with 60,000 examples
X_train = X[:60000, :]
y_train = y[:60000]

# Splitting data into testing set with 10,000 examples
X_test = X[60000:, :]
y_test = y[60000:]

m = X.shape[0]
input_layer_size = 784  # Images are of (28 X 28) px so there will be 784 features
hidden_layer_size = 100
num_labels = 10  # There are 10 classes [0, 9]

# Randomly initialising Thetas
initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
initial_Theta2 = initialise(num_labels, hidden_layer_size)

# Unrolling parameters into a single column vector
initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
maxiter = 100
lambda_reg = 0.1  # To avoid overfitting
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)

# Placeholder for accuracy over iterations
train_accuracy = []
test_accuracy = []

# Callback function to store accuracy at each iteration
def callbackF(params):
    global train_accuracy, test_accuracy
    Theta1 = np.reshape(params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))
    train_pred = predict(Theta1, Theta2, X_train)
    test_pred = predict(Theta1, Theta2, X_test)
    train_accuracy.append(np.mean(train_pred == y_train) * 100)
    test_accuracy.append(np.mean(test_pred == y_test) * 100)

# Calling minimize function to minimize cost function and to train weights
results = minimize(neural_network, x0=initial_nn_params, args=myargs, 
          options={'disp': True, 'maxiter': maxiter}, method="L-BFGS-B", jac=True, callback=callbackF)

nn_params = results["x"]  # Trained Theta is extracted

# Weights are split back to Theta1, Theta2
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (
                              hidden_layer_size, input_layer_size + 1))  # shape = (100, 785)
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
                      (num_labels, hidden_layer_size + 1))  # shape = (10, 101)

# Checking test set accuracy of our model
pred = predict(Theta1, Theta2, X_test)
print('Test Set Accuracy: {:f}'.format((np.mean(pred == y_test) * 100)))

# Checking train set accuracy of our model
pred = predict(Theta1, Theta2, X_train)
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y_train) * 100)))

# Evaluating precision of our model
true_positive = 0
for i in range(len(pred)):
    if pred[i] == y_train[i]:
        true_positive += 1
false_positive = len(y_train) - true_positive
print('Precision =', true_positive/(true_positive + false_positive))

# Saving Thetas in .txt file
np.savetxt('Theta1.txt', Theta1, delimiter=' ')
np.savetxt('Theta2.txt', Theta2, delimiter=' ')

# Plotting training and test accuracy over iterations
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy over Iterations')
plt.legend()
plt.show()

# Visualizing the learned weights (Theta1)
fig, axarr = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axarr[i, j].imshow(Theta1[i * 10 + j, 1:].reshape(28, 28), cmap='gray')
        axarr[i, j].axis('off')
plt.suptitle('Visualization of Learned Weights (Theta1)')
plt.show()

# Visualizing activations
def visualize_activations(X, Theta1, Theta2):
    # Forward propagation
    m = X.shape[0]
    a1 = np.hstack([np.ones((m, 1)), X])
    z2 = a1.dot(Theta1.T)
    a2 = np.hstack([np.ones((m, 1)), sigmoid(z2)])
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(a1[0, 1:].reshape(28, 28), cmap='gray')
    ax[0].set_title('Input Layer')
    ax[1].imshow(a2[0, 1:].reshape(10, 10), cmap='gray')
    ax[1].set_title('Hidden Layer')
    ax[2].imshow(a3[0, :].reshape(1, 10), cmap='gray')
    ax[2].set_title('Output Layer')
    plt.show()

# Assuming X_test is defined and contains test data
visualize_activations(X_test[:1], Theta1, Theta2)