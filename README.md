# Neural Network for MNIST Digit Classification

This project implements a neural network to classify handwritten digits from the MNIST dataset. The neural network is trained using backpropagation and gradient descent.

## Project Structure

- `main.py`: The main script to load data, train the neural network, and evaluate its performance.
- `mnist-original.mat`: The dataset file containing the MNIST data.
- `Model.py`: Contains the implementation of the neural network and the cost function.
- `Prediction.py`: Contains the function to make predictions using the trained neural network.
- `RandInitialise.py`: Contains the function to randomly initialize the weights of the neural network.
- `Theta1.txt` and `Theta2.txt`: Files to save the trained weights of the neural network.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/vijayn7/Digit_Classifier.git
    cd DIGIT_CLASSIFIER
    ```

2. Install the required dependencies:
    ```sh
    pip install numpy scipy matplotlib
    ```

## Usage

1. Load the MNIST dataset and preprocess the data:
    ```python
    from scipy.io import loadmat
    data = loadmat('mnist-original.mat')
    X = data['data'].transpose() / 255
    y = data['label'].flatten()
    ```

2. Split the data into training and testing sets:
    ```python
    X_train = X[:60000, :]
    y_train = y[:60000]
    X_test = X[60000:, :]
    y_test = y[60000:]
    ```

3. Initialize the neural network parameters:
    ```python
    from RandInitialise import initialise
    input_layer_size = 784
    hidden_layer_size = 100
    num_labels = 10
    initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
    initial_Theta2 = initialise(num_labels, hidden_layer_size)
    initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
    ```

4. Train the neural network:
    ```python
    from scipy.optimize import minimize
    from Model import neural_network
    lambda_reg = 0.1
    maxiter = 100
    myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)
    results = minimize(neural_network, x0=initial_nn_params, args=myargs, options={'disp': True, 'maxiter': maxiter}, method="L-BFGS-B", jac=True)
    nn_params = results["x"]
    ```

5. Evaluate the model:
    ```python
    from Prediction import predict
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))
    pred = predict(Theta1, Theta2, X_test)
    print('Test Set Accuracy: {:f}'.format((np.mean(pred == y_test) * 100)))
    ```

6. Visualize the learned weights and activations:
    ```python
    import matplotlib.pyplot as plt
    def visualize_activations(X, Theta1, Theta2):
        m = X.shape[0]
        a1 = np.hstack([np.ones((m, 1)), X])
        z2 = a1.dot(Theta1.T)
        a2 = np.hstack([np.ones((m, 1)), 1 / (1 + np.exp(-z2))])
        z3 = a2.dot(Theta2.T)
        a3 = 1 / (1 + np.exp(-z3))

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(a1[0, 1:].reshape(28, 28), cmap='gray')
        ax[0].set_title('Input Layer')
        ax[1].imshow(a2[0, 1:].reshape(10, 10), cmap='gray')
        ax[1].set_title('Hidden Layer')
        ax[2].imshow(a3[0, :].reshape(1, 10), cmap='gray')
        ax[2].set_title('Output Layer')
        plt.show()

    visualize_activations(X_test[:1], Theta1, Theta2)
    ```

## License

This project is licensed under the MIT License.