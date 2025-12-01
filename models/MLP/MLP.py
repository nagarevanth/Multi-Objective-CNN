import numpy as np
import wandb

class ActivationFunctions:
    @staticmethod
    def activate(act_fn, z):
        if act_fn == 'linear':
            return z
        elif act_fn == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif act_fn == 'relu':
            return np.maximum(0, z)
        elif act_fn == 'tanh':
            return np.tanh(z)
        elif act_fn == 'softmax':
            exps = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
            return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def derivative(act_fn, z):
        if act_fn == 'linear':
            return 1
        elif act_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        elif act_fn == 'relu':
            return (z > 0).astype(float)
        elif act_fn == 'tanh':
            return 1 - np.tanh(z) ** 2
        # No derivative implementation needed for softmax, handled in loss functions for classification tasks


class MLP:
    def __init__(self, input_size, output_size, lr=0.01, num_epoch=1000, act_fn='relu', 
                 opt_fn='SGD', n_neuron=[64, 32], batch_size=32, task_type='classification_bi', tol=1e-6,isWandb = 0):
        self.lr = lr
        self.num_epoch = num_epoch
        self.act_fn = act_fn
        self.opt_fn = opt_fn
        self.n_hidden = len(n_neuron)
        self.n_neuron = n_neuron
        self.batch_size = batch_size
        self.task_type = task_type
        self.layer_sizes = [input_size] + self.n_neuron + [output_size]
        self.weights, self.biases = self.init_params()
        self.tol = tol
        self.isWandb = isWandb

    def init_params(self):
        weights = []
        biases = []
        for i in range(len(self.layer_sizes) - 1):
            weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01)
            biases.append(np.zeros((1, self.layer_sizes[i + 1])))
        return weights, biases

    def forward(self, X):
        self.a_list = [X]  # Activations list
        self.z_list = []   # Linear transformations list
        a = X

        # Forward pass through each layer
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = ActivationFunctions.activate(self.act_fn, z)
            
            self.z_list.append(z)
            self.a_list.append(a)

        # Output layer
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        if self.task_type == 'classification_bi' or self.task_type == 'classification_ml':
            a = ActivationFunctions.activate('sigmoid', z)  # Use sigmoid for binary classification
        elif self.task_type == 'classification':
            a = ActivationFunctions.activate('softmax', z)  # Use softmax for multi-class classification
        else:
            a = ActivationFunctions.activate('linear', z)  # No activation for regression

        self.z_list.append(z)
        self.a_list.append(a)
        return self.a_list[-1]

    def bce_loss(self, y_pred, y_true):
        epsilon = 1e-9  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)
    
    def cross_entropy_loss(self, y_pred, y_val):
        # Ensure predictions are not exactly 0 or 1 to avoid log(0)
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        
        # Compute cross-entropy loss for one-hot encoded y_val
        loss = -np.sum(y_val * np.log(y_pred)) / y_val.shape[0]
        
        return loss

    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, X, y):
        m = X.shape[0]
        gradients_w = []
        gradients_b = []

        if self.task_type == 'classification_bi' or self.task_type == 'classification_ml':
            delta = (self.a_list[-1] - y) / m  # Derivative of BCE Loss
        else:
            delta = (self.a_list[-1] - y) / (2*m)  # Derivative of MSE Loss for regression

        for i in range(self.n_hidden, -1, -1):
            dJ_dw = np.dot(self.a_list[i].T, delta) if i != 0 else np.dot(X.T, delta)
            dJ_db = np.sum(delta, axis=0, keepdims=True)

            # Update delta for the previous layer
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * ActivationFunctions.derivative(self.act_fn, self.z_list[i - 1])
            gradients_w.append(dJ_dw)
            gradients_b.append(dJ_db)

        gradients_w.reverse()
        gradients_b.reverse()
        return gradients_w, gradients_b

    def update_params(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * gradients_w[i]
            self.biases[i] -= self.lr * gradients_b[i]

    def optimizer(self, X, y):
        if self.opt_fn == 'SGD':
            self.sgd_optimizer(X, y)
        elif self.opt_fn == 'MBGD':
            self.mini_batch_optimizer(X, y)
        elif self.opt_fn == 'BGD':
            self.batch_optimizer(X, y)

    def sgd_optimizer(self, X, y):
        for i in range(X.shape[0]):
            X_i = X[i:i+1]
            y_i = y[i:i+1]
            self.forward(X_i)
            gradients_w, gradients_b = self.backward(X_i, y_i)
            self.update_params(gradients_w, gradients_b)

    def mini_batch_optimizer(self, X, y):
        for i in range(0, X.shape[0], self.batch_size):
            X_batch = X[i:i + self.batch_size]
            y_batch = y[i:i + self.batch_size]
            self.forward(X_batch)
            gradients_w, gradients_b = self.backward(X_batch, y_batch)
            self.update_params(gradients_w, gradients_b)

    def batch_optimizer(self, X, y):
        self.forward(X)
        gradients_w, gradients_b = self.backward(X, y)
        self.update_params(gradients_w, gradients_b)

    def check_early_stop(self, X_val, y_val, prev_loss):
        y_pred = self.predict(X_val)
        if self.isWandb == 1:
            if self.task_type == 'classification':
                wandb.log({'val_acc':np.mean(y_pred == y_val)})
            elif self.task_type == 'regression':
                wandb.log({'val_mse': np.mean((y_pred - y_val) **2)})
            
        # print(y_val.shape)
        # print(y_pred.shape)
        if self.task_type == 'classification_bi' or self.task_type == 'classification_ml':
            loss = self.bce_loss(y_pred, y_val)
        elif self.task_type == 'classification':
            loss = self.cross_entropy_loss(y_pred, y_val)
        else:
            loss = self.mse_loss(y_pred, y_val)
        
        if loss < prev_loss:
            return 1, loss
        else:
            return 0, loss

    def fit(self, X, y, X_val, y_val):
        prev_loss = -np.inf
        prev_loss_tr = np.inf
        losses = []
        for epoch in range(self.num_epoch):
            y_pred = self.forward(X)
            # Calculate loss for monitoring
            if self.task_type == 'classification_bi' or self.task_type == 'classification_ml':
                loss = self.bce_loss(y_pred, y)
            elif self.task_type == 'classification':
                # print(y_pred.shape)
                # print(y.shape)
                loss = self.cross_entropy_loss(y_pred, y)
            else:
                loss = self.mse_loss(y_pred, y)
            self.optimizer(X, y)
            prev_loss, early_stop = self.check_early_stop(X_val, y_val, prev_loss)
            if self.isWandb == 1:
                wandb.log({"epoch":epoch+1,"loss":loss})

            if early_stop == 1:
                break

            if np.abs(loss - prev_loss_tr) < self.tol:
                break
            prev_loss_tr = loss
            if epoch % 100 == 0:
                # wandb.log({"epoch": epoch + 1, "loss": loss})
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
            losses.append(loss)
        return losses

    def predict(self, X):
        y_pred = self.forward(X)
        if self.task_type == 'classification_bi' or self.task_type == 'classification_ml':
            return (y_pred >= 0.5).astype(int)  # Binary classification prediction
        elif self.task_type == 'classification':
            # Convert softmax output to one-hot encoding
            one_hot_pred = np.zeros_like(y_pred)
            one_hot_pred[np.arange(y_pred.shape[0]), np.argmax(y_pred, axis=1)] = 1
            return one_hot_pred  # One-hot encoded predictions for multi-class classification
        else:
            return y_pred  # For regression


    def gradient_check(self, X, Y):
        epsilon = 1e-8  # Small value for numerical gradient calculation
        numerical_grads = []  # Changed to a list
        analytical_grads,_ = self.backward(X, Y)

        # Ensure analytical gradients are in numpy array format
        if isinstance(analytical_grads, list):
            analytical_grads = [np.array(grad) for grad in analytical_grads]

        # Checking gradients for each layer
        for layer in range(len(self.weights)):
            numerical_grad = np.zeros_like(self.weights[layer])  # Initialize as array for the current layer
            
            for i in range(self.weights[layer].shape[0]):
                for j in range(self.weights[layer].shape[1]):
                    original_value = self.weights[layer][i][j]

                    # Perturb the weights
                    self.weights[layer][i][j] = original_value + epsilon
                    loss_plus = self.forward(X)
                    loss_plus = self.calculate_loss(loss_plus, Y)

                    self.weights[layer][i][j] = original_value - epsilon
                    loss_minus = self.forward(X)
                    loss_minus = self.calculate_loss(loss_minus, Y)

                    numerical_grad[i][j] = (loss_plus - loss_minus) / (2 * epsilon)
                    self.weights[layer][i][j] = original_value  # Restore original value

            numerical_grads.append(numerical_grad)  # Append the numerical gradient for this layer
            differences = np.abs(numerical_grad - analytical_grads[layer])

            if np.mean(differences) < 0.5:
                print(f"Gradient check passed for layer {layer}")

    def calculate_loss(self, predictions, Y):
        if self.task_type == 'classification_bi' or self.task_type == 'classification_ml':
            return (predictions-Y)
        elif self.task_type == 'classification':
            return self.cross_entropy_loss(predictions, Y)
        else:
            return self.mse_loss(predictions, Y)

