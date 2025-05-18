import numpy as np
from PIL import Image
import os


# Load and preprocess images
def load_images(folder, label, max_images=100):
    data, labels = [], []
    count = 0
    for file in os.listdir(folder):
        if count >= max_images:
            break
        try:
            path = os.path.join(folder, file)
            img = Image.open(path).convert("RGB").resize((64, 64))
            img_array = np.array(img) / 255.0
            data.append(img_array.reshape(-1))  # Flatten the image
            labels.append(label)
            count += 1
        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue
    return data, labels


# Load data
with_mask_data, with_mask_labels = load_images("project/Train/WithMask", 0)
without_mask_data, without_mask_labels = load_images("project/Train/WithoutMask", 1)

val_with_mask_data, val_with_mask_labels = load_images("project/Validation/WithMask", 0)
val_without_mask_data, val_without_mask_labels = load_images("project/Validation/WithoutMask", 1)

# Combine data and labels
X = np.array(with_mask_data + without_mask_data)
y = np.array(with_mask_labels + without_mask_labels)

y_val = np.array(val_with_mask_labels + val_without_mask_labels)

# Normalize with mean subtraction
X_mean = np.mean(X, axis=0)
X -= X_mean

# One-hot encoding for 2 classes
y_one_hot = np.zeros((len(y), 2))
y_one_hot[np.arange(len(y)), y] = 1


# Neural Network definition
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.lr = learning_rate
        self.params = self.initialize_parameters(input_size, hidden_size, output_size)

    def initialize_parameters(self, input_size, hidden_size, output_size):
        W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        b2 = np.zeros((1, output_size))
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward_pass(self, X):
        W1, b1, W2, b2 = self.params["W1"], self.params["b1"], self.params["W2"], self.params["b2"]
        Z1 = np.dot(X, W1) + b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = self.sigmoid(Z2)
        return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    def backprop(self, X, y_true, A1, A2):
        m = y_true.shape[0]
        dZ2 = A2 - y_true  # derivative of cross-entropy with softmax
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = np.dot(dZ2, self.params['W2'].T) * A1 * (1 - A1)  # sigmoid derivative
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_params(self, grads):
        for key in grads:
            param_key = key.replace('d', '')
            self.params[param_key] -= self.lr * grads[key]

    def train_short(self, X_train, y_train, batch_size=32, steps=20):
        for step in range(steps):
            indices = np.random.permutation(len(X_train))
            X_train, y_train = X_train[indices], y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                outputs = self.forward_pass(X_batch)
                grads = self.backprop(X_batch, y_batch, outputs["A1"], outputs["A2"])
                self.update_params(grads)

    def predict(self, X):
        outputs = self.forward_pass(X)
        return np.argmax(outputs["A2"], axis=1)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)


# Training and classification
if __name__ == "__main__":
    input_size = 64 * 64 * 3
    hidden_size = 128
    output_size = 2
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.01)

    nn.train_short(X, y_one_hot, steps=20)

    X_val = np.array(val_with_mask_data + val_without_mask_data)
    X_val -= X_mean  # Normalize with training mean

    val_preds = nn.predict(X_val)
    val_accuracy = nn.accuracy(y_val, val_preds)
    print(f"Validation Accuracy: {val_accuracy:.4f}")


    def classify_image(image_path, model):
        try:
            img = Image.open(image_path).convert("RGB").resize((64, 64))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, -1)
            img_array -= X_mean
            pred = model.predict(img_array)
            return "without_mask" if pred[0] == 1 else "with_mask"
        except Exception as e:
            return f"Error classifying image: {e}"


    # Example usage
    test_image = "project/Validation/WithMask/Augmented_28_4264624.png"
    print(classify_image(test_image, nn))
    print(classify_image("project/Validation/WithMask/141.png", nn))


