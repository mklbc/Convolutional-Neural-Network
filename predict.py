import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from dataset.fashion_mnist import load_data

# Modeli yükle
model = load_model("models/fashion_mnist_cnn.h5")

# Test verisini al
_, _, test_X, test_Y = load_data()

# Tahmin yap
predictions = model.predict(test_X)
predicted_classes = np.argmax(predictions, axis=1)


for i in range(5):
    plt.imshow(test_X[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_classes[i]}")
    plt.show()
