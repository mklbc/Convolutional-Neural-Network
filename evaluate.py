from keras.models import load_model
from dataset.fashion_mnist import load_data

# Modeli yükle
model = load_model("models/fashion_mnist_cnn.h5")

# Test verisini yükle
_, _, test_X, test_Y = load_data()

# Modeli değerlendir
test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=1)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")
