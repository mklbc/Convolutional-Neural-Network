import tensorflow as tf
from model import create_model
from dataset.fashion_mnist import load_data

# Veriyi yükle
train_X, train_Y, test_X, test_Y = load_data()

# Modeli oluştur
model = create_model()

# Modeli derle
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Modeli eğit
history = model.fit(train_X, train_Y, epochs=20, batch_size=64, validation_split=0.2, verbose=1)

# Eğitilmiş modeli kaydet
model.save("models/fashion_mnist_cnn.h5")
