import matplotlib.pyplot as plt

def plot_history(history):
    """Eğitim sürecindeki doğruluk ve kayıpları görselleştirir."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))

    plt.figure(figsize=(12, 4))

    # Doğruluk grafiği
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Eğitim Doğruluğu')
    plt.plot(epochs, val_acc, 'r', label='Doğrulama Doğruluğu')
    plt.legend()
    plt.title('Eğitim ve Doğrulama Doğruluğu')

    # Kayıp grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Eğitim Kaybı')
    plt.plot(epochs, val_loss, 'r', label='Doğrulama Kaybı')
    plt.legend()
    plt.title('Eğitim ve Doğrulama Kaybı')

    plt.show()
