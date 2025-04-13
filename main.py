import os

print("Options:")
print("1. Train Model")
print("2. Test Model")
print("3. Make Predictions with the Model")
seçim = input("Enter your choice: ")

if seçim == "1":
    os.system("python train.py")
elif seçim == "2":
    os.system("python evaluate.py")
elif seçim == "3":
    os.system("python predict.py")
else:
    print("Invalid selection!")
