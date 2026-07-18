import os

print(os.path.getsize("model_data/yolo.h5"))

from h5py import File

f = File("model_data/yolo.h5","r")
print(list(f.keys()))

from tensorflow.keras.models import load_model

model = load_model("model_data/yolo.h5", compile=False)

print("Loaded successfully")