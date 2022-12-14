import gdown
import os

data_p34_url = 'https://drive.google.com/u/0/uc?id=1T1PkyWh_5agXP6OGcGJ4DeUx0XO0wx7Q&export=download'
data_p64_url = 'https://drive.google.com/u/0/uc?id=1sLtIqi7hNdy7uqEGa1893zB78TQO8wS7&export=download'
model_url = 'https://drive.google.com/u/0/uc?id=10VeSD46ck1r3ncvXoFCFbaDQ6u6bdd30&export=download'
model_Tnet_url = 'https://drive.google.com/u/0/uc?id=12eM8zSGOBget2QF7-xwBjPUdcmsATb29&export=download'

# dataset
print("Downloading datasets...")
gdown.download(data_p34_url, 'data_p34.npz')
os.rename("./data_p34.npz", "./emotion_cls/dataset/data_p34.npz")
gdown.download(data_p64_url, 'data_p64.npz')
os.rename("./data_p64.npz", "./emotion_cls/dataset/data_p64.npz")

# model
print("Downloading models...")
gdown.download(model_url, 'model.bin')
os.rename("./model.bin", "./checkpoints/model.bin")
gdown.download(model_Tnet_url, 'model_Tnet.bin')
os.rename("./model_Tnet.bin", "./checkpoints/model_Tnet.bin")