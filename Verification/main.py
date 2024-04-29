import os
import cv2
import time
import random
import numpy as np
import Preprocessing as pp
import tensorflow as tf
import Distance_Layer as DL
import Siamese_Model as SM
from keras.optimizers import Adam
import warnings

random.seed(5)
np.random.seed(5)
tf.random.set_seed(5)
save_all = False
epochs = 30
batch_size = 64

max_acc = 0
train_loss = []
test_metrics = []
ROOT = "D:\PycharmProjects\pythonProject1\CV project\Data\Product Recoginition\Training Data"
warnings.filterwarnings('ignore')
train_list, val_list = pp.split_dataset(ROOT, split=0.9)
print("Length of training list:", len(train_list))
print("Length of validation list :", len(val_list))

train_triplet = pp.create_triplets(ROOT, train_list)
val_triplet = pp.create_triplets(ROOT, val_list)

print("Number of training triplets:", len(train_triplet))
print("Number of testing triplets :", len(val_triplet))

print("\nExamples of triplets:")
for i in range(5):
    print(train_triplet[i])

siamese_network = DL.get_siamese_network()
siamese_network.summary()

siamese_model = SM.SiameseModel(siamese_network)
optimizer = Adam(learning_rate=1e-5, epsilon=1e-7)
siamese_model.compile(optimizer=optimizer)

os.environ['KMP_WARNINGS'] = 'off'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

for epoch in range(1, epochs + 1):
    t = time.time()

    # Training the model on train data
    epoch_loss = []
    for data in pp.get_batch(train_triplet, ROOT, batch_size=batch_size):
        loss = siamese_model.train_on_batch(data)
        epoch_loss.append(loss)
    epoch_loss = sum(epoch_loss) / len(epoch_loss)
    train_loss.append(epoch_loss)

    print(f"\nEPOCH: {epoch} \t (Epoch done in {int(time.time() - t)} sec)")
    print(f"Loss on train    = {epoch_loss:.5f}")

    # Testing the model on test data
    metric = pp.test_on_triplets(val_triplet=val_triplet,siamese_model=siamese_model, ROOT=ROOT, batch_size=batch_size)
    test_metrics.append(metric)
    accuracy = metric[0]

    # Saving the model weights
    if save_all or accuracy >= max_acc:
        siamese_model.save_weights("siamese_model")
        max_acc = accuracy

# Saving the model after all epochs run
siamese_model.save_weights("siamese_model-final")