import os
import cv2
import time
import random
import numpy as np
import tensorflow as tf
from keras_applications.mobilenet import preprocess_input
from keras.applications import Xception, ResNet50, MobileNet
from keras import backend, layers
from keras.models import Sequential
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

random.seed(5)
np.random.seed(5)
tf.random.set_seed(5)
image_size = (130,130)
input_shape = (130,130,3)
def read_image(index , ROOT):
    end = "web"+index[1]
    path = os.path.join(ROOT, index[0], end)
    image = cv2.imread(path,1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,image_size)
    image=np.array(image,dtype='float32')/255
    return image

def split_dataset(directory, split=0.9):
    folders = os.listdir(directory)
    num_train = int(len(folders) * split)

    random.shuffle(folders)

    train_list, val_list = {}, {}

    # Creating Train-list
    for folder in folders[:num_train]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        train_list[folder] = num_files

    # Creating Test-list
    for folder in folders[num_train:]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        val_list[folder] = num_files

    return train_list, val_list


def create_triplets(directory, folder_list, max_files=18):
    triplets = []
    folders = list(folder_list.keys())

    for folder in folders:
        path = os.path.join(directory, folder)

        files = list(os.listdir(path))[:max_files]
        num_files = len(files)

        for i in range(num_files - 1):
            for j in range(i + 1, num_files):
                anchor = (folder, f"{i + 1}.png")
                positive = (folder, f"{j + 1}.png")

                neg_folder = folder
                while neg_folder == folder:
                    neg_folder = random.choice(folders)
                neg_file = random.randint(1, folder_list[neg_folder] - 1)
                negative = (neg_folder, f"{neg_file}.png")

                triplets.append((anchor, positive, negative))

    random.shuffle(triplets)
    return triplets




def get_batch(triplet_list,ROOT, batch_size=64, preprocess=True):
    batch_steps = len(triplet_list) // batch_size

    for i in range(batch_steps + 1):
        anchor = []
        positive = []
        negative = []

        j = i * batch_size
        while j < (i + 1) * batch_size and j < len(triplet_list):
            ##print(triplet_list[j])
            a, p, n = triplet_list[j]
            anchor.append(read_image(a, ROOT))
            positive.append(read_image(p, ROOT))
            negative.append(read_image(n, ROOT))
            j += 1

        anchor=np.array(anchor)
        positive = np.array(positive)
        negative = np.array(negative)
        if preprocess:
            anchor = preprocess_input(anchor)
            positive = preprocess_input(positive)
            negative = preprocess_input(negative)

        yield ([anchor, positive, negative])

# def resnet_model(input_shape):
#     model = ResNet50(
#         weights=None,
#         include_top=False,
#         input_shape=input_shape)
#     x = model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dropout(0.7)(x)
#     predictions = Dense(40, activation='softmax')(x)  # Corrected activation function
#     res_model = Model(inputs=model.input, outputs=predictions)
#     return res_model

def get_encoder(input_shape):

    pretrained_model = MobileNet(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )

#     for i in range(len(pretrained_model.layers) - 27):
#         pretrained_model.layers[i].trainable = False

    encode_model = Sequential([
        pretrained_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")
    return encode_model


def test_on_triplets(val_triplet,siamese_model, ROOT, batch_size=256):
    pos_scores, neg_scores = [], []

    for data in get_batch(val_triplet, ROOT, batch_size=batch_size):
        prediction = siamese_model.predict(data)
        pos_scores += list(prediction[0])
        neg_scores += list(prediction[1])

    accuracy = np.sum(np.array(pos_scores) < np.array(neg_scores)) / len(pos_scores)
    ap_mean = np.mean(pos_scores)
    an_mean = np.mean(neg_scores)
    ap_stds = np.std(pos_scores)
    an_stds = np.std(neg_scores)

    print(f"Accuracy on test = {accuracy:.5f}")
    return accuracy, ap_mean, an_mean, ap_stds, an_stds

def read_test_data(directory):
    folders = os.listdir(directory)
    test_list = {}
    for folder in folders:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        test_list[folder] = num_files
    return test_list

def extract_encoder(model):
    encoder = get_encoder(input_shape)
    i=0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i+=1
    return encoder


def classify_images(face_list1, face_list2,model, threshold=1.3):
    # Getting the encodings for the passed faces
    tensor1 = model.predict(face_list1)
    tensor2 = model.predict(face_list2)

    distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)
    return prediction


def ModelMetrics(pos_list, neg_list):
    true = np.array([0] * len(pos_list) + [1] * len(neg_list))
    pred = np.append(pos_list, neg_list)

    # Compute and print the accuracy
    print(f"\nAccuracy of model: {accuracy_score(true, pred)}\n")

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(true, pred)

    categories = ['Similar', 'Different']
    names = ['True Similar', 'False Similar', 'False Different', 'True Different']
    percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.show()