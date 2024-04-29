
import keras.models
import numpy as np
import Preprocessing as pp
batch_size = 32
test_root = 'D:\PycharmProjects\pythonProject1\CV project\Data\Product Recoginition\Validation Data'
validation_list = pp.read_test_data(test_root)
validation_triplet = pp.create_triplets(test_root, validation_list)

print("Number of validation triplets :", len(validation_triplet))
model = keras.models.load_model("D:\PycharmProjects\pythonProject1\CV project\cvModel1.h5")
pos_list = np.array([])
neg_list = np.array([])

for data in pp.get_batch(validation_triplet,test_root, batch_size):
    a, p, n = data
    pos_list = np.append(pos_list, pp.classify_images(a, p, model))
    neg_list = np.append(neg_list, pp.classify_images(a, n, model))
    break

pp.ModelMetrics(pos_list, neg_list)
