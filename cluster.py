import tensorflow as tf
import numpy as np
import csv
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#===som===

def load_data():
    dataset = []

    with open('O192-COMP7117-AS01-00-clustering.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(readCSV):
            if i != 0:
                calorie_density = (float(row[4]) + float(row[5])) / 100
                try:
                    fat_ratio = (float(row[8]) + float(row[9])) / (float(row[7]) + float(row[11]))
                except:
                    # if divisor is 0
                    fat_ratio = 1.0

                if fat_ratio == 0.0:
                    fat_ratio = 1.0

                sugar = float(row[14])
                protein = float(row[16])
                salt = float(row[17])

                dataset.append([calorie_density, fat_ratio, sugar, protein, salt])

    return dataset

def preprocessing():
    data = load_data()

    np_features = np.array(data)

    #===process features===
    # normalize here
    scaler = StandardScaler()
    normalize_res = scaler.fit_transform(np_features)

    # then do pca here
    pca = PCA(n_components=3)
    pca_res = pca.fit_transform(normalize_res)
    
    return pca_res

class SOM:
    def __init__(self, height, width, inp_dimension):
        # step 1 inisialisasi
        self.height = height
        self.width = width
        self.inp_dimension = inp_dimension

        # step 2 inisialisasi node
        self.node = [tf.to_float([i,j]) for i in range(height) for j in range(width)]
        self.input = tf.placeholder(tf.float32,[inp_dimension])
        self.weight = tf.Variable(tf.random_normal([height*width, inp_dimension]))

        # buat fungsi cari node terdekat, update weight
        self.best_matching_unit = self.get_bmu(self.input)
        self.updated_weight = self.get_update_weight(self.best_matching_unit, self.input)
    
    def get_bmu(self, input):
        # hitung distances terpendek pake euclidean
        expand_input = tf.expand_dims(input, 0)
        euclidean = tf.square(tf.subtract(expand_input, self.weight))
        distances = tf.reduce_sum(euclidean,1) # 0 akan menambahkan secara kesamping

        # cari index distance terpendek
        min_index = tf.argmin(distances, 0) # 1 akan mengkecek ke samping | 0 ke bawah
        bmu_location = tf.stack([tf.mod(min_index, self.width), tf.div(min_index, self.width)])
        return tf.to_float(bmu_location)

    def get_update_weight(self, bmu, input):
        # inisialisasi learning rate & sigma
        learning_rate = 0.5
        sigma = tf.to_float(tf.maximum(self.height, self.width)/ 2)

        # hitung perbedaan jarak bmu ke node lain pake euclidean(bmu = best matching unit)
        expand_bmu = tf.expand_dims(bmu, 0)
        euclidean = tf.square(tf.subtract(expand_bmu, self.node))
        distances = tf.reduce_sum(euclidean, 1)
        ns = tf.exp(tf.negative(tf.div(distances**2, 2*sigma**2)))

        # hitung rate
        rate = tf.multiply(ns, learning_rate)
        num_of_node = self.height * self.width
        rate_value = tf.stack([tf.tile(tf.slice(rate,[i],[1]), [self.inp_dimension]) for i in range(num_of_node)])

        # hitung update weight-nya
        weight_diff = tf.multiply(rate_value, tf.subtract(tf.stack([input for i in range (num_of_node)]), self.weight))

        updated_weight = tf.add(self.weight, weight_diff)
        return tf.assign(self.weight, updated_weight)
    
    def train(self, dataset, num_of_epoch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range (num_of_epoch):
                for data in dataset:
                    feed = { self.input : data}
                    sess.run([self.updated_weight],feed)
            self.weight_value  = sess.run(self.weight)
            self.node_value = sess.run(self.node)
            cluster = [ [] for i in range(self.width)]
            for i, location in enumerate(self.node_value): # diakan meng-return index-nya
                cluster[int(location[0])].append(self.weight_value[i])
            self.cluster = cluster

data = preprocessing()

print(data.shape)

som = SOM(5,5,3)
som.train(data, 5000)

plt.imshow(som.cluster)
plt.show()