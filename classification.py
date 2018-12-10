import tensorflow as tf
import numpy as np
import csv

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#===classification===

label_name = ['a', 'b', 'c', 'd', 'e']

number_of_input = 5 # number of pca dimension
number_of_output = 5 # number of unique output (a, b, c, d, e)

x = tf.placeholder(tf.float32, [None, number_of_input])
y = tf.placeholder(tf.float32, [None, number_of_output])

lr = 0.01
epochs = 5000

model_path = 'model/checkpoint.ckpt'

def load_data():
    dataset = []

    with open('O192-COMP7117-AS01-00-classification.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(readCSV):
            features = []
            # do something here with the data

            label = row[4]

            features.append(row[5])
            features.append(row[6])
            features.append(row[7])
            features.append(row[8])
            features.append(row[9])
            features.append(row[10])
            features.append(row[12])
            features.append(row[12])
            features.append(row[14])
            features.append(row[15])
            features.append(row[17])
            features.append(row[18])
            features.append(row[19])
            features.append(row[20])
            features.append(row[23])

            # one data = ([features], label)
            if i != 0:
                # don't insert the first data
                dataset.append((features, label))
    
    return dataset

def preprocessing():
    data = load_data()
    feature_list = []
    label_list = []
    processed_data = []

    for feat, label in data:
        feature_list.append(feat)
        label_list.append(label)

    np_features = np.array(feature_list)

    #===process features===
    # normalize here
    scaler = StandardScaler()
    normalize_res = scaler.fit_transform(np_features)

    # then do pca here
    pca = PCA(n_components=5)
    pca_res = pca.fit_transform(normalize_res)

    #===process labels===
    # turn it into one hot
    label_matrix = np.zeros((len(label_list), len(label_name)))
    for first_dim, one_label in enumerate(label_list):
        for second_dim, one_label_name in enumerate(label_name):
            if one_label == one_label_name:
                label_matrix[first_dim, second_dim] = 1
        
    # stitch the data then return it
    for i in range(len(label_list)):
        processed_data.append((pca_res[i], label_matrix[i]))
    
    data_numpy = np.array(processed_data)
    np.random.shuffle(data_numpy)

    return data_numpy

def fully_connected(inp, in_degree, out_degree):
    w = tf.Variable(tf.random_normal([in_degree, out_degree]))
    b = tf.Variable(tf.random_normal([out_degree]))

    z = tf.matmul(inp, w) + b
    a = tf.nn.sigmoid(z)
    return a

def build_model (inp):
    number_of_hidden = [5, 10, 3] # TODO: edit this line

    layer_1 = fully_connected(inp, number_of_input, number_of_hidden[0])
    layer_2 = fully_connected(layer_1, number_of_hidden[0], number_of_output)

    # you also can add another layer if you wish

    return layer_2   

def train(model, train, validation, test):
    err = tf.reduce_mean(.5*(y-model)**2)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(err)
    correct_prediction= tf.equal(tf.argmax(model,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epochs):
            train_dictionary = {x:[i[0]for i in train], 
                y:[i[1] for i in train]}

            validation_dictionary = {x:[i[0] for i in validation],
                y:[i[1] for i in validation]}
                
            # code to train
            _, train_loss, train_acc = sess.run([optimizer, err, accuracy], feed_dict=train_dictionary)
            if i%100 == 0:
                print(f'epoch: {i}, train loss: {train_loss}, train acc:{train_acc}')

            # every 500 epoch do something
            if i%500 == 0 and i!=0:
                if i == 500:
                    # code to evaluate
                    val_loss, val_acc = sess.run([err, accuracy], validation_dictionary)
                    best_val_loss = val_loss
                    print(f'currently epoch {i}, val loss is {val_loss}, val acc {val_acc}, saving model...')
                    saver.save(sess, model_path)
                
                else:
                    # compare to previous loss
                    val_loss, val_acc = sess.run([err, accuracy], validation_dictionary)
                    if val_loss < best_val_loss:
                        print(f'currently epoch {i}, val loss is {val_loss}, val acc {val_acc}, saving model...')
                        saver.save(sess, model_path)
                        best_val_loss = val_loss
                    else:
                        print(f'currently epoch {i}, val loss is {val_loss}, not saving model...')

        test_dictionary = {x:[i[0] for i in test],
            y:[i[1] for i in test]}

        test_loss, test_acc = sess.run([err, accuracy], test_dictionary)
        print(f'Performance on test data. loss: {test_loss}, acc: {test_acc}')

data = preprocessing()
total_data = data.shape[0]

hm_train = int(.7*total_data)
hm_val = int(.2*total_data)
hm_test = int(.1*total_data)

train_data = data[:hm_train]
val_data = data[hm_train:hm_train+hm_val]
test_data = data[-hm_test:]

train(build_model(x), train_data, val_data, test_data)