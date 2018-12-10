import tensorflow as tf
import numpy as np
import csv

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
    
def train(train_data):
    #clustering algo: input 3, cluster = i don't know
    
    #5000 epoch
    return 0

def visualize(train_result):
    #use matplotlib
    return 0

data = preprocessing()
print(data[:5])