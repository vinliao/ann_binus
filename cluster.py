import tensorflow as tf
import numpy as np
import csv

#===som===
"""
feature selection
feature extraction (PCA)

cluster (apply som algorithm or some shit)
train it, you need to follow the specification on docx
"""

def load_data():
    dataset = []

    with open('O192-COMP7117-AS01-00-clustering.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            #dataset.append(row)

            calorie_density = (row[4] + row[5]) / 100
            fat_ratio = (row[8] + row[9]) / (row[7] + row[11])
            if fat_ratio == 0:
                fat_ratio = 1
            sugar = row[14]
            protein = row[16]
            salt = row[17]

            dataset.append([calorie_density, fat_ratio, sugar, protein, salt])
    
    #do something here
    #maybe do pca here idk brah

    return dataset

load_data()

def select_feature(data):
    #create new features here
    """
    feature[0] = something
    feature[1] = something
    ...
    """
    
    #perform pca here
    #take the three highest score of pca

    return data
    
def train(train_data):
    #clustering algo: input 3, cluster = i don't know
    
    #5000 epoch
    return 0

def visualize(train_result):
    #use matplotlib
    return 0
