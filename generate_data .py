# -*- coding: utf-8 -*-
"""generate_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zDtHQvRSei74I56tpZR4iIlLu8COOYn6
"""


from tensorflow.keras.preprocessing import  image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
from imutils import paths
import random
import os
import shutil


def get_data_generator():
    class1 = "malaria/cell_images/Parasitized"
    class2 = "malaria/cell_images/Uninfected"
    dataset_path = "malaria/cell_images"
    training_path = "malaria/training"
    testing_path = "malaria/testing"
    validation_path = "malaria/validation"

    training_split = 0.8
    validation_split = 0.1
    bs = 32                 # batch_size ==> bs

    image_path = list( paths.list_images(class1))
    image_path = image_path + list( paths.list_images(class2))
    random.seed(42)
    random.shuffle(image_path)

    i = int( len(image_path) * training_split)
    # train_path is the list of images for training data
    # test_path is the list of images for testing data
    train_path = image_path[:i]
    test_path = image_path[i:]
    # further spliting training data into validation and training data
    # val_path is the list of images for validation data
    # train_path is the list of images fro training data
    i = int( len(train_path) * validation_split)
    val_path = train_path[:i]
    train_path = train_path[i:]
    # defining training/ testing/ validation dataset
    datasets = [ 
                ("training", train_path, training_path),
                ("validation", val_path, validation_path),
                ("testing", test_path, testing_path)
                ]

    image_size = [224, 224]
    train_data_generator = ImageDataGenerator( rescale = 1/255, 
                                            shear_range = 0.2,
                                            zoom_range = 0.2,
                                            horizontal_flip = True
                                          )
    data_generator = ImageDataGenerator( rescale = 1/255)

    testing_data = data_generator.flow_from_directory(
          testing_path, 
          target_size = image_size, 
          class_mode = 'categorical',
          batch_size = bs,
          shuffle=False
        )
    
    validation_data = data_generator.flow_from_directory(
          validation_path, 
          target_size = image_size, 
          class_mode = 'categorical',
          batch_size = bs,
          shuffle=False
        )

    training_data = train_data_generator.flow_from_directory(
        training_path,
        target_size = image_size, 
        class_mode = 'categorical',
        batch_size = bs,
        shuffle=True
      )

    return training_data,validation_data,testing_data