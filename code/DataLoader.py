import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import os
from keras.applications.densenet import preprocess_input

def load_data(training_data_dir, testing_data_dir, testing_annotation_file_name, target_size=(224, 224)):
    train_images, train_labels = load_training_data(training_data_dir, target_size)
    test_images, test_metadata = load_testing_data(testing_data_dir, testing_annotation_file_name, target_size)
    return train_images, train_labels, test_images, test_metadata

def load_training_data(training_data_dir, target_size=(224, 224)):
    images = []
    labels = []

    for class_folder_name in os.listdir(training_data_dir):
        class_folder_path = os.path.join(training_data_dir, class_folder_name)
        print(f"Processing class: {class_folder_name}")  # Debugging print statement
        if os.path.isdir(class_folder_path):
            annotation_file_name = f'GT-{class_folder_name}.csv'
            annotation_file_path = os.path.join(class_folder_path, annotation_file_name)
            if not os.path.isfile(annotation_file_path):
                continue  # Handle missing annotation file
            annotations = pd.read_csv(annotation_file_path, delimiter=';')
            for _, row in annotations.iterrows():
                image_path = os.path.join(class_folder_path, row['Filename'])
                if not os.path.isfile(image_path):
                    continue
                image = Image.open(image_path)
                image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)  # Resize the image
                image = np.array(image)
                image = preprocess_input(image)  # Normalize for DenseNet
                images.append(image)
                labels.append(row['ClassId'])

    return np.array(images), np.array(labels)

def load_testing_data(testing_data_dir, annotation_file_name, target_size=(224, 224)):
    images = []
    metadata = []

    annotation_file_path = os.path.join(testing_data_dir, annotation_file_name)
    if not os.path.isfile(annotation_file_path):
        return np.array(images), metadata  # Handle missing annotation file

    annotations = pd.read_csv(annotation_file_path, delimiter=';')
    
    for _, row in annotations.iterrows():
        image_path = os.path.join(testing_data_dir, row['Filename'])
        if not os.path.isfile(image_path):
            continue
        image = Image.open(image_path)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image = np.array(image)
        image = preprocess_input(image)  # Normalize for DenseNet
        images.append(image)
        metadata.append(row.to_dict())

    return np.array(images), metadata


def checks():
    train_images, train_labels, test_images, test_metadata = load_data(
        'data/GTSRB_Training_Images/Final_Training/Images', 
        'data/GTSRB_Testing_Images/Final_Test/Images', 
        'GT-final_test.test.csv'
    )


    # Verify the number of training samples
    print("Training data shape:", train_images.shape)
    print("Training labels shape:", train_labels.shape)
    print("Testing data shape:", test_images.shape)
    print("Testing metadata shape:", len(test_metadata))

    # Verify the number of classes
    unique_classes = np.unique(train_labels)
    print("Number of unique classes:", len(unique_classes))

checks()