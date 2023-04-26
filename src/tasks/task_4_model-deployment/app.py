# import necessary libraries
# for deployment
from flask import Flask,render_template,request

# for data manipulation
import pandas as pd
# for mathematical computations
import numpy as np
#for visualization
import matplotlib.pyplot as plt
import seaborn as sns


# for deep learning
import keras
import tensorflow
from keras.models import load_model


#GIS 
import gdal
import rasterio

# instatiate the flask app
app = Flask(__name__)


# Define a function that takes a "file_path" and returns a "tif" file as a 3-dimensional numpy array as: WIDTH X HEIGHT X BANDS
def gtiff_to_array(file_path):
    data = gdal.Open(file_path)
    bands = [data.GetRasterBand(i+1).ReadAsArray() for i in range(data.RasterCount)]
    return np.stack(bands, axis=2)

# Set up the patch size and identification of the number of bands
patch_size = 256
nbands = 4 

# Define a function for converting images and labels into patches of 256 x 256
def gridwise_sample(imgarray, patchsize):
    nrows, ncols, nbands = imgarray.shape
    patchsamples = np.zeros(shape=(0, patchsize, patchsize, nbands), dtype=imgarray.dtype)
    for i in range(int(nrows/patchsize)):
        for j in range(int(ncols/patchsize)):
            tocat = imgarray[i * patchsize:(i + 1)*patchsize,
                             j * patchsize:(j + 1)*patchsize, :]
            tocat = np.nan_to_num(tocat, nan=np.nanmedian(tocat))
            tocat[tocat < -1] = np.median(tocat)
            tocat = np.expand_dims(tocat, axis=0)
            patchsamples = np.concatenate((patchsamples, tocat), axis=0)
    return patchsamples


# Load the trained model
model = load_model('TRAINED_MODEL_4.hdf5')

# Define a function to preprocess the uploaded image and make predictions
def preprocess_image_and_predict(file_path):
    # Load the uploaded image as a numpy array
    uploaded_image_array = gtiff_to_array(file_path)

    # Convert the image into patches of 256 x 256
    image_patches = gridwise_sample(uploaded_image_array, patch_size)

    # Perform predictions on each patch
    num_patches = image_patches.shape[0]
    predicted_patches = np.zeros(shape=(num_patches, patch_size, patch_size, 1), dtype=np.uint8)
    for i in range(num_patches):
        # Extract the patch and transpose it to have the channel dimension as the last axis
        patch = image_patches[i, :, :, :]

        # Perform prediction on the patch
        prediction = model.predict(np.expand_dims(patch, axis=0))

        # Store the predicted patch
        predicted_patches[i, :, :, :] = prediction[0]

    # Reshape predicted patches array into a 3D array with the same shape as the original image
    nrows, ncols, nbands = uploaded_image_array.shape
    num_patches_x = int(nrows / patch_size)
    num_patches_y = int(ncols / patch_size)
    stride = patch_size
    predicted_image_array = np.zeros((nrows, ncols, 1), dtype=np.uint8)
    count_array = np.zeros((nrows, ncols))
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            # Compute the coordinates of the patch
            x1 = i * stride
            y1 = j * stride
            x2 = x1 + patch_size
            y2 = y1 + patch_size

            # Add the predicted patch to the predicted image
            predicted_image_array[x1:x2, y1:y2, :] += predicted_patches[i*num_patches_y+j, :, :, :]
            count_array[x1:x2, y1:y2] += 1

    # Compute the mean of the predicted patches where the count is non-zero
    predicted_image_array = predicted_image_array / np.expand_dims(count_array, axis=-1)

    # Create a dictionary to map class numbers to labels
    class_labels = {
        10: "Tree Cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-Up",
        60: "Bare / Sparse Vegetation",
        80: "Permanent Water Bodies",
        90: "Herbaceous Wetland",
    }

    # Convert predicted image array to a class label array
    class_label_array = np.argmax(predicted_image_array, axis=-1)

    # Map class numbers to labels
    for class_num, class_label in class_labels.items():
        class_label_array[class_label_array == class_num] = class_label

    return class_label_array
# define home page
@app.route('/')
@app.route('/home')
def Home_page():
    return render_template('home.html')

# define prediction page
@app.route('/prediction', methods=['GET', 'POST'])
def Prediction_page():
    if request.method == 'POST':
        # handle file upload here
        file = request.files['file']
        # preprocess the uploaded image and make predictions
        class_label_array = preprocess_image_and_predict(file)

        # create a dictionary to store the percentage score of each class
        score_dict = {}
        for class_num, class_label in class_labels.items():
            score_dict[class_label] = round((np.count_nonzero(class_label_array == class_label) / class_label_array.size) * 100, 2)

        # render the prediction template and pass the predicted class labels and their percentage scores as arguments
        return render_template('prediction.html', class_labels=class_labels, class_label_array=class_label_array, score_dict=score_dict)
    else:
        return render_template('prediction.html')



     
if __name__ == '__main__':
    app.run(debug=True)