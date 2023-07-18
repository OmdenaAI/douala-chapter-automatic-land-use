from tensorflow.keras.models import load_model
from PIL import Image
import os
import io
import numpy as np
import pandas as pd
import gdown
#from osgeo import gdal
import rasterio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.colors
import streamlit as st

# Set the patch size and stride
PATCH_SIZE = 256
STRIDE = 128
LABEL_CLASS_WITH_COLOR = {
    10: {"label_class":"Tree Cover", "label_color":"#006400"},
    20: {"label_class":"Shrubland", "label_color":"#FFBB22"},
    30: {"label_class":"Grassland", "label_color":"#FFFF4C"},
    40: {"label_class":"Cropland", "label_color":"#F096FF"},
    50: {"label_class":"Built-Up", "label_color":"#FA0000"},
    60: {"label_class":"Bare / Sparse Vegetation", "label_color":"#B4B4B4"},
    80: {"label_class":"Permanent Water Bodies", "label_color":"#0064C8"},
    90: {"label_class":"Herbaceous Wetland", "label_color":"#0096A0"},
  }

def get_file_gdrive(input_url,output_filename):
  return gdown.download(input_url, output_filename, quiet=False)

def fetch_model(modelpath):
  if os.path.exists(os.path.abspath(modelpath)):
    model=load_model(modelpath, compile=False)
  else:
    input_url = 'https://drive.google.com/uc?id=1O6Cb0-_Tz9Ra4S976owgYSbWSRqvaMJy'
    #output_filename = '/content/drive/MyDrive/Omdena_Projects/Automating_Land_Use_and_Land_Cover_Mapping_StreamApp/models/TRAINED_MODEL_5.hdf5'
    os.mkdir(os.path.dirname(modelpath).replace('.','')) 
    get_file_gdrive(input_url, modelpath)
    model=load_model(modelpath, compile=False)
  return model

def gdal_uploaded_image_array(upload_image_obj):
  #data_array = upload_image_obj.read()
  #drv = gdal.GetDriverByName("GTiff")
  #gdal_image = drv.Create("256.tif", 256, 256, 4, gdal.GDT_Byte)
  #gdal_image = None
  #gdal.FileFromMemBuffer("/vsimem/256.tif", data_array)
  #gdal_image = gdal.Open("/vsimem/256.tif")
  #gdal_image_array = np.transpose(gdal_image.ReadAsArray(), (1, 2, 0))
  #gdal_image = None
  #if os.path.exists(os.path.abspath("256.tif")):
  #  os.remove(os.path.abspath("256.tif"))
  save_path='./Model/temp.tif'
  with open(save_path, mode='wb') as w:
    w.write(upload_image_obj.getvalue())    
  ds_raster = rasterio.open(save_path)
  #st.header(str(type(ds_raster)) + "  " + str(ds_raster.shape) + "  " + str(ds_raster.bounds))
  gdal_image_array = np.transpose(ds_raster.read(), (1, 2, 0))
  return gdal_image_array

def model_result(model, gdal_image_array):
  num_patches_x = int(np.ceil((gdal_image_array.shape[0] - PATCH_SIZE) / STRIDE)) + 1
  num_patches_y = int(np.ceil((gdal_image_array.shape[1] - PATCH_SIZE) / STRIDE)) + 1
  num_patches = num_patches_x * num_patches_y

  predicted_patches = np.zeros((num_patches, PATCH_SIZE, PATCH_SIZE, 8))

  for i in range(num_patches_x):
      for j in range(num_patches_y):
          x1 = i * STRIDE
          y1 = j * STRIDE
          x2 = x1 + PATCH_SIZE
          y2 = y1 + PATCH_SIZE
          patch = gdal_image_array[x1:x2, y1:y2, :]
          prediction = model.predict(np.expand_dims(patch, axis=0))
          predicted_patches[i*num_patches_y+j, :, :, :] = prediction[0]
  
  predicted_image_array = np.zeros((gdal_image_array.shape[0], gdal_image_array.shape[1], 8))
  count_array = np.zeros((gdal_image_array.shape[0], gdal_image_array.shape[1]))
  for i in range(num_patches_x):
      for j in range(num_patches_y):
          x1 = i * STRIDE
          y1 = j * STRIDE
          x2 = x1 + PATCH_SIZE
          y2 = y1 + PATCH_SIZE
          predicted_image_array[x1:x2, y1:y2, :] += predicted_patches[i*num_patches_y+j, :, :, :]
          count_array[x1:x2, y1:y2] += 1

  predicted_image_array = predicted_image_array / np.expand_dims(count_array, axis=-1)
  return predicted_image_array

def image_to_rgb(predicted_image_array):
  return Image.fromarray(np.uint8(predicted_image_array.argmax(axis=-1)))

def convert_8_band_to_4_band(predicted_image_band):
  combined_dstack=np.dstack((predicted_image_band[:, :, 0], predicted_image_band[:, :, 1], predicted_image_band[:, :, 2], predicted_image_band[:, :, 3] ))
  return combined_dstack


def convert_to_rgb(predicted_image_array):
  predicted_class_indices = np.argmax(predicted_image_array, axis=-1)
  # Map predicted class indices to original values
  mapping_reverse = {0: 10, 1: 20, 2: 30, 3: 40, 4: 50, 5: 60, 6: 80, 7: 90}
  map_predicted = np.vectorize(mapping_reverse.get)
  predicted_class_indices_mapped = map_predicted(predicted_class_indices)
  return predicted_class_indices_mapped

def predicted_image_with_class_label_plot_style1(predicted_image_array, fig_width = 5, fig_height = 5):
  predicted_class_indices_mapped=convert_to_rgb(predicted_image_array)
  fig = plt.figure(figsize = (fig_width, fig_height))
  figplot = fig.add_subplot(1, 1, 1)
  values = np.unique(predicted_class_indices_mapped.ravel())
  #im = plt.imshow(predicted_class_indices_mapped[:, :, 0])
  im = plt.imshow(predicted_class_indices_mapped)
  figplot.set_title("Predicted Image Label Classes with Color Codes : ")

  patches = [mpatches.Patch(color = LABEL_CLASS_WITH_COLOR[i]["label_color"], label = LABEL_CLASS_WITH_COLOR[i]["label_class"]) for i in LABEL_CLASS_WITH_COLOR]
  plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  return fig

def predicted_image_with_class_label_plot_style2(predicted_image_array, fig_width = 7, fig_height = 6):
  predicted_class_indices_mapped=convert_to_rgb(predicted_image_array)
  fig = plt.figure(figsize = (fig_width, fig_height))
  plt.title("Predicted Image Label Classes with Color Codes : ")
  colors = ["#000000" for r in range(256)]
  for key, value in LABEL_CLASS_WITH_COLOR.items():
    colors[key] = value["label_color"]

  cmap = matplotlib.colors.ListedColormap(colors)
  im = plt.imshow(predicted_class_indices_mapped)#[:, :, 0])
  normalizer = matplotlib.colors.Normalize(vmin=0, vmax=255)
  values = [key for key in LABEL_CLASS_WITH_COLOR]
  boundaries = [(values[i + 1] + values[i]) / 2 for i in range(len(values) - 1)]
  boundaries = [0] + boundaries + [255]
  ticks = [(boundaries[i + 1] + boundaries[i]) / 2 for i in range(len(boundaries) - 1)]
  tick_labels = [LABEL_CLASS_WITH_COLOR[i]["label_class"] for i in LABEL_CLASS_WITH_COLOR]
  colorbar = plt.colorbar(
   mappable=cm.ScalarMappable(norm=normalizer, cmap=cmap),
    boundaries=boundaries,
    values=values,)
  colorbar.set_ticks(ticks, labels=tick_labels)
  return fig
