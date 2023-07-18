import streamlit as st
import os
import io
from datetime import date
import glob as glob
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import base64
from streamlit_image_comparison import image_comparison
from project_utils.page_layout_helper import  main_header
from project_utils.predict_helper import gdal_uploaded_image_array, fetch_model, image_to_rgb, model_result, convert_8_band_to_4_band, convert_to_rgb, predicted_image_with_class_label_plot_style1, predicted_image_with_class_label_plot_style2
 
ALLOWED_EXTENSIONS = [".tiff"]

MODEL_PATH='./Model/TRAINED_MODEL_5.hdf5'

WIDTH=256

HEIGHT=256


@st.cache_resource(show_spinner="Please wait for Model to Load...")
def get_model(path):
  return fetch_model(path)

def modelpredict(model, upload_image_obj):
  return model_result(model, upload_image_obj)

def page_sidebar():
  st.divider()
  st.sidebar.subheader("Upload a ROI Image of Format(.tif) of resolution (256x256) : ")
  #st.sidebar.subheader.markdown("### Upload a ROI Image of Format(.tif) of resolution (256x256) : ")
  return st.sidebar.file_uploader("", type=ALLOWED_EXTENSIONS)


def main():
  main_header()
  uploaded_file =page_sidebar()
  model_obj = get_model(os.path.abspath(MODEL_PATH))
  with st.container():
    if uploaded_file is not None:
      #st.write(np.moveaxis((uploaded_file.getvalue()), 0, -1))
      #st.write(io.StringIO(uploaded_file.getvalue().decode("utf-8")).read())
      #satellite_input_imageobj=Image.open(io.BytesIO(uploaded_file.getvalue()))
      #satellite_input_imageobj=np.array(uploaded_file.getvalue())#.decode("utf-8").read() #Image.open(uploaded_file)
      satellite_input_imageobj=gdal_uploaded_image_array(uploaded_file)
      model_predict_result=modelpredict(model_obj,satellite_input_imageobj)
      predicted_output_imageobj=convert_8_band_to_4_band(model_predict_result)
      col1, col2 = st.columns([6,6], gap="small")
      with col1:
        st.markdown('### **Uploaded Satellite Image**',unsafe_allow_html=True)
        st.image(satellite_input_imageobj,width=WIDTH)  #300 #640
      with col2:
        st.markdown('### **Predicted Image**',unsafe_allow_html=True)
        st.image(predicted_output_imageobj, width=WIDTH) 
      #st.info('## The predicted model accuracy for the uploaded image is 53%')
      st.markdown("<br>", unsafe_allow_html=True)

      image_comparison(
       img1=Image.fromarray(np.uint8((satellite_input_imageobj / satellite_input_imageobj.max()) * 255.)),#image_to_rgb(satellite_input_imageobj),#satellite_input_imageobj,
       img2=Image.fromarray(np.uint8((predicted_output_imageobj / predicted_output_imageobj.max()) * 255.)),#image_to_rgb(predicted_output_imageobj),#model_predict_result),
       label1="Original Satellite Image (8 bit)",
       label2="Predicted Label Image (8 bit)",
       width=700,
       starting_position=50,
       show_labels=True,
       make_responsive=True,
       in_memory=True,
      )
      st.markdown("<br>", unsafe_allow_html=True)

      #st.image(predicted_image_with_class_label(model_predict_result), width=500)
      
      st.markdown("<br>", unsafe_allow_html=True) 
      st.pyplot(predicted_image_with_class_label_plot_style1(model_predict_result))
      #st.plotly_chart(predicted_image_with_class_label(model_predict_result))
      st.markdown("<br>", unsafe_allow_html=True) 
      st.pyplot(predicted_image_with_class_label_plot_style2(model_predict_result))

if __name__ == "__main__":
  main()     
