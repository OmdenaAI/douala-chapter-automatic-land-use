import ee
import os
#import geemap
import json
import streamlit as st
import geemap.foliumap as geemap
import geopandas as gpd
from project_utils.page_layout_helper import  main_header
main_header()
AOI_GEOJSON = st.secrets["cameroon_aoi_bbx"] 
PREDICTED_IMAGE_ASSET_PATH = st.secrets["predicted_image_asset"]
JSON_DATA = st.secrets["google_map_auth_json_data"]

json_object = json.loads(JSON_DATA, strict=False)
json_object = json.dumps(json_object)
os.environ['EARTHENGINE_TOKEN']=json_object
geemap.ee_initialize(token_name='EARTHENGINE_TOKEN', auth_mode='notebook', service_account=True)
m = geemap.Map(center=(7.3696, 12.3446), zoom=6)
image=ee.Image(PREDICTED_IMAGE_ASSET_PATH)
#image=ee.Image("https://code.earthengine.google.com/?asset=users/roygeoai/OMDENACMRLC2021")
databox = gpd.read_file(os.path.abspath(AOI_GEOJSON))
aoi = geemap.geopandas_to_ee(databox).geometry()
m.centerObject(aoi,6)
m.addLayer(image)
m.addLayerControl()
m.to_streamlit()
