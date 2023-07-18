import streamlit as st
from azure.storage.blob import BlobServiceClient

def fetch_all_blob_files_list():
  file_list={}
  STORAGE_CREDENTIAL = st.secrets["azure_blob_storage"]["storage_credential_token"]
  STORAGE_ACCOUNT_URL = st.secrets["azure_blob_storage"]["storage_resource_url"]
  STORAGE_CONTAINER_NAME = 'iot'
  prefix_value="final_output/"
  service = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=STORAGE_CREDENTIAL)
  container_client = service.get_container_client(STORAGE_CONTAINER_NAME)
  blobs = container_client.list_blobs(name_starts_with=prefix_value)
  #blobs = container_client.list_blobs()
  for blob in blobs:
    file_list[blob.name]=container_client.get_blob_client(blob.name).url
  return file_list
