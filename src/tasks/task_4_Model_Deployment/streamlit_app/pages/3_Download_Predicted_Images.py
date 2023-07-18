
import streamlit as st
from project_utils.page_layout_helper import  main_header
from project_utils.azure_blob_storage_helper import fetch_all_blob_files_list

DOWNLOAD_BUTTON_STYLE="""
<style>
.button {
  display: inline-block;
  padding: 15px 25px;
  font-size: 22px;
  cursor: pointer;
  text-align: center;
  text-decoration: none;
  outline: none;
  color: #fff;
  background-color: #F0F2F6;
  border: none;
  border-radius: 15px;
  box-shadow: 0 9px #999;
  padding: 10px;
  width: 500px;
  transition: all 0.5s;
  cursor: pointer;
  margin: 5px;
}
.button:hover {background-color: #BCCEFB}

.button:active {
  background-color: #BCCEFB';
  box-shadow: 0 5px #666;
  transform: translateY(4px);
}

.button:hover span {
  padding-right: 25px;
}

.button:hover span:after {
  opacity: 1;
  right: 0;
}

.button span {
  cursor: pointer;
  display: inline-block;
  position: relative;
  transition: 0.5s;
}

.button span:after {
  position: absolute;
  opacity: 0;
  top: 0;
  right: -20px;
  transition: 0.5s;
}
</style>
"""

BLOB_LIST={}

@st.cache_data
def list_all_files(province):
  BLOB_LIST=fetch_all_blob_files_list()
  return BLOB_LIST

def call_button_function(selected_value):
  st.session_state.SET_KEY = True 

def main():
  main_header()
  if 'SET_KEY' not in st.session_state:
    st.session_state.SET_KEY = False
  st.write(DOWNLOAD_BUTTON_STYLE, unsafe_allow_html=True)
  BLOB_LIST=fetch_all_blob_files_list()
  with st.container():
    option = st.selectbox('Please select the ROI region option to download the files : ',
        ('Full Camerron region','Adamawa','Central','East','Far North','Littoral','North','Northwest','South','Southwest','West'))
    
    if option=='Full Camerron region':
        selected_value='Full_Predicted_Map_Data'
    else:
        selected_value=option
    
    st.button(label = 'Generate Download Links', key=None, on_click = call_button_function, args=(selected_value,), kwargs=None, disabled=False,)
    
    if st.session_state.SET_KEY:
        if option in ['Full Camerron region','Adamawa','Central','East','Far North','Littoral','North','Northwest','South','Southwest','West']:
            for key, value in BLOB_LIST.items():
                if selected_value == str(key.split("/")[-2]):
                    #html_tag = f'''<a href={value} class="button" style="vertical-align:middle" target="_blank" type="button" aria-pressed="true"><span>{key.split("/")[-1]}</span></a>'''
                    #st.markdown(html_tag, unsafe_allow_html=True)
                    st.markdown(f"<a href='{value}' target='_blank'>{key.split('/')[-1]}</a>", unsafe_allow_html=True)
                    
        else:
            st.write("Please select a valid option")
        st.session_state.SET_KEY=False
    
    
if __name__ == "__main__":
  main()  
