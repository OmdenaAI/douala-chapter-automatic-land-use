
import streamlit as st
from datetime import date
import os
import glob as glob
from project_utils.page_layout_helper import set_page_settings, get_page_title, main_header


def active_contributors():
  ACTIVE_CONTRIBUTORS_PAGE_CHAPTERLEAD='''
| Chapter Name | Project Coordinator Name |
|--|--|
| Cameroon Chapter Lead | YOUNKAP NINA Duplex |
'''
  
  ACTIVE_CONTRIBUTORS_PAGE_TASKLEAD='''
| Task Name | Task Lead Name |
|--|--|
| Data Selection | Daria Akhbari |
| Data Collection | Kaushik Roy |
| Data Preprocessing and Visualization | Noelia | 
| Model Development and Training | Akhil Chibber |
| Deployment and Dashboard | Vinod Cherian |
'''

  ACTIVE_CONTRIBUTORS_PAGE_MEMBERS_LIST='''
| Active Contributors Names in alphabetical (a-z) orders |
|--|
| Abhi Agarwal, Akhil Chibber, Daria Akhbari, Deepali, Deepanshu Rajput, Elias Dzobo, Getrude Obwoge, Joseph N. Moturi, Kaushik Roy, Noelia, Rayy Benhin, Sanjiv, Sugandaram M, Vinod Cherian, Yaninthé |
'''

  with st.container():
    st.markdown(ACTIVE_CONTRIBUTORS_PAGE_CHAPTERLEAD, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(ACTIVE_CONTRIBUTORS_PAGE_TASKLEAD, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(ACTIVE_CONTRIBUTORS_PAGE_MEMBERS_LIST, unsafe_allow_html=True)


def about_project():
  ABOUT_PROJECT_CONTENT='''
#####  This project is initiated by the Omdena Cameroon Chapter to solve Real World Problems.
### The problem
Mapping the extent of land use and land cover categories over time is essential for better environmental monitoring, urban planning, nature protection, conflict prevention, disaster reduction, rescue planning as well as long-term climate adaptation efforts.
This initiative’s goal is to build a Machine Learning model that accurately classifies Land Use and Land Cover (LULC) in satellite imagery. Then use the trained model to automatically generate the LULC map for a region of interest. Finally, create a Web GIS dashboard containing the LULC Map of the region of interest.
The project results will be made open source. The aim is to help connect local organizations and communities to use AI tools and Earth Observations data as an action to cope with local challenges such as land use monitoring and the world’s most critical challenges like climate change. We also hope to encourage citizen science by open-source the dataset and code.
### The goals
The goals of this project are: 
+ The Web GIS dashboard containing LULC Map of the region of interest.
+ The ML models(s) with best performance.
+ The datasets collected during the project on Google Drive for open access.
+ GitHub Repo with Well-documented open source code.
+ Documentation of the work and approach.
'''

  with st.container():
    st.markdown(ABOUT_PROJECT_CONTENT, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def about_project_style():
  ABOUT_PROJECT_STYLE='''
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 24px;
  padding: 0px 20px;
  background-color: rgb(240, 242, 246);
}
</style>
'''
  st.write(ABOUT_PROJECT_STYLE, unsafe_allow_html=True)


def main():
  set_page_settings()
  main_header()
  about_project_style()
  project_tab, team_tab = st.tabs(["  **About Project** ", "  **Active Team Contributors**  "])

  with project_tab:
    about_project()
   
  with team_tab:
    active_contributors()

if __name__ == "__main__":
  main()   
