import streamlit as st
from PIL import Image
import os

from lr import lr_main
from dt import dt_main
from knn import knn_main
from nb import nb_main
from svm import svm_main

def main():
    
    icon = Image.open('favicon.ico')
    st.set_page_config(
        layout="centered",
        initial_sidebar_state="expanded",
        page_title='DummyLearn.com',
        page_icon=icon
    )

    hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    """
    st.markdown(hide_footer_style, unsafe_allow_html=True)

    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    image = Image.open('logo.png')
    st.sidebar.image(image)

    pages_ml_classifier = {
        'Logistic Regression Classifier': lr_main,
        'Decision Tree Classifier': dt_main,
        'K-Nearest Neighbors Classifier': knn_main,
        'Naive Bayes Classifier': nb_main,
        'Support Vector Machine Classifier': svm_main
        }
    
    ml_module_selection =  st.sidebar.selectbox('Select Classifier',['Logistic Regression Classifier',
                                                                  'Decision Tree Classifier',
                                                                  'K-Nearest Neighbors Classifier',
                                                                  'Naive Bayes Classifier',
                                                                  'Support Vector Machine Classifier'])
    
    
    pages_ml_classifier[ml_module_selection]()

if __name__ == '__main__':
    main()
