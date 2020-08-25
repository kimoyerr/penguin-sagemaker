"""
This script creates a UI based on Streamlit (https://www.streamlit.io/) for penguin dataset.
Inspired by the following post: https://towardsdatascience.com/building-machine-learning-apps-with-streamlit-667cef3ff509
"""

# Import internal modules
import os
from os.path import abspath, dirname

# External libraries
import streamlit as st
import pandas as pd

# Local modules
from penguin.ui import train_ui, SessionState

# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

project_dir = dirname(abspath(__file__))
os.chdir(project_dir)

# Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

BASE_URL = "http://0.0.0.0:8080/"
PING_URL = BASE_URL + "ping"

# Session state
state = SessionState.get(mlflow_res=pd.DataFrame(), best_train_submit_button=False, sel_best_run=0)

# Main Page sidebar
st.image(os.path.join(project_dir, 'penguin', 'ui', 'images', 'data-original.png'), use_column_width=True)
st.sidebar.subheader('Penguin ')
ml_steps = ['Get Started', 'Train', 'Inference']
sel_step = st.sidebar.radio('Workflow Steps', ml_steps)

# Page: Get Started
if sel_step == 'Get Started':
    st.title('Penguins Dataset')
    st.markdown("""
            Streamlit-COA is a module to use the [streamlit tool] (https://www.streamlit.io/) and build a quick but clean UI
            """)

    st.markdown("""
    [Original Penguins Repo](https://github.com/allisonhorst/palmerpenguins)
    """)


########################

if sel_step == 'Train':
    train_ui.page(state)

if sel_step == 'Inference':
    build_mode = st.sidebar.radio('Mode', ['Upload', 'Simulate'])
    if build_mode == 'Upload':
        build.page(session)
    else:
        simulate.genotype(session)

if sel_step == 'Test':
    test_mode = st.sidebar.radio('Mode', ['Upload', 'Simulate'])
    if test_mode == 'Upload':
        test.page(session)
    else:
        simulate.phenotype(session)

if sel_step == 'Learn':
    learn.page(session)

