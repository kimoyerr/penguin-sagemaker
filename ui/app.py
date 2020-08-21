"""
This script creates a UI based on Streamlit (https://www.streamlit.io/) for penguin dataset.
Inspired by the following post: https://towardsdatascience.com/building-machine-learning-apps-with-streamlit-667cef3ff509
"""

# Import internal modules
import os
import sys
from os.path import abspath, dirname
import subprocess
import time

# External libraries
import streamlit as st

# Local modules
from ui import train_ui

# Logging
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

project_dir = dirname(dirname(abspath(__file__)))
os.chdir(project_dir)

BASE_URL = "http://0.0.0.0:8080/"
PING_URL = BASE_URL + "ping"

# Main Page sidebar
st.image(os.path.join(project_dir, 'ui', 'images', 'data-original.png'), use_column_width=True)
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
    train_ui.page()

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

