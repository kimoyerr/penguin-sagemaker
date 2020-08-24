"""
This script creates the scatter plots using Bokeh

"""

# Internal libraries

from math import pi

# External libraries
import numpy as np
import streamlit as st

# Bokeh
from bokeh.io import show
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap
from bokeh.models import Legend, LegendItem


def plot_scatter(df, x, y):
    """

    :return:
    """
    # First change some column names
    tmp = df.copy()

    tmp.columns = tmp.columns.str.replace('.','_')
    x_orig = x
    y_orig = y
    x = x.replace('.','_')
    y = y.replace('.', '_')

    tmp.columns = tmp.columns.str.replace('-', '_')
    x = x.replace('-','_')
    y = y.replace('-', '_')

    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
    TOOLTIPS = [("index", "$index")]
    for col in tmp.columns:
        TOOLTIPS.append(tuple((col, "@"+col)))

    p = figure(plot_width=900, plot_height=600, tools=TOOLS, tooltips=TOOLTIPS, toolbar_location='below')
    p.circle(x.replace('-','_'),y.replace('-','_'), size=6, color='plum', legend='circle', source=tmp)
    p.title.text = 'Design Counts vs Amino Acid Distances'
    p.xaxis.axis_label = x_orig
    p.yaxis.axis_label = y_orig
    st.bokeh_chart(p, use_container_width=True)