import streamlit as st

from src.ui.base_layout import style_backgroud_dashboard,style_base_layout
from src.components.header import header_dashboard
from src.components.footer import footer_dashboard

def student_screen():
    st.header("Student Screen!!")