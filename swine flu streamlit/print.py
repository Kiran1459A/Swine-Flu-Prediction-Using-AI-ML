from subprocess import call
import streamlit as st
from streamlit_option_menu import option_menu
import app
v=option_menu(menu_title='pond',
    options=['home','account']
)

if v=='home':
    app.main()