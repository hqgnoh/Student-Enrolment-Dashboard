import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
# import database as db
import sqlite3

st.set_page_config(
    page_title="Student Enrollment Dashboard",
    page_icon=""
)

st.title('University Student Enrollment Dashboard')

def creds_entered():
    if st.session_state["user"].strip() == "admin" and st.session_state["passwd"].strip() == "admin":
        st.session_state["authenticated"] = True
    else:
        st.session_state["authenticated"] = False
        if not st.session_state["passwd"]:
            st.warning("Please enter password")
        elif not st.session_state["user"]:
            st.warning("Please enter username")
        else:
            st.error("Invalid Username/Password")

def authenticate_user():
    if "authenticated" not in st.session_state:
        st.header("Login Page")
        st.info("Please press ENTER to Login",icon=None)
        st.text_input(label="Username :", value="", key="user", on_change=creds_entered)
        st.text_input(label="Password :", value="", key="passwd", type="password", on_change=creds_entered)
        return False
    else:
        if st.session_state["authenticated"]:
            return True
        else:
            st.header("Login Page")
            st.info("Please press ENTER to Login",icon=None)
            st.text_input(label="Username :", value="", key="user", on_change=creds_entered)
            st.text_input(label="Password :", value="", key="passwd", type="password", on_change=creds_entered)
            return False

if authenticate_user():
        st.sidebar.success("Welcome admin!")

        #image = Image.open("/content/drive/MyDrive/ColabNotebooks/UCSI-Logo.png")
        #st.sidebar.image(image)

        with st.sidebar:
            selected = option_menu("List of Faculties", ["General", "Faculty of Humanities and Social Sciences", "Faculty of Business", "Faculty of Music", "Faculty of Architecture", "Faculty of Applied Sciences", "Faculty of Engineering", "Faculty of Medicine", "Faculty of Computer Sciences"],
            icons = ['globe','people','graph-up','music-note-beamed','building','eyedropper','gear','capsule','laptop'], menu_icon="list", default_index=0)

        def fetch_student_data():
            conn = sqlite3.connect("student_enrollment.db")
            select_query = "SELECT * FROM admission"
            try:
                data = pd.read_sql_query(select_query, conn)
                return data
            except sqlite3.Error as e:
                st.error(f"Error fetching data: {e}")
                return pd.DataFrame()

        st.header("Student Enrollment Data")

        student_data = fetch_student_data()
        st.dataframe(student_data)

# # User Authentication
# users = db.fetch_all_users()
#
# usernames = [user["key"] for user in users]
# names = [user["name"] for user in users]
# hashed_passwords = [user["password"] for user in users]
#
# authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
#     "enrollment_dashboard", "abcdef")
#
# names, authentication_status, username = authenticator.login("Login", "main")
#
# if authentication_status == False:
#     st.error("Username or Password is incorrect")
#
# if authentication_status == None:
#     st.warning("Please enter your username and password")
#
# if authentication_status:
#
#     authenticator.logout("Logout", "sidebar")
#     st.sidebar.title(f"Welcome {name}")
