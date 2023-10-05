import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from pathlib import Path
import streamlit_authenticator as stauth
import plotly.express as px
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
        st.sidebar.title("Home Page")
        st.sidebar.subheader("Select Filters")
        st.sidebar.caption("Note: Select the desired program to view")

        #image = Image.open("/content/drive/MyDrive/ColabNotebooks/UCSI-Logo.png")
        #st.sidebar.image(image)

        # with st.sidebar:
        #     selected = option_menu("List of Faculties", ["General", "Faculty of Humanities and Social Sciences", "Faculty of Business", "Faculty of Music", "Faculty of Architecture", "Faculty of Applied Sciences", "Faculty of Engineering", "Faculty of Medicine", "Faculty of Computer Sciences"],
        #     icons = ['globe','people','graph-up','music-note-beamed','building','eyedropper','gear','capsule','laptop'], menu_icon="list", default_index=0)

        def fetch_student_data():
            conn = sqlite3.connect("student_enrollment.db")
            select_query = "SELECT * FROM admissions;"
            try:
                data = pd.read_sql_query(select_query, conn)
                return data
            except sqlite3.Error as e:
                st.error(f"Error fetching data: {e}")
                return pd.DataFrame()

        st.header("Overall Current Student Enrollment Data")

        def fetch_student_data_metric(faculty):
            conn = sqlite3.connect("student_enrollment.db")
            select_query = f"SELECT Intake, Enrollment FROM admissions WHERE faculty = '{faculty}' AND Intake >= 2020;"
            try:
                data = pd.read_sql_query(select_query, conn)
                return data
            except sqlite3.Error as e:
                st.error(f"Error fetching data: {e}")
                return pd.DataFrame()

        # List of faculties
        faculties = ["Faculty of Business", "Faculty of Engineering", "Faculty of Computer Sciences"]

        # Create three columns
        col1, col2, col3 = st.columns(3)

        # Display the total enrollment for each faculty in Streamlit metrics
        for i, faculty in enumerate(faculties):
            student_data = fetch_student_data_metric(faculty)
            total_enrollment = student_data["Enrollment"].sum()
            col = [col1, col2, col3][i]
            col.metric(faculty, f"{total_enrollment}", )

        st.caption("Note: The above figures are student enrolment data from 2020 to current")
        # st.header("Analysis of Current Student Enrollment Data")

        student_data = fetch_student_data()
        df = pd.DataFrame(student_data)

        # Widget for selecting programs to filter
        selected_programs = st.sidebar.multiselect("Select Programs:", df["Program"].unique())

        # Filter data based on the selected programs
        filtered_data = df[df["Program"].isin(selected_programs)]

        # Extract the year and month parts from the intake date strings and convert to integers
        df["Year"] = df["Intake"].str.split("-").str[0].astype(int)
        df["Month"] = df["Intake"].str.split("-").str[1].astype(int)

        # Filter data based on the selected programs
        filtered_data = df[df["Program"].isin(selected_programs)]

        # Widget for selecting the range of years to calculate the sum
        st.sidebar.caption("Note: Select start, end years and intakes to auto calculate total enrollment numbers")
        start_year = st.sidebar.number_input("Start Year:", min_value=int(min(df["Year"])), max_value=int(max(df["Year"])))
        end_year = st.sidebar.number_input("End Year:", min_value=start_year, max_value=int(max(df["Year"])))

        # Create a mapping of month numbers to month names
        month_names = {
            1: "January",
            5: "May",
            9: "September",
        }

        # Widget for selecting specific months using month names
        allowed_months = [1, 5, 9]
        selected_month_numbers = st.sidebar.multiselect("Select Intakes:", allowed_months, format_func=lambda x: month_names[x])

        # Calculate the sum of enrollment numbers in the selected range and months
        filtered_data_within_range = filtered_data[
            (filtered_data["Year"] >= start_year) & (filtered_data["Year"] <= end_year) &
            (filtered_data["Month"].isin(selected_month_numbers))
        ]

        total_enrollment_within_range = filtered_data_within_range["Enrollment"].sum()

        # Display the calculated sum
        col1, col2 = st.columns([2, 1])
        col1.subheader(f"Total Enrollment in :green[{selected_programs}] from :orange[{start_year}] to :orange[{end_year}] for :red[{', '.join([month_names[num] for num in selected_month_numbers])}] :")
        col2.title(f"{total_enrollment_within_range}")

        # Create a dynamic line chart using Plotly to display enrollment over intakes
        fig = px.line(
            filtered_data,
            x="Intake",
            y="Enrollment",
            color="Program",
            title="Enrollment Over Intakes by Program",
        )
        st.plotly_chart(fig)

        # Display filtered data
        with st.expander("Click for Filtered Enrollment Data (Table)"):
            st.dataframe(filtered_data)


        # df = pd.DataFrame(student_data)
        #
        # # Get a list of columns that have string (text) data type
        # text_columns = df.select_dtypes(include=["object"]).columns
        #
        # # Widget for selecting a column to filter
        # filter_column = st.selectbox("Select a column to filter:", text_columns)
        #
        # if filter_column:
        #     filter_value = st.multiselect(f"Select {filter_column}:", df[filter_column].unique())
        #     if filter_value:
        #         df = df[df[filter_column].isin(filter_value)]
        #
        # st.write("Filtered DataFrame:")
        # st.write(df)

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
