import streamlit as st
import sqlite3
import pandas as pd
from sqlalchemy import create_engine

import streamlit as st

st.sidebar.title('University Student Enrollment Dashboard')

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
    st.sidebar.title("Insert Data Page")

    # Function to create a SQLite database connection
    def create_connection():
        conn = None
        try:
            conn = sqlite3.connect("student_enrollment.db")
        except sqlite3.Error as e:
            st.error(f"Error connecting to database: {e}")
        return conn

    # Function to create the table if it doesn't exist
    def create_table(conn):
        create_table_query = """
            CREATE TABLE IF NOT EXISTS admissions (
                id INTEGER PRIMARY KEY,
                Intake TEXT NOT NULL,
                Faculty TEXT NOT NULL,
                Program TEXT NOT NULL,
                Enrollment INTEGER NOT NULL
            )
        """
        try:
            conn.execute(create_table_query)
        except sqlite3.Error as e:
            st.error(f"Error creating table: {e}")

    # Function to insert user data into the table
    def insert_admissions_data(conn, intake, faculty, program, enrollment):
        insert_query = "INSERT INTO admissions (Intake, Faculty, Program, Enrollment) VALUES (?, ?, ?, ?)"
        try:
            conn.execute(insert_query, (intake, faculty, program, enrollment))
            conn.commit()
            st.success('Data has been successfully inserted!')
        except sqlite3.Error as e:
            st.error(f"Error inserting data: {e}")

    # Function to fetch all user data from the table
    def fetch_student_data(conn):
        select_query = "SELECT * FROM admissions"
        try:
            data = pd.read_sql_query(select_query, conn)
            return data
        except sqlite3.Error as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    # def main():
    conn = create_connection()
    create_table(conn)
    st.title("Student Enrollment Input Form")

    # st.info("Please select the first day of the month when choosing the intake", icon="ℹ️")
    # intake = st.date_input("Select an Intake: (Intakes are in Jan, May, Sep)", format="YYYY-MM-DD")
    intake = st.text_input("Enter intake:")

    faculties = ["Faculty of Engineering", "Faculty of Business", "Faculty of Computer Sciences"]
    faculty = st.selectbox("Select Faculty:", faculties)

    programs = {
        "Faculty of Engineering": ["Chemical Engineering", "Mechanical Engineering", "Electrical Engineering"],
        "Faculty of Business": ["Marketing", "Accounting", "Business Administration"],
        "Faculty of Computer Sciences": ["Computer Science", "Data Science", "Mobile Networking"]
    }
    program = st.selectbox("Select Program:", programs[faculty])

    enrollment = st.text_input("Enter the number of students enrolled:")

    if st.button("Add"):
        insert_admissions_data(conn, intake, faculty, program, enrollment)

        # Display the data from the database
    st.subheader("Student Enrollment Data")
    student_data = fetch_student_data(conn)
    st.dataframe(student_data)

    # if st.button("Delete Rows?"):
    row_id_input = st.text_input('Enter the Row ID to Delete')
    if st.button('Delete'):
        cursor = conn.cursor()
        cursor.execute(f'''DELETE FROM admissions WHERE id = {row_id_input}''')
        conn.commit()
        conn.close()


# if __name__ == "__main__":
#     main()


# def create_gender_table(conn):
#     create_table_query = """
#         CREATE TABLE IF NOT EXISTS gender (
#             id INTEGER PRIMARY KEY,
#             gender TEXT NOT NULL,
#         )
#     """
#     try:
#         conn.execute(create_table_query)
#     except sqlite3.Error as e:
#         st.error(f"Error creating gender table: {e}")
#
# def insert_gender_data(conn, name, gender, age):
#     insert_query = "INSERT INTO gender (name, gender, age) VALUES (?, ?, ?)"
#     try:
#         conn.execute(insert_query, (name, gender, age))
#         conn.commit()
#     except sqlite3.Error as e:
#         st.error(f"Error inserting gender data: {e}")

# def admissions_page():
#     conn = create_connection()
#     create_admissions_table(conn)
#     create_gender_table(conn)
#
#     st.title("Admissions Page")
#     st.header("Student Enrollment and Gender Data")
#
#     # Admissions data input section
#     st.subheader("Admissions Data")
#     year = st.text_input("Enter the year:")
#     intake = st.selectbox("Select Intake:", ["January", "May", "September"])
#     faculty = st.selectbox("Select Faculty:", ["Faculty of Engineering", "Faculty of Business", "Faculty of Computer Sciences"])
#     program = st.selectbox("Select Program:", ["Chemical Engineering", "Mechanical Engineering", "Electrical Engineering", "Marketing", "Accounting", "Business Administration", "Computer Science", "Data Science", "Mobile Networking"])
#     enrollment = st.text_input("Enter the number of students enrolled:")
#     if st.button("Add Admissions Data"):
#         insert_admissions_data(conn, year, intake, faculty, program, enrollment)
#
#     # Gender data input section
#     st.subheader("Gender Data")
#     name = st.text_input("Enter name:")
#     gender = st.selectbox("Select gender:", ["Male", "Female"])
#     age = st.text_input("Enter age:")
#     if st.button("Add Gender Data"):
#         insert_gender_data(conn, name, gender, age)
#
#     # Display the admissions data
#     st.subheader("Admissions Data")
#     admissions_data = fetch_admissions_data(conn)
#     st.dataframe(admissions_data)
#
#     # Display the gender data
#     st.subheader("Gender Data")
#     gender_data = fetch_gender_data(conn)
#     st.dataframe(gender_data)
#
# def fetch_admissions_data(conn):
#     select_query = "SELECT * FROM admissions"
#     try:
#         data = pd.read_sql_query(select_query, conn)
#         return data
#     except sqlite3.Error as e:
#         st.error(f"Error fetching admissions data: {e}")
#         return pd.DataFrame()
#
# def fetch_gender_data(conn):
#     select_query = "SELECT * FROM gender"
#     try:
#         data = pd.read_sql_query(select_query, conn)
#         return data
#     except sqlite3.Error as e:
#         st.error(f"Error fetching gender data: {e}")
#         return pd.DataFrame()
