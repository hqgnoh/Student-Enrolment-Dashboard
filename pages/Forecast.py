#%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sqlite3
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import plotly.express as px
from calendar import month_abbr
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

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
    st.sidebar.title("Forecast Page")
    with st.sidebar:
        selected = option_menu("List of Faculties", ["Faculty of Business", "Faculty of Engineering", "Faculty of Computer Sciences"],
        icons = ['graph-up','gear','laptop'], menu_icon="list", default_index=0)

    #Faculty of Business
    def page_faculty_business():
        st.title("Faculty of Business")
        selected_department_business = st.sidebar.selectbox("Select Department:", ["Marketing", "Accounting", "Business Administration"])

        # Marketing Department
        def marketing():
            st.header("Marketing Department")

            def fetch_student_data():
                conn = sqlite3.connect("student_enrollment.db")
                select_query = "SELECT Intake, Enrollment FROM admissions WHERE program = 'Marketing';"
                try:
                    data = pd.read_sql_query(select_query, conn)
                    return data
                except sqlite3.Error as e:
                    st.error(f"Error fetching data: {e}")
                    return pd.DataFrame()

            data = fetch_student_data()

            data['Intake'] = pd.to_datetime(data['Intake'])
            data.set_index('Intake', inplace=True)

            with st.expander("Click for Current Enrollment"):
                student_data = fetch_student_data()
                st.write(student_data)

            st.write("Prediction Models:")

            st.subheader('Prophet Model')
            with st.expander("Click to expand"):
                #Prophet
                prophet_data = data.resample('M').sum()
                prophet_data.reset_index(inplace=True)
                prophet_data.columns = ['ds', 'y']

                model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
                model.fit(prophet_data)
                num_years = 10
                future = model.make_future_dataframe(periods=num_years * 12, freq='M')
                forecast = model.predict(future)
                forecasted_values_prophet = forecast[forecast['ds'].dt.month.isin([1, 5, 9])]
                forecasted_values_prophet = forecasted_values_prophet[['ds', 'yhat']].rename(columns={'yhat': 'Forecasted Enrollment'})

                selected_semester = st.selectbox('Select Semester to forecast:', ['January', 'May', 'September'], key="prophet")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                st.subheader(f"Forecasted Enrollment for Month {selected_semester} Over 10 Years:")
                selected_forecast = forecasted_values_prophet[forecasted_values_prophet['ds'].dt.month == selected_month]
                selected_forecast = selected_forecast[selected_forecast['ds'].dt.year >= 2023]
                st.table(selected_forecast)

                fig = px.line(forecasted_values_prophet, x='ds', y='Forecasted Enrollment', title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('Polynomial Regression')
            with st.expander("Click to expand"):
                #Polynomial
                historical_data = data.resample('M').sum()
                historical_data['Year'] = historical_data.index.year
                historical_data['Month'] = historical_data.index.month
                degree = 2
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(historical_data[['Year', 'Month']])
                model = LinearRegression()
                model.fit(X_poly, historical_data['Enrollment'])
                num_years = 10
                forecasted_values_poly = []
                for year in range(2023, 2023 + num_years):
                    for month in [1, 5, 9]:
                        X_pred = poly.transform([[year, month]])
                        forecast = model.predict(X_pred)
                        forecasted_values_poly.append((year, month, forecast[0]))
                forecast_df = pd.DataFrame(forecasted_values_poly, columns=['Year', 'Month', 'Forecasted Enrollment'])

                st.write('Polynomial Regression Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                st.write(forecast_df)
                selected_year = st.slider('Select Year', 2023, 2032, 2023, key="Polynomial")
                selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="Polynomial1")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                selected_forecast = forecast_df[(forecast_df['Month'] == selected_month) & (forecast_df['Year'] == selected_year)]

                if not selected_forecast.empty:
                    # Display the forecasted enrollment for the selected semester and year
                    st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                    st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                else:
                    # Handle the case where there are no matching records
                    st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                fig = px.line(forecast_df, x='Year', y='Forecasted Enrollment',
                              title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('LSTM Model')
            with st.expander("Click to expand"):
                with st.spinner('Loading...'):
                    time.sleep(1)
                    tf.keras.utils.set_random_seed(42)
                    filtered_data = data[data.index.month.isin([1, 5, 9])]
                    num_years = 10
                    sequence_length = 4
                    forecasted_values_LSTM = []
                    model = Sequential()
                    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    for year in range(2023, 2023 + num_years):
                        for month in [1, 5, 9]:  # January, May, September
                            subset = filtered_data[filtered_data.index.month == month]
                            scaler = MinMaxScaler()
                            scaled_data = scaler.fit_transform(subset[['Enrollment']])
                            sequences = []
                            for i in range(len(scaled_data) - sequence_length):
                                sequences.append(scaled_data[i:i + sequence_length])
                            X = np.array(sequences)
                            y = scaled_data[sequence_length:]
                            X = X.reshape(X.shape[0], X.shape[1], 1)
                            model.fit(X, y, epochs=100, batch_size=64, verbose=0)
                            last_sequence = scaled_data[-sequence_length:]
                            forecast = model.predict(np.array([last_sequence]))
                            forecast = scaler.inverse_transform(forecast)[0][0]
                            forecasted_values_LSTM.append((year, month, forecast))

                    forecast_df_LSTM = pd.DataFrame(forecasted_values_LSTM, columns=['Year', 'Month', 'Forecasted Enrollment'])
                    forecast_df_LSTM['Date'] = pd.to_datetime(forecast_df_LSTM[['Year', 'Month']].assign(day=1))

                    st.write('LSTM Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                    st.write(forecast_df_LSTM)
                    selected_year = st.slider('Select Year', 2023, 2032, 2023, key="LSTM")
                    selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="LSTM1")
                    month_mapping = {'January': 1, 'May': 5, 'September': 9}
                    selected_month = month_mapping[selected_semester]
                    selected_forecast = forecast_df_LSTM[(forecast_df_LSTM['Month'] == selected_month) & (forecast_df_LSTM['Year'] == selected_year)]

                    if not selected_forecast.empty:
                        # Display the forecasted enrollment for the selected semester and year
                        st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                        st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                    else:
                        # Handle the case where there are no matching records
                        st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                    month_names = [month_abbr[month] for month in forecast_df_LSTM['Month']]
                    fig = px.line(forecast_df_LSTM, x='Year', y='Forecasted Enrollment', color=month_names,
                                  title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                    fig.update_layout(legend_title_text='Intake')
                    st.plotly_chart(fig)

        # Accounting Department
        def accounting():
            st.header("Accounting Department")

            def fetch_student_data():
                conn = sqlite3.connect("student_enrollment.db")
                select_query = "SELECT Intake, Enrollment FROM admissions WHERE program = 'Accounting';"
                try:
                    data = pd.read_sql_query(select_query, conn)
                    return data
                except sqlite3.Error as e:
                    st.error(f"Error fetching data: {e}")
                    return pd.DataFrame()

            data = fetch_student_data()

            data['Intake'] = pd.to_datetime(data['Intake'])
            data.set_index('Intake', inplace=True)

            with st.expander("Click for Current Enrollment"):
                student_data = fetch_student_data()
                st.write(student_data)

            st.write("Prediction Models:")

            st.subheader('Prophet Model')
            with st.expander("Click to expand"):
                #Prophet
                prophet_data = data.resample('M').sum()
                prophet_data.reset_index(inplace=True)
                prophet_data.columns = ['ds', 'y']

                model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
                model.fit(prophet_data)
                num_years = 10
                future = model.make_future_dataframe(periods=num_years * 12, freq='M')
                forecast = model.predict(future)
                forecasted_values_prophet = forecast[forecast['ds'].dt.month.isin([1, 5, 9])]
                forecasted_values_prophet = forecasted_values_prophet[['ds', 'yhat']].rename(columns={'yhat': 'Forecasted Enrollment'})

                selected_semester = st.selectbox('Select Semester to forecast:', ['January', 'May', 'September'], key="prophet")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                st.subheader(f"Forecasted Enrollment for Month {selected_semester} Over 10 Years:")
                selected_forecast = forecasted_values_prophet[forecasted_values_prophet['ds'].dt.month == selected_month]
                selected_forecast = selected_forecast[selected_forecast['ds'].dt.year >= 2023]
                st.table(selected_forecast)

                fig = px.line(forecasted_values_prophet, x='ds', y='Forecasted Enrollment', title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('Polynomial Regression')
            with st.expander("Click to expand"):
                #Polynomial
                historical_data = data.resample('M').sum()
                historical_data['Year'] = historical_data.index.year
                historical_data['Month'] = historical_data.index.month
                degree = 2
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(historical_data[['Year', 'Month']])
                model = LinearRegression()
                model.fit(X_poly, historical_data['Enrollment'])
                num_years = 10
                forecasted_values_poly = []
                for year in range(2023, 2023 + num_years):
                    for month in [1, 5, 9]:
                        X_pred = poly.transform([[year, month]])
                        forecast = model.predict(X_pred)
                        forecasted_values_poly.append((year, month, forecast[0]))
                forecast_df = pd.DataFrame(forecasted_values_poly, columns=['Year', 'Month', 'Forecasted Enrollment'])

                st.write('Polynomial Regression Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                st.write(forecast_df)
                selected_year = st.slider('Select Year', 2023, 2032, 2023, key="Polynomial")
                selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="Polynomial1")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                selected_forecast = forecast_df[(forecast_df['Month'] == selected_month) & (forecast_df['Year'] == selected_year)]

                if not selected_forecast.empty:
                    # Display the forecasted enrollment for the selected semester and year
                    st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                    st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                else:
                    # Handle the case where there are no matching records
                    st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                fig = px.line(forecast_df, x='Year', y='Forecasted Enrollment',
                              title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('LSTM Model')
            with st.expander("Click to expand"):
                with st.spinner('Loading...'):
                    time.sleep(1)
                    tf.keras.utils.set_random_seed(42)
                    filtered_data = data[data.index.month.isin([1, 5, 9])]
                    num_years = 10
                    sequence_length = 4
                    forecasted_values_LSTM = []
                    model = Sequential()
                    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    for year in range(2023, 2023 + num_years):
                        for month in [1, 5, 9]:  # January, May, September
                            subset = filtered_data[filtered_data.index.month == month]
                            scaler = MinMaxScaler()
                            scaled_data = scaler.fit_transform(subset[['Enrollment']])
                            sequences = []
                            for i in range(len(scaled_data) - sequence_length):
                                sequences.append(scaled_data[i:i + sequence_length])
                            X = np.array(sequences)
                            y = scaled_data[sequence_length:]
                            X = X.reshape(X.shape[0], X.shape[1], 1)
                            model.fit(X, y, epochs=100, batch_size=64, verbose=0)
                            last_sequence = scaled_data[-sequence_length:]
                            forecast = model.predict(np.array([last_sequence]))
                            forecast = scaler.inverse_transform(forecast)[0][0]
                            forecasted_values_LSTM.append((year, month, forecast))

                    forecast_df_LSTM = pd.DataFrame(forecasted_values_LSTM, columns=['Year', 'Month', 'Forecasted Enrollment'])
                    forecast_df_LSTM['Date'] = pd.to_datetime(forecast_df_LSTM[['Year', 'Month']].assign(day=1))

                    st.write('LSTM Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                    st.write(forecast_df_LSTM)
                    selected_year = st.slider('Select Year', 2023, 2032, 2023, key="LSTM")
                    selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="LSTM1")
                    month_mapping = {'January': 1, 'May': 5, 'September': 9}
                    selected_month = month_mapping[selected_semester]
                    selected_forecast = forecast_df_LSTM[(forecast_df_LSTM['Month'] == selected_month) & (forecast_df_LSTM['Year'] == selected_year)]

                    if not selected_forecast.empty:
                        # Display the forecasted enrollment for the selected semester and year
                        st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                        st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                    else:
                        # Handle the case where there are no matching records
                        st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                    month_names = [month_abbr[month] for month in forecast_df_LSTM['Month']]
                    fig = px.line(forecast_df_LSTM, x='Year', y='Forecasted Enrollment', color=month_names,
                                  title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                    fig.update_layout(legend_title_text='Intake')
                    st.plotly_chart(fig)

        # Business Administration Department
        def business_admin():
            st.header("Business Administration Department")

            def fetch_student_data():
                conn = sqlite3.connect("student_enrollment.db")
                select_query = "SELECT Intake, Enrollment FROM admissions WHERE program = 'Business Administration';"
                try:
                    data = pd.read_sql_query(select_query, conn)
                    return data
                except sqlite3.Error as e:
                    st.error(f"Error fetching data: {e}")
                    return pd.DataFrame()

            data = fetch_student_data()

            data['Intake'] = pd.to_datetime(data['Intake'])
            data.set_index('Intake', inplace=True)

            with st.expander("Click for Current Enrollment"):
                student_data = fetch_student_data()
                st.write(student_data)

            st.write("Prediction Models:")

            st.subheader('Prophet Model')
            with st.expander("Click to expand"):
                #Prophet
                prophet_data = data.resample('M').sum()
                prophet_data.reset_index(inplace=True)
                prophet_data.columns = ['ds', 'y']

                model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
                model.fit(prophet_data)
                num_years = 10
                future = model.make_future_dataframe(periods=num_years * 12, freq='M')
                forecast = model.predict(future)
                forecasted_values_prophet = forecast[forecast['ds'].dt.month.isin([1, 5, 9])]
                forecasted_values_prophet = forecasted_values_prophet[['ds', 'yhat']].rename(columns={'yhat': 'Forecasted Enrollment'})

                selected_semester = st.selectbox('Select Semester to forecast:', ['January', 'May', 'September'], key="prophet")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                st.subheader(f"Forecasted Enrollment for Month {selected_semester} Over 10 Years:")
                selected_forecast = forecasted_values_prophet[forecasted_values_prophet['ds'].dt.month == selected_month]
                selected_forecast = selected_forecast[selected_forecast['ds'].dt.year >= 2023]
                st.table(selected_forecast)

                fig = px.line(forecasted_values_prophet, x='ds', y='Forecasted Enrollment', title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('Polynomial Regression')
            with st.expander("Click to expand"):
                #Polynomial
                historical_data = data.resample('M').sum()
                historical_data['Year'] = historical_data.index.year
                historical_data['Month'] = historical_data.index.month
                degree = 2
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(historical_data[['Year', 'Month']])
                model = LinearRegression()
                model.fit(X_poly, historical_data['Enrollment'])
                num_years = 10
                forecasted_values_poly = []
                for year in range(2023, 2023 + num_years):
                    for month in [1, 5, 9]:
                        X_pred = poly.transform([[year, month]])
                        forecast = model.predict(X_pred)
                        forecasted_values_poly.append((year, month, forecast[0]))
                forecast_df = pd.DataFrame(forecasted_values_poly, columns=['Year', 'Month', 'Forecasted Enrollment'])

                st.write('Polynomial Regression Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                st.write(forecast_df)
                selected_year = st.slider('Select Year', 2023, 2032, 2023, key="Polynomial")
                selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="Polynomial1")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                selected_forecast = forecast_df[(forecast_df['Month'] == selected_month) & (forecast_df['Year'] == selected_year)]

                if not selected_forecast.empty:
                    # Display the forecasted enrollment for the selected semester and year
                    st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                    st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                else:
                    # Handle the case where there are no matching records
                    st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                fig = px.line(forecast_df, x='Year', y='Forecasted Enrollment',
                              title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('LSTM Model')
            with st.expander("Click to expand"):
                with st.spinner('Loading...'):
                    time.sleep(1)
                    tf.keras.utils.set_random_seed(42)
                    filtered_data = data[data.index.month.isin([1, 5, 9])]
                    num_years = 10
                    sequence_length = 4
                    forecasted_values_LSTM = []
                    model = Sequential()
                    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    for year in range(2023, 2023 + num_years):
                        for month in [1, 5, 9]:  # January, May, September
                            subset = filtered_data[filtered_data.index.month == month]
                            scaler = MinMaxScaler()
                            scaled_data = scaler.fit_transform(subset[['Enrollment']])
                            sequences = []
                            for i in range(len(scaled_data) - sequence_length):
                                sequences.append(scaled_data[i:i + sequence_length])
                            X = np.array(sequences)
                            y = scaled_data[sequence_length:]
                            X = X.reshape(X.shape[0], X.shape[1], 1)
                            model.fit(X, y, epochs=100, batch_size=64, verbose=0)
                            last_sequence = scaled_data[-sequence_length:]
                            forecast = model.predict(np.array([last_sequence]))
                            forecast = scaler.inverse_transform(forecast)[0][0]
                            forecasted_values_LSTM.append((year, month, forecast))

                    forecast_df_LSTM = pd.DataFrame(forecasted_values_LSTM, columns=['Year', 'Month', 'Forecasted Enrollment'])
                    forecast_df_LSTM['Date'] = pd.to_datetime(forecast_df_LSTM[['Year', 'Month']].assign(day=1))

                    st.write('LSTM Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                    st.write(forecast_df_LSTM)
                    selected_year = st.slider('Select Year', 2023, 2032, 2023, key="LSTM")
                    selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="LSTM1")
                    month_mapping = {'January': 1, 'May': 5, 'September': 9}
                    selected_month = month_mapping[selected_semester]
                    selected_forecast = forecast_df_LSTM[(forecast_df_LSTM['Month'] == selected_month) & (forecast_df_LSTM['Year'] == selected_year)]

                    if not selected_forecast.empty:
                        # Display the forecasted enrollment for the selected semester and year
                        st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                        st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                    else:
                        # Handle the case where there are no matching records
                        st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                    month_names = [month_abbr[month] for month in forecast_df_LSTM['Month']]
                    fig = px.line(forecast_df_LSTM, x='Year', y='Forecasted Enrollment', color=month_names,
                                  title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                    fig.update_layout(legend_title_text='Intake')
                    st.plotly_chart(fig)

        if selected_department_business == "Marketing":
            marketing()

        elif selected_department_business == "Accounting":
            accounting()

        elif selected_department_business == "Business Administration":
            business_admin()

    #Faculty of Engineering
    def page_faculty_engineering():
        st.title("Faculty of Engineering")
        selected_department_business = st.sidebar.selectbox("Select Department:", ["Chemical", "Mechanical", "Electrical"])

        # Chemical Engineering Department
        def chemical():
            st.header("Chemical Engineering Department")

            def fetch_student_data():
                conn = sqlite3.connect("student_enrollment.db")
                select_query = "SELECT Intake, Enrollment FROM admissions WHERE program = 'Chemical Engineering';"
                try:
                    data = pd.read_sql_query(select_query, conn)
                    return data
                except sqlite3.Error as e:
                    st.error(f"Error fetching data: {e}")
                    return pd.DataFrame()

            data = fetch_student_data()

            data['Intake'] = pd.to_datetime(data['Intake'])
            data.set_index('Intake', inplace=True)

            with st.expander("Click for Current Enrollment"):
                student_data = fetch_student_data()
                st.write(student_data)

            st.write("Prediction Models:")

            st.subheader('Prophet Model')
            with st.expander("Click to expand"):
                #Prophet
                prophet_data = data.resample('M').sum()
                prophet_data.reset_index(inplace=True)
                prophet_data.columns = ['ds', 'y']

                model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
                model.fit(prophet_data)
                num_years = 10
                future = model.make_future_dataframe(periods=num_years * 12, freq='M')
                forecast = model.predict(future)
                forecasted_values_prophet = forecast[forecast['ds'].dt.month.isin([1, 5, 9])]
                forecasted_values_prophet = forecasted_values_prophet[['ds', 'yhat']].rename(columns={'yhat': 'Forecasted Enrollment'})

                selected_semester = st.selectbox('Select Semester to forecast:', ['January', 'May', 'September'], key="prophet")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                st.subheader(f"Forecasted Enrollment for Month {selected_semester} Over 10 Years:")
                selected_forecast = forecasted_values_prophet[forecasted_values_prophet['ds'].dt.month == selected_month]
                selected_forecast = selected_forecast[selected_forecast['ds'].dt.year >= 2023]
                st.table(selected_forecast)

                fig = px.line(forecasted_values_prophet, x='ds', y='Forecasted Enrollment', title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('Polynomial Regression')
            with st.expander("Click to expand"):
                #Polynomial
                historical_data = data.resample('M').sum()
                historical_data['Year'] = historical_data.index.year
                historical_data['Month'] = historical_data.index.month
                degree = 2
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(historical_data[['Year', 'Month']])
                model = LinearRegression()
                model.fit(X_poly, historical_data['Enrollment'])
                num_years = 10
                forecasted_values_poly = []
                for year in range(2023, 2023 + num_years):
                    for month in [1, 5, 9]:
                        X_pred = poly.transform([[year, month]])
                        forecast = model.predict(X_pred)
                        forecasted_values_poly.append((year, month, forecast[0]))
                forecast_df = pd.DataFrame(forecasted_values_poly, columns=['Year', 'Month', 'Forecasted Enrollment'])

                st.write('Polynomial Regression Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                st.write(forecast_df)
                selected_year = st.slider('Select Year', 2023, 2032, 2023, key="Polynomial")
                selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="Polynomial1")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                selected_forecast = forecast_df[(forecast_df['Month'] == selected_month) & (forecast_df['Year'] == selected_year)]

                if not selected_forecast.empty:
                    # Display the forecasted enrollment for the selected semester and year
                    st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                    st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                else:
                    # Handle the case where there are no matching records
                    st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                fig = px.line(forecast_df, x='Year', y='Forecasted Enrollment',
                              title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('LSTM Model')
            with st.expander("Click to expand"):
                with st.spinner('Loading...'):
                    time.sleep(1)
                    tf.keras.utils.set_random_seed(42)
                    filtered_data = data[data.index.month.isin([1, 5, 9])]
                    num_years = 10
                    sequence_length = 4
                    forecasted_values_LSTM = []
                    model = Sequential()
                    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    for year in range(2023, 2023 + num_years):
                        for month in [1, 5, 9]:  # January, May, September
                            subset = filtered_data[filtered_data.index.month == month]
                            scaler = MinMaxScaler()
                            scaled_data = scaler.fit_transform(subset[['Enrollment']])
                            sequences = []
                            for i in range(len(scaled_data) - sequence_length):
                                sequences.append(scaled_data[i:i + sequence_length])
                            X = np.array(sequences)
                            y = scaled_data[sequence_length:]
                            X = X.reshape(X.shape[0], X.shape[1], 1)
                            model.fit(X, y, epochs=100, batch_size=64, verbose=0)
                            last_sequence = scaled_data[-sequence_length:]
                            forecast = model.predict(np.array([last_sequence]))
                            forecast = scaler.inverse_transform(forecast)[0][0]
                            forecasted_values_LSTM.append((year, month, forecast))

                    forecast_df_LSTM = pd.DataFrame(forecasted_values_LSTM, columns=['Year', 'Month', 'Forecasted Enrollment'])
                    forecast_df_LSTM['Date'] = pd.to_datetime(forecast_df_LSTM[['Year', 'Month']].assign(day=1))

                    st.write('LSTM Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                    st.write(forecast_df_LSTM)
                    selected_year = st.slider('Select Year', 2023, 2032, 2023, key="LSTM")
                    selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="LSTM1")
                    month_mapping = {'January': 1, 'May': 5, 'September': 9}
                    selected_month = month_mapping[selected_semester]
                    selected_forecast = forecast_df_LSTM[(forecast_df_LSTM['Month'] == selected_month) & (forecast_df_LSTM['Year'] == selected_year)]

                    if not selected_forecast.empty:
                        # Display the forecasted enrollment for the selected semester and year
                        st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                        st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                    else:
                        # Handle the case where there are no matching records
                        st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                    month_names = [month_abbr[month] for month in forecast_df_LSTM['Month']]
                    fig = px.line(forecast_df_LSTM, x='Year', y='Forecasted Enrollment', color=month_names,
                                  title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                    fig.update_layout(legend_title_text='Intake')
                    st.plotly_chart(fig)

        # Mechanical Engineering Department
        def mechanical():
            st.header("Mechanical Engineering Department")

            def fetch_student_data():
                conn = sqlite3.connect("student_enrollment.db")
                select_query = "SELECT Intake, Enrollment FROM admissions WHERE program = 'Mechanical Engineering';"
                try:
                    data = pd.read_sql_query(select_query, conn)
                    return data
                except sqlite3.Error as e:
                    st.error(f"Error fetching data: {e}")
                    return pd.DataFrame()

            data = fetch_student_data()

            data['Intake'] = pd.to_datetime(data['Intake'])
            data.set_index('Intake', inplace=True)

            with st.expander("Click for Current Enrollment"):
                student_data = fetch_student_data()
                st.write(student_data)

            st.write("Prediction Models:")

            st.subheader('Prophet Model')
            with st.expander("Click to expand"):
                #Prophet
                prophet_data = data.resample('M').sum()
                prophet_data.reset_index(inplace=True)
                prophet_data.columns = ['ds', 'y']

                model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
                model.fit(prophet_data)
                num_years = 10
                future = model.make_future_dataframe(periods=num_years * 12, freq='M')
                forecast = model.predict(future)
                forecasted_values_prophet = forecast[forecast['ds'].dt.month.isin([1, 5, 9])]
                forecasted_values_prophet = forecasted_values_prophet[['ds', 'yhat']].rename(columns={'yhat': 'Forecasted Enrollment'})

                selected_semester = st.selectbox('Select Semester to forecast:', ['January', 'May', 'September'], key="prophet")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                st.subheader(f"Forecasted Enrollment for Month {selected_semester} Over 10 Years:")
                selected_forecast = forecasted_values_prophet[forecasted_values_prophet['ds'].dt.month == selected_month]
                selected_forecast = selected_forecast[selected_forecast['ds'].dt.year >= 2023]
                st.table(selected_forecast)

                fig = px.line(forecasted_values_prophet, x='ds', y='Forecasted Enrollment', title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('Polynomial Regression')
            with st.expander("Click to expand"):
                #Polynomial
                historical_data = data.resample('M').sum()
                historical_data['Year'] = historical_data.index.year
                historical_data['Month'] = historical_data.index.month
                degree = 2
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(historical_data[['Year', 'Month']])
                model = LinearRegression()
                model.fit(X_poly, historical_data['Enrollment'])
                num_years = 10
                forecasted_values_poly = []
                for year in range(2023, 2023 + num_years):
                    for month in [1, 5, 9]:
                        X_pred = poly.transform([[year, month]])
                        forecast = model.predict(X_pred)
                        forecasted_values_poly.append((year, month, forecast[0]))
                forecast_df = pd.DataFrame(forecasted_values_poly, columns=['Year', 'Month', 'Forecasted Enrollment'])

                st.write('Polynomial Regression Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                st.write(forecast_df)
                selected_year = st.slider('Select Year', 2023, 2032, 2023, key="Polynomial")
                selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="Polynomial1")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                selected_forecast = forecast_df[(forecast_df['Month'] == selected_month) & (forecast_df['Year'] == selected_year)]

                if not selected_forecast.empty:
                    # Display the forecasted enrollment for the selected semester and year
                    st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                    st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                else:
                    # Handle the case where there are no matching records
                    st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                fig = px.line(forecast_df, x='Year', y='Forecasted Enrollment',
                              title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('LSTM Model')
            with st.expander("Click to expand"):
                with st.spinner('Loading...'):
                    time.sleep(1)
                    tf.keras.utils.set_random_seed(42)
                    filtered_data = data[data.index.month.isin([1, 5, 9])]
                    num_years = 10
                    sequence_length = 4
                    forecasted_values_LSTM = []
                    model = Sequential()
                    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    for year in range(2023, 2023 + num_years):
                        for month in [1, 5, 9]:  # January, May, September
                            subset = filtered_data[filtered_data.index.month == month]
                            scaler = MinMaxScaler()
                            scaled_data = scaler.fit_transform(subset[['Enrollment']])
                            sequences = []
                            for i in range(len(scaled_data) - sequence_length):
                                sequences.append(scaled_data[i:i + sequence_length])
                            X = np.array(sequences)
                            y = scaled_data[sequence_length:]
                            X = X.reshape(X.shape[0], X.shape[1], 1)
                            model.fit(X, y, epochs=100, batch_size=64, verbose=0)
                            last_sequence = scaled_data[-sequence_length:]
                            forecast = model.predict(np.array([last_sequence]))
                            forecast = scaler.inverse_transform(forecast)[0][0]
                            forecasted_values_LSTM.append((year, month, forecast))

                    forecast_df_LSTM = pd.DataFrame(forecasted_values_LSTM, columns=['Year', 'Month', 'Forecasted Enrollment'])
                    forecast_df_LSTM['Date'] = pd.to_datetime(forecast_df_LSTM[['Year', 'Month']].assign(day=1))

                    st.write('LSTM Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                    st.write(forecast_df_LSTM)
                    selected_year = st.slider('Select Year', 2023, 2032, 2023, key="LSTM")
                    selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="LSTM1")
                    month_mapping = {'January': 1, 'May': 5, 'September': 9}
                    selected_month = month_mapping[selected_semester]
                    selected_forecast = forecast_df_LSTM[(forecast_df_LSTM['Month'] == selected_month) & (forecast_df_LSTM['Year'] == selected_year)]

                    if not selected_forecast.empty:
                        # Display the forecasted enrollment for the selected semester and year
                        st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                        st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                    else:
                        # Handle the case where there are no matching records
                        st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                    month_names = [month_abbr[month] for month in forecast_df_LSTM['Month']]
                    fig = px.line(forecast_df_LSTM, x='Year', y='Forecasted Enrollment', color=month_names,
                                  title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                    fig.update_layout(legend_title_text='Intake')
                    st.plotly_chart(fig)

        # Electrical Engineering Department
        def electrical():
            st.header("Electrical Engineering Department")

            def fetch_student_data():
                conn = sqlite3.connect("student_enrollment.db")
                select_query = "SELECT Intake, Enrollment FROM admissions WHERE program = 'Electrical Engineering';"
                try:
                    data = pd.read_sql_query(select_query, conn)
                    return data
                except sqlite3.Error as e:
                    st.error(f"Error fetching data: {e}")
                    return pd.DataFrame()

            data = fetch_student_data()

            data['Intake'] = pd.to_datetime(data['Intake'])
            data.set_index('Intake', inplace=True)

            with st.expander("Click for Current Enrollment"):
                student_data = fetch_student_data()
                st.write(student_data)

            st.write("Prediction Models:")

            st.subheader('Prophet Model')
            with st.expander("Click to expand"):
                #Prophet
                prophet_data = data.resample('M').sum()
                prophet_data.reset_index(inplace=True)
                prophet_data.columns = ['ds', 'y']

                model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
                model.fit(prophet_data)
                num_years = 10
                future = model.make_future_dataframe(periods=num_years * 12, freq='M')
                forecast = model.predict(future)
                forecasted_values_prophet = forecast[forecast['ds'].dt.month.isin([1, 5, 9])]
                forecasted_values_prophet = forecasted_values_prophet[['ds', 'yhat']].rename(columns={'yhat': 'Forecasted Enrollment'})

                selected_semester = st.selectbox('Select Semester to forecast:', ['January', 'May', 'September'], key="prophet")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                st.subheader(f"Forecasted Enrollment for Month {selected_semester} Over 10 Years:")
                selected_forecast = forecasted_values_prophet[forecasted_values_prophet['ds'].dt.month == selected_month]
                selected_forecast = selected_forecast[selected_forecast['ds'].dt.year >= 2023]
                st.table(selected_forecast)

                fig = px.line(forecasted_values_prophet, x='ds', y='Forecasted Enrollment', title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('Polynomial Regression')
            with st.expander("Click to expand"):
                #Polynomial
                historical_data = data.resample('M').sum()
                historical_data['Year'] = historical_data.index.year
                historical_data['Month'] = historical_data.index.month
                degree = 2
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(historical_data[['Year', 'Month']])
                model = LinearRegression()
                model.fit(X_poly, historical_data['Enrollment'])
                num_years = 10
                forecasted_values_poly = []
                for year in range(2023, 2023 + num_years):
                    for month in [1, 5, 9]:
                        X_pred = poly.transform([[year, month]])
                        forecast = model.predict(X_pred)
                        forecasted_values_poly.append((year, month, forecast[0]))
                forecast_df = pd.DataFrame(forecasted_values_poly, columns=['Year', 'Month', 'Forecasted Enrollment'])

                st.write('Polynomial Regression Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                st.write(forecast_df)
                selected_year = st.slider('Select Year', 2023, 2032, 2023, key="Polynomial")
                selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="Polynomial1")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                selected_forecast = forecast_df[(forecast_df['Month'] == selected_month) & (forecast_df['Year'] == selected_year)]

                if not selected_forecast.empty:
                    # Display the forecasted enrollment for the selected semester and year
                    st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                    st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                else:
                    # Handle the case where there are no matching records
                    st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                fig = px.line(forecast_df, x='Year', y='Forecasted Enrollment',
                              title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('LSTM Model')
            with st.expander("Click to expand"):
                with st.spinner('Loading...'):
                    time.sleep(1)
                    tf.keras.utils.set_random_seed(42)
                    filtered_data = data[data.index.month.isin([1, 5, 9])]
                    num_years = 10
                    sequence_length = 4
                    forecasted_values_LSTM = []
                    model = Sequential()
                    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    for year in range(2023, 2023 + num_years):
                        for month in [1, 5, 9]:  # January, May, September
                            subset = filtered_data[filtered_data.index.month == month]
                            scaler = MinMaxScaler()
                            scaled_data = scaler.fit_transform(subset[['Enrollment']])
                            sequences = []
                            for i in range(len(scaled_data) - sequence_length):
                                sequences.append(scaled_data[i:i + sequence_length])
                            X = np.array(sequences)
                            y = scaled_data[sequence_length:]
                            X = X.reshape(X.shape[0], X.shape[1], 1)
                            model.fit(X, y, epochs=100, batch_size=64, verbose=0)
                            last_sequence = scaled_data[-sequence_length:]
                            forecast = model.predict(np.array([last_sequence]))
                            forecast = scaler.inverse_transform(forecast)[0][0]
                            forecasted_values_LSTM.append((year, month, forecast))

                    forecast_df_LSTM = pd.DataFrame(forecasted_values_LSTM, columns=['Year', 'Month', 'Forecasted Enrollment'])
                    forecast_df_LSTM['Date'] = pd.to_datetime(forecast_df_LSTM[['Year', 'Month']].assign(day=1))

                    st.write('LSTM Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                    st.write(forecast_df_LSTM)
                    selected_year = st.slider('Select Year', 2023, 2032, 2023, key="LSTM")
                    selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="LSTM1")
                    month_mapping = {'January': 1, 'May': 5, 'September': 9}
                    selected_month = month_mapping[selected_semester]
                    selected_forecast = forecast_df_LSTM[(forecast_df_LSTM['Month'] == selected_month) & (forecast_df_LSTM['Year'] == selected_year)]

                    if not selected_forecast.empty:
                        # Display the forecasted enrollment for the selected semester and year
                        st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                        st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                    else:
                        # Handle the case where there are no matching records
                        st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                    month_names = [month_abbr[month] for month in forecast_df_LSTM['Month']]
                    fig = px.line(forecast_df_LSTM, x='Year', y='Forecasted Enrollment', color=month_names,
                                  title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                    fig.update_layout(legend_title_text='Intake')
                    st.plotly_chart(fig)

        if selected_department_business == "Chemical":
            chemical()

        elif selected_department_business == "Mechanical":
            mechanical()

        elif selected_department_business == "Electrical":
            electrical()

    #Faculty of Computer Sciences
    def page_faculty_computer():
        st.title("Faculty of Computer Sciences")
        selected_department_business = st.sidebar.selectbox("Select Department:", ["Computer Science", "Data Science", "Mobile Networking"])

        # Computer Science Department
        def computer_science():
            st.header("Computer Science Department")

            def fetch_student_data():
                conn = sqlite3.connect("student_enrollment.db")
                select_query = "SELECT Intake, Enrollment FROM admissions WHERE program = 'Computer Science';"
                try:
                    data = pd.read_sql_query(select_query, conn)
                    return data
                except sqlite3.Error as e:
                    st.error(f"Error fetching data: {e}")
                    return pd.DataFrame()

            data = fetch_student_data()

            data['Intake'] = pd.to_datetime(data['Intake'])
            data.set_index('Intake', inplace=True)

            with st.expander("Click for Current Enrollment"):
                student_data = fetch_student_data()
                st.write(student_data)

            st.write("Prediction Models:")

            st.subheader('Prophet Model')
            with st.expander("Click to expand"):
                #Prophet
                prophet_data = data.resample('M').sum()
                prophet_data.reset_index(inplace=True)
                prophet_data.columns = ['ds', 'y']

                model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
                model.fit(prophet_data)
                num_years = 10
                future = model.make_future_dataframe(periods=num_years * 12, freq='M')
                forecast = model.predict(future)
                forecasted_values_prophet = forecast[forecast['ds'].dt.month.isin([1, 5, 9])]
                forecasted_values_prophet = forecasted_values_prophet[['ds', 'yhat']].rename(columns={'yhat': 'Forecasted Enrollment'})

                selected_semester = st.selectbox('Select Semester to forecast:', ['January', 'May', 'September'], key="prophet")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                st.subheader(f"Forecasted Enrollment for Month {selected_semester} Over 10 Years:")
                selected_forecast = forecasted_values_prophet[forecasted_values_prophet['ds'].dt.month == selected_month]
                selected_forecast = selected_forecast[selected_forecast['ds'].dt.year >= 2023]
                st.table(selected_forecast)

                fig = px.line(forecasted_values_prophet, x='ds', y='Forecasted Enrollment', title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('Polynomial Regression')
            with st.expander("Click to expand"):
                #Polynomial
                historical_data = data.resample('M').sum()
                historical_data['Year'] = historical_data.index.year
                historical_data['Month'] = historical_data.index.month
                degree = 2
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(historical_data[['Year', 'Month']])
                model = LinearRegression()
                model.fit(X_poly, historical_data['Enrollment'])
                num_years = 10
                forecasted_values_poly = []
                for year in range(2023, 2023 + num_years):
                    for month in [1, 5, 9]:
                        X_pred = poly.transform([[year, month]])
                        forecast = model.predict(X_pred)
                        forecasted_values_poly.append((year, month, forecast[0]))
                forecast_df = pd.DataFrame(forecasted_values_poly, columns=['Year', 'Month', 'Forecasted Enrollment'])

                st.write('Polynomial Regression Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                st.write(forecast_df)
                selected_year = st.slider('Select Year', 2023, 2032, 2023, key="Polynomial")
                selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="Polynomial1")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                selected_forecast = forecast_df[(forecast_df['Month'] == selected_month) & (forecast_df['Year'] == selected_year)]

                if not selected_forecast.empty:
                    # Display the forecasted enrollment for the selected semester and year
                    st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                    st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                else:
                    # Handle the case where there are no matching records
                    st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                fig = px.line(forecast_df, x='Year', y='Forecasted Enrollment',
                              title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('LSTM Model')
            with st.expander("Click to expand"):
                with st.spinner('Loading...'):
                    time.sleep(1)
                    tf.keras.utils.set_random_seed(42)
                    filtered_data = data[data.index.month.isin([1, 5, 9])]
                    num_years = 10
                    sequence_length = 4
                    forecasted_values_LSTM = []
                    model = Sequential()
                    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    for year in range(2023, 2023 + num_years):
                        for month in [1, 5, 9]:  # January, May, September
                            subset = filtered_data[filtered_data.index.month == month]
                            scaler = MinMaxScaler()
                            scaled_data = scaler.fit_transform(subset[['Enrollment']])
                            sequences = []
                            for i in range(len(scaled_data) - sequence_length):
                                sequences.append(scaled_data[i:i + sequence_length])
                            X = np.array(sequences)
                            y = scaled_data[sequence_length:]
                            X = X.reshape(X.shape[0], X.shape[1], 1)
                            model.fit(X, y, epochs=100, batch_size=64, verbose=0)
                            last_sequence = scaled_data[-sequence_length:]
                            forecast = model.predict(np.array([last_sequence]))
                            forecast = scaler.inverse_transform(forecast)[0][0]
                            forecasted_values_LSTM.append((year, month, forecast))

                    forecast_df_LSTM = pd.DataFrame(forecasted_values_LSTM, columns=['Year', 'Month', 'Forecasted Enrollment'])
                    forecast_df_LSTM['Date'] = pd.to_datetime(forecast_df_LSTM[['Year', 'Month']].assign(day=1))

                    st.write('LSTM Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                    st.write(forecast_df_LSTM)
                    selected_year = st.slider('Select Year', 2023, 2032, 2023, key="LSTM")
                    selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="LSTM1")
                    month_mapping = {'January': 1, 'May': 5, 'September': 9}
                    selected_month = month_mapping[selected_semester]
                    selected_forecast = forecast_df_LSTM[(forecast_df_LSTM['Month'] == selected_month) & (forecast_df_LSTM['Year'] == selected_year)]

                    if not selected_forecast.empty:
                        # Display the forecasted enrollment for the selected semester and year
                        st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                        st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                    else:
                        # Handle the case where there are no matching records
                        st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                    month_names = [month_abbr[month] for month in forecast_df_LSTM['Month']]
                    fig = px.line(forecast_df_LSTM, x='Year', y='Forecasted Enrollment', color=month_names,
                                  title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                    fig.update_layout(legend_title_text='Intake')
                    st.plotly_chart(fig)

        # Data Science Department
        def data_science():
            st.header("Data Science Department")

            def fetch_student_data():
                conn = sqlite3.connect("student_enrollment.db")
                select_query = "SELECT Intake, Enrollment FROM admissions WHERE program = 'Data Science';"
                try:
                    data = pd.read_sql_query(select_query, conn)
                    return data
                except sqlite3.Error as e:
                    st.error(f"Error fetching data: {e}")
                    return pd.DataFrame()

            data = fetch_student_data()

            data['Intake'] = pd.to_datetime(data['Intake'])
            data.set_index('Intake', inplace=True)

            with st.expander("Click for Current Enrollment"):
                student_data = fetch_student_data()
                st.write(student_data)

            st.write("Prediction Models:")

            st.subheader('Prophet Model')
            with st.expander("Click to expand"):
                #Prophet
                prophet_data = data.resample('M').sum()
                prophet_data.reset_index(inplace=True)
                prophet_data.columns = ['ds', 'y']

                model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
                model.fit(prophet_data)
                num_years = 10
                future = model.make_future_dataframe(periods=num_years * 12, freq='M')
                forecast = model.predict(future)
                forecasted_values_prophet = forecast[forecast['ds'].dt.month.isin([1, 5, 9])]
                forecasted_values_prophet = forecasted_values_prophet[['ds', 'yhat']].rename(columns={'yhat': 'Forecasted Enrollment'})

                selected_semester = st.selectbox('Select Semester to forecast:', ['January', 'May', 'September'], key="prophet")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                st.subheader(f"Forecasted Enrollment for Month {selected_semester} Over 10 Years:")
                selected_forecast = forecasted_values_prophet[forecasted_values_prophet['ds'].dt.month == selected_month]
                selected_forecast = selected_forecast[selected_forecast['ds'].dt.year >= 2023]
                st.table(selected_forecast)

                fig = px.line(forecasted_values_prophet, x='ds', y='Forecasted Enrollment', title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('Polynomial Regression')
            with st.expander("Click to expand"):
                #Polynomial
                historical_data = data.resample('M').sum()
                historical_data['Year'] = historical_data.index.year
                historical_data['Month'] = historical_data.index.month
                degree = 2
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(historical_data[['Year', 'Month']])
                model = LinearRegression()
                model.fit(X_poly, historical_data['Enrollment'])
                num_years = 10
                forecasted_values_poly = []
                for year in range(2023, 2023 + num_years):
                    for month in [1, 5, 9]:
                        X_pred = poly.transform([[year, month]])
                        forecast = model.predict(X_pred)
                        forecasted_values_poly.append((year, month, forecast[0]))
                forecast_df = pd.DataFrame(forecasted_values_poly, columns=['Year', 'Month', 'Forecasted Enrollment'])

                st.write('Polynomial Regression Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                st.write(forecast_df)
                selected_year = st.slider('Select Year', 2023, 2032, 2023, key="Polynomial")
                selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="Polynomial1")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                selected_forecast = forecast_df[(forecast_df['Month'] == selected_month) & (forecast_df['Year'] == selected_year)]

                if not selected_forecast.empty:
                    # Display the forecasted enrollment for the selected semester and year
                    st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                    st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                else:
                    # Handle the case where there are no matching records
                    st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                fig = px.line(forecast_df, x='Year', y='Forecasted Enrollment',
                              title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('LSTM Model')
            with st.expander("Click to expand"):
                with st.spinner('Loading...'):
                    time.sleep(1)
                    tf.keras.utils.set_random_seed(42)
                    filtered_data = data[data.index.month.isin([1, 5, 9])]
                    num_years = 10
                    sequence_length = 4
                    forecasted_values_LSTM = []
                    model = Sequential()
                    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    for year in range(2023, 2023 + num_years):
                        for month in [1, 5, 9]:  # January, May, September
                            subset = filtered_data[filtered_data.index.month == month]
                            scaler = MinMaxScaler()
                            scaled_data = scaler.fit_transform(subset[['Enrollment']])
                            sequences = []
                            for i in range(len(scaled_data) - sequence_length):
                                sequences.append(scaled_data[i:i + sequence_length])
                            X = np.array(sequences)
                            y = scaled_data[sequence_length:]
                            X = X.reshape(X.shape[0], X.shape[1], 1)
                            model.fit(X, y, epochs=100, batch_size=64, verbose=0)
                            last_sequence = scaled_data[-sequence_length:]
                            forecast = model.predict(np.array([last_sequence]))
                            forecast = scaler.inverse_transform(forecast)[0][0]
                            forecasted_values_LSTM.append((year, month, forecast))

                    forecast_df_LSTM = pd.DataFrame(forecasted_values_LSTM, columns=['Year', 'Month', 'Forecasted Enrollment'])
                    forecast_df_LSTM['Date'] = pd.to_datetime(forecast_df_LSTM[['Year', 'Month']].assign(day=1))

                    st.write('LSTM Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                    st.write(forecast_df_LSTM)
                    selected_year = st.slider('Select Year', 2023, 2032, 2023, key="LSTM")
                    selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="LSTM1")
                    month_mapping = {'January': 1, 'May': 5, 'September': 9}
                    selected_month = month_mapping[selected_semester]
                    selected_forecast = forecast_df_LSTM[(forecast_df_LSTM['Month'] == selected_month) & (forecast_df_LSTM['Year'] == selected_year)]

                    if not selected_forecast.empty:
                        # Display the forecasted enrollment for the selected semester and year
                        st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                        st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                    else:
                        # Handle the case where there are no matching records
                        st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                    month_names = [month_abbr[month] for month in forecast_df_LSTM['Month']]
                    fig = px.line(forecast_df_LSTM, x='Year', y='Forecasted Enrollment', color=month_names,
                                  title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                    fig.update_layout(legend_title_text='Intake')
                    st.plotly_chart(fig)

        # Mobile Networking Department
        def mobile_networking():
            st.header("Mobile Networking Department")

            def fetch_student_data():
                conn = sqlite3.connect("student_enrollment.db")
                select_query = "SELECT Intake, Enrollment FROM admissions WHERE program = 'Mobile Networking';"
                try:
                    data = pd.read_sql_query(select_query, conn)
                    return data
                except sqlite3.Error as e:
                    st.error(f"Error fetching data: {e}")
                    return pd.DataFrame()

            data = fetch_student_data()

            data['Intake'] = pd.to_datetime(data['Intake'])
            data.set_index('Intake', inplace=True)

            with st.expander("Click for Current Enrollment"):
                student_data = fetch_student_data()
                st.write(student_data)

            st.write("Prediction Models:")

            st.subheader('Prophet Model')
            with st.expander("Click to expand"):
                #Prophet
                prophet_data = data.resample('M').sum()
                prophet_data.reset_index(inplace=True)
                prophet_data.columns = ['ds', 'y']

                model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
                model.fit(prophet_data)
                num_years = 10
                future = model.make_future_dataframe(periods=num_years * 12, freq='M')
                forecast = model.predict(future)
                forecasted_values_prophet = forecast[forecast['ds'].dt.month.isin([1, 5, 9])]
                forecasted_values_prophet = forecasted_values_prophet[['ds', 'yhat']].rename(columns={'yhat': 'Forecasted Enrollment'})

                selected_semester = st.selectbox('Select Semester to forecast:', ['January', 'May', 'September'], key="prophet")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                st.subheader(f"Forecasted Enrollment for Month {selected_semester} Over 10 Years:")
                selected_forecast = forecasted_values_prophet[forecasted_values_prophet['ds'].dt.month == selected_month]
                selected_forecast = selected_forecast[selected_forecast['ds'].dt.year >= 2023]
                st.table(selected_forecast)

                fig = px.line(forecasted_values_prophet, x='ds', y='Forecasted Enrollment', title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('Polynomial Regression')
            with st.expander("Click to expand"):
                #Polynomial
                historical_data = data.resample('M').sum()
                historical_data['Year'] = historical_data.index.year
                historical_data['Month'] = historical_data.index.month
                degree = 2
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(historical_data[['Year', 'Month']])
                model = LinearRegression()
                model.fit(X_poly, historical_data['Enrollment'])
                num_years = 10
                forecasted_values_poly = []
                for year in range(2023, 2023 + num_years):
                    for month in [1, 5, 9]:
                        X_pred = poly.transform([[year, month]])
                        forecast = model.predict(X_pred)
                        forecasted_values_poly.append((year, month, forecast[0]))
                forecast_df = pd.DataFrame(forecasted_values_poly, columns=['Year', 'Month', 'Forecasted Enrollment'])

                st.write('Polynomial Regression Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                st.write(forecast_df)
                selected_year = st.slider('Select Year', 2023, 2032, 2023, key="Polynomial")
                selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="Polynomial1")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]
                selected_forecast = forecast_df[(forecast_df['Month'] == selected_month) & (forecast_df['Year'] == selected_year)]

                if not selected_forecast.empty:
                    # Display the forecasted enrollment for the selected semester and year
                    st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                    st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                else:
                    # Handle the case where there are no matching records
                    st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                fig = px.line(forecast_df, x='Year', y='Forecasted Enrollment',
                              title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                st.plotly_chart(fig)

            st.subheader('LSTM Model')
            with st.expander("Click to expand"):
                with st.spinner('Loading...'):
                    time.sleep(1)
                    tf.keras.utils.set_random_seed(42)
                    filtered_data = data[data.index.month.isin([1, 5, 9])]
                    num_years = 10
                    sequence_length = 4
                    forecasted_values_LSTM = []
                    model = Sequential()
                    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    for year in range(2023, 2023 + num_years):
                        for month in [1, 5, 9]:  # January, May, September
                            subset = filtered_data[filtered_data.index.month == month]
                            scaler = MinMaxScaler()
                            scaled_data = scaler.fit_transform(subset[['Enrollment']])
                            sequences = []
                            for i in range(len(scaled_data) - sequence_length):
                                sequences.append(scaled_data[i:i + sequence_length])
                            X = np.array(sequences)
                            y = scaled_data[sequence_length:]
                            X = X.reshape(X.shape[0], X.shape[1], 1)
                            model.fit(X, y, epochs=100, batch_size=64, verbose=0)
                            last_sequence = scaled_data[-sequence_length:]
                            forecast = model.predict(np.array([last_sequence]))
                            forecast = scaler.inverse_transform(forecast)[0][0]
                            forecasted_values_LSTM.append((year, month, forecast))

                    forecast_df_LSTM = pd.DataFrame(forecasted_values_LSTM, columns=['Year', 'Month', 'Forecasted Enrollment'])
                    forecast_df_LSTM['Date'] = pd.to_datetime(forecast_df_LSTM[['Year', 'Month']].assign(day=1))

                    st.write('LSTM Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                    st.write(forecast_df_LSTM)
                    selected_year = st.slider('Select Year', 2023, 2032, 2023, key="LSTM")
                    selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="LSTM1")
                    month_mapping = {'January': 1, 'May': 5, 'September': 9}
                    selected_month = month_mapping[selected_semester]
                    selected_forecast = forecast_df_LSTM[(forecast_df_LSTM['Month'] == selected_month) & (forecast_df_LSTM['Year'] == selected_year)]

                    if not selected_forecast.empty:
                        # Display the forecasted enrollment for the selected semester and year
                        st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                        st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                    else:
                        # Handle the case where there are no matching records
                        st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                    month_names = [month_abbr[month] for month in forecast_df_LSTM['Month']]
                    fig = px.line(forecast_df_LSTM, x='Year', y='Forecasted Enrollment', color=month_names,
                                  title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                    fig.update_layout(legend_title_text='Intake')
                    st.plotly_chart(fig)

        if selected_department_business == "Computer Science":
            computer_science()

        elif selected_department_business == "Data Science":
            data_science()

        elif selected_department_business == "Mobile Networking":
            mobile_networking()

    page_names_to_funcs = {
        "Faculty of Business" : page_faculty_business,
        "Faculty of Engineering" : page_faculty_engineering,
        "Faculty of Computer Sciences" : page_faculty_computer
    }
    page_names_to_funcs[selected]()
