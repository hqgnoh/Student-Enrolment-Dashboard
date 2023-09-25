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


st.title('University Student Enrollment Dashboard')

#image = Image.open("/content/drive/MyDrive/ColabNotebooks/UCSI-Logo.png")
#st.sidebar.image(image)

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
    with st.sidebar:
        selected = option_menu("List of Faculties", ["Faculty of Business", "Faculty of Engineering", "Faculty of Computer Sciences"],
        icons = ['graph-up','gear','laptop'], menu_icon="list", default_index=0)

    #Faculty of Business
    def page_faculty_business():
        st.title("Faculty of Business")
        selected_department_business = st.selectbox("Select Department:", ["Marketing", "Accounting", "Business Administration"])

        def marketing():
            st.header("Marketing")

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

            # Set random seed for reproducibility
            seed = 42
            np.random.seed(seed)

            data['Intake'] = pd.to_datetime(data['Intake'])
            data.set_index('Intake', inplace=True)

            with st.expander("Click for Current Enrollment"):
                student_data = fetch_student_data()
                st.write(student_data)

            st.write("Select the prediction model:")
            tab1, tab2, tab3 = st.tabs(["Prophet", "Polynomial Regression", "LSTM"])

            with tab1:
                #Prophet

                # Create a DataFrame for Prophet input
                prophet_data = data.resample('M').sum()
                prophet_data.reset_index(inplace=True)
                prophet_data.columns = ['ds', 'y']

                # Create the Prophet model
                model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)

                # Fit the model to the data
                model.fit(prophet_data)

                # Define the number of years you want to forecast
                num_years = 10

                # Create a DataFrame for future dates
                future = model.make_future_dataframe(periods=num_years * 12, freq='M')

                # Generate the forecasts
                forecast = model.predict(future)

                # Extract the forecasts for Jan, May, and Sep
                forecasted_values_prophet = forecast[forecast['ds'].dt.month.isin([1, 5, 9])]
                forecasted_values_prophet = forecasted_values_prophet[['ds', 'yhat']].rename(columns={'yhat': 'Forecasted Enrollment'})

                # Streamlit app
                st.header('Prophet Model')

                selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="prophet")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]

                # Display the forecasted enrollment for the selected month
                st.subheader(f"Forecasted Enrollment for Month {selected_semester}:")

                # Filter the forecasted values for the selected month
                selected_forecast = forecasted_values_prophet[forecasted_values_prophet['ds'].dt.month == selected_month]

                selected_forecast = selected_forecast[selected_forecast['ds'].dt.year >= 2023]

                st.table(selected_forecast)

                # Create a line chart with historical and forecasted values using Plotly
                fig = px.line(forecasted_values_prophet, x='ds', y='Forecasted Enrollment', title='Enrollment Forecast')
                st.plotly_chart(fig)


            with tab2:
                #Polynomial

                # Create a DataFrame for historical data
                historical_data = data.resample('M').sum()

                # Extract year and month for polynomial regression
                historical_data['Year'] = historical_data.index.year
                historical_data['Month'] = historical_data.index.month

                # Define the degree of polynomial for regression
                degree = 2
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(historical_data[['Year', 'Month']])
                model = LinearRegression()
                model.fit(X_poly, historical_data['Enrollment'])

                # Define the number of years you want to forecast
                num_years = 10

                # Create a DataFrame for forecasted values
                forecasted_values_poly = []
                for year in range(2023, 2023 + num_years):
                    for month in [1, 5, 9]:  # January, May, September
                        X_pred = poly.transform([[year, month]])
                        forecast = model.predict(X_pred)
                        forecasted_values_poly.append((year, month, forecast[0]))

                forecast_df = pd.DataFrame(forecasted_values_poly, columns=['Year', 'Month', 'Forecasted Enrollment'])

                # Streamlit app
                st.header('Polynomial Regression')

                # Create a slider for selecting the year
                selected_year = st.slider('Select Year', 2023, 2032, 2023, key="Polynomial")

                # Create a slider for selecting the semester
                selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="Polynomial1")
                month_mapping = {'January': 1, 'May': 5, 'September': 9}
                selected_month = month_mapping[selected_semester]

                # Filter the DataFrame based on the selected semester and year
                selected_forecast = forecast_df[(forecast_df['Month'] == selected_month) & (forecast_df['Year'] == selected_year)]

                # Check if there are matching records before accessing them
                if not selected_forecast.empty:
                    # Display the forecasted enrollment for the selected semester and year
                    st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                    st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                else:
                    # Handle the case where there are no matching records
                    st.subheader(f"No forecast available for {selected_semester} {selected_year}.")

                # Create a line chart with forecasted values using Plotly
                fig = px.line(forecast_df, x='Year', y='Forecasted Enrollment',
                              title='Enrollment Forecast')
                st.plotly_chart(fig)


            with tab3:
                with st.spinner('Loading...'):
                    time.sleep(1)

                    # Filter data for January, May, and September
                    filtered_data = data[data.index.month.isin([1, 5, 9])]

                    # Define the number of years you want to forecast
                    num_years = 10

                    # Define the sequence length
                    sequence_length = 4  # You can adjust this based on your data's patterns

                    # Create an array to hold forecasted values
                    forecasted_values_LSTM = []

                    # Define and compile the LSTM model outside the loop
                    model = Sequential()
                    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')

                    # Loop through the years and months for forecasting
                    for year in range(2023, 2023 + num_years):
                        for month in [1, 5, 9]:  # January, May, September
                            subset = filtered_data[filtered_data.index.month == month]

                            # Normalize data
                            scaler = MinMaxScaler()
                            scaled_data = scaler.fit_transform(subset[['Enrollment']])

                            # Create sequences for LSTM input
                            sequences = []
                            for i in range(len(scaled_data) - sequence_length):
                                sequences.append(scaled_data[i:i + sequence_length])

                            X = np.array(sequences)
                            y = scaled_data[sequence_length:]

                            # Reshape input for LSTM
                            X = X.reshape(X.shape[0], X.shape[1], 1)

                            # Train the model
                            model.fit(X, y, epochs=100, batch_size=64, verbose=0)

                            # Generate forecast using the trained model
                            last_sequence = scaled_data[-sequence_length:]
                            forecast = model.predict(np.array([last_sequence]))
                            forecast = scaler.inverse_transform(forecast)[0][0]
                            forecasted_values_LSTM.append((year, month, forecast))

                    # Create a DataFrame for forecasted values
                    forecast_df_LSTM = pd.DataFrame(forecasted_values_LSTM, columns=['Year', 'Month', 'Forecasted Enrollment'])
                    forecast_df_LSTM['Date'] = pd.to_datetime(forecast_df_LSTM[['Year', 'Month']].assign(day=1))

                    # Streamlit app
                    st.header('LSTM Model')

                    # Display forecasted values
                    st.write('LSTM Forecasted Enrollment for the Next {} Years - Jan, May, Sep:'.format(num_years))
                    st.write(forecast_df_LSTM)

                    # Create a slider for selecting the year
                    selected_year = st.slider('Select Year', 2023, 2032, 2023, key="LSTM")

                    # Create a slider for selecting the semester
                    selected_semester = st.selectbox('Select Semester (Month)', ['January', 'May', 'September'], key="LSTM1")
                    month_mapping = {'January': 1, 'May': 5, 'September': 9}
                    selected_month = month_mapping[selected_semester]

                    # Filter the DataFrame based on the selected semester and year
                    selected_forecast = forecast_df_LSTM[(forecast_df_LSTM['Month'] == selected_month) & (forecast_df_LSTM['Year'] == selected_year)]

                    # Check if there are matching records before accessing them
                    if not selected_forecast.empty:
                        # Display the forecasted enrollment for the selected semester and year
                        st.subheader(f"Forecasted Enrollment for {selected_semester} {selected_year}:")
                        st.header(f"{selected_forecast['Forecasted Enrollment'].values[0]:.2f}")
                    else:
                        # Handle the case where there are no matching records
                        st.subheader(f"No forecast available for {selected_semester} {selected_year}.")


                    # Create a single line chart for all years using Plotly
                    month_names = [month_abbr[month] for month in forecast_df_LSTM['Month']]
                    fig = px.line(forecast_df_LSTM, x='Year', y='Forecasted Enrollment', color=month_names,
                                  title='Forecasted Enrollment for Jan, May, and Sep Over the Years')
                    fig.update_layout(legend_title_text='Intake')
                    st.plotly_chart(fig)

            selected_graphs = st.multiselect("Select Graphs to Display", ['Prophet', 'Polynomial Regression', 'LSTM'])

            if 'Prophet' in selected_graphs:
                st.subheader("Prophet")
                fig = px.line(forecasted_values_prophet, x='ds', y='Forecasted Enrollment', title='Enrollment Forecast')
                st.plotly_chart(fig)

            if 'Polynomial Regression' in selected_graphs:
                st.subheader("Polynomial Regression")
                fig = px.line(forecast_df, x='Year', y='Forecasted Enrollment',
                              title='Enrollment Forecast')
                st.plotly_chart(fig)

            if 'LSTM' in selected_graphs:
                st.subheader("LSTM")
                month_names = [month_abbr[month] for month in forecast_df['Month']]
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
        st.header("Faculty of Engineering")
        st.write("Sort by:")
        tab1, tab2, tab3 = st.tabs(["Chemical", "Mechanical", "Electrical"])

        with tab1:
          col1, col2 = st.columns([2, 3])
          with col1:
            st.subheader("Total Number of Admissions Per Year")
            data = {
              "Year": [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013],
              "Enrollment": [344.00, 343.00, 363.00, 329.00, 330.00, 334.00, 337.00, 312.00, 326.00, 294.00]}
            df = pd.DataFrame(data)
            st.dataframe(df)

          with col2:
            X = np.array(data['Year']).reshape(-1, 1)
            y = np.array(data['Enrollment'])
            poly_transformer = PolynomialFeatures(degree=2)
            X_poly = poly_transformer.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            future_years = np.arange(2023, 2031).reshape(-1, 1)
            future_years_poly = poly_transformer.transform(future_years)
            future_enrollment = model.predict(future_years_poly)

            st.subheader("Select the year to forecast:")
            forecast_years = st.slider("Years", min_value=2023, max_value=2030, value=2023, key="admission")
            forecast_index = np.where(future_years.flatten() == forecast_years)[0][0]

            st.subheader(f"Forecasted Enrollment for {forecast_years}:")
            st.header(f"{future_enrollment[forecast_index]:.2f}")

          fig = go.Figure()
          fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Actual'))
          X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
          X_fit_poly = poly_transformer.transform(X_fit)
          y_fit = model.predict(X_fit_poly)
          fig.add_trace(go.Scatter(x=X_fit.flatten(), y=y_fit, mode='lines', name='Polynomial Regression'))
          future_years_plot = future_years[:forecast_years].flatten()
          fig.add_trace(go.Scatter(x=future_years_plot, y=future_enrollment[:forecast_years], mode='markers', name='Forecasted'))
          fig.update_layout(title='Student Enrollment Over Time (Polynomial Regression)',
                            xaxis_title='Year', yaxis_title='Enrollment')
          st.plotly_chart(fig)

        with tab2:
          col1, col2 = st.columns([2, 3])
          with col1:
            st.subheader("Male and Female Admissions Per Year")
            data = {
              "Year": [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013],
              "Male": [254.00, 256.00, 267.00, 250.00, 245.00, 250.00, 259.00, 229.00, 239.00, 228.00],
              "Female": [90.00, 87.00, 96.00, 79.00, 85.00, 84.00, 78.00, 83.00, 87.00, 66.00]}
            df = pd.DataFrame(data)
            st.dataframe(df)

          with col2:
              X = np.array(data["Year"]).reshape(-1, 1)
              poly_transformer = PolynomialFeatures(degree=2)
              X_poly = poly_transformer.fit_transform(X)
              model_male = LinearRegression()
              model_male.fit(X_poly, data["Male"])
              model_female = LinearRegression()
              model_female.fit(X_poly, data["Female"])

              st.subheader("Select the year to forecast:")
              forecast_years = st.slider("Years", min_value=2023, max_value=2030, value=2023, key="gender")

              future_years = np.arange(2023, 2031).reshape(-1, 1)
              future_years_poly = poly_transformer.transform(future_years)
              future_enrollment_male = model_male.predict(future_years_poly)
              future_enrollment_female = model_female.predict(future_years_poly)

              st.subheader(f"Forecasted Admissions for {forecast_years}:")
              st.subheader(f"Male: {future_enrollment_male[forecast_years - 2023]:.2f}")
              st.subheader(f"Female: {future_enrollment_female[forecast_years - 2023]:.2f}")

          fig = go.Figure()
          fig.add_trace(go.Scatter(x=X.flatten(), y=data["Male"], mode="markers", name="Male"))
          fig.add_trace(go.Scatter(x=X.flatten(), y=data["Female"], mode="markers", name="Female"))
          X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
          X_fit_poly = poly_transformer.transform(X_fit)
          y_fit_male = model_male.predict(X_fit_poly)
          y_fit_female = model_female.predict(X_fit_poly)
          fig.add_trace(go.Scatter(x=X_fit.flatten(), y=y_fit_male, mode="lines", name="Male Polynomial Regression"))
          fig.add_trace(go.Scatter(x=X_fit.flatten(), y=y_fit_female, mode="lines", name="Female Polynomial Regression"))
          future_years_plot = future_years[: forecast_years].flatten()
          fig.add_trace(go.Scatter(x=future_years_plot, y=future_enrollment_male[: forecast_years], mode="markers", name="Forecasted Male"))
          fig.add_trace(go.Scatter(x=future_years_plot, y=future_enrollment_female[: forecast_years], mode="markers", name="Forecasted Female"))

          fig.update_layout(title="Male and Female Admissions Over Time (Polynomial Regression)", xaxis_title="Year", yaxis_title="Admissions")
          st.plotly_chart(fig)

        with tab3:
          col1, col2 = st.columns([2, 3])
          with col1:
            st.subheader("Local and International Admissions Per Year")
            data = {
              "Year": [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013],
              "Local": [208, 208, 219, 224, 224, 227, 253, 234, 245, 221],
              "International": [136, 135, 144, 105, 106, 107, 84, 78, 81, 73]}
            df = pd.DataFrame(data)
            st.dataframe(df)

          with col2:
              st.subheader("Select the category for forecast:")
              category = st.selectbox("Category", ["Local", "International"])

              X = np.array(data["Year"]).reshape(-1, 1)
              if category == "Local":
                  y = np.array(data["Local"])
              else:
                  y = np.array(data["International"])

              poly_transformer = PolynomialFeatures(degree=2)
              X_poly = poly_transformer.fit_transform(X)
              model = LinearRegression()
              model.fit(X_poly, y)

              st.subheader("Select the year to forecast:")
              forecast_years = st.slider("Years", min_value=2023, max_value=2030, value=2023, key="Ethnicity")

              future_years = np.arange(2023, 2031).reshape(-1, 1)
              future_years_poly = poly_transformer.transform(future_years)
              future_enrollment = model.predict(future_years_poly)

              st.subheader(f"Forecasted {category} Admissions for {forecast_years}:")
              st.header(f"{future_enrollment[forecast_years - 2023]:.2f}")

          fig = go.Figure()
          fig.add_trace(go.Scatter(x=X.flatten(), y=data["Local"], mode="markers", name="Local"))
          fig.add_trace(go.Scatter(x=X.flatten(), y=data["International"], mode="markers", name="International"))
          X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
          X_fit_poly = poly_transformer.transform(X_fit)
          y_fit = model.predict(X_fit_poly)
          fig.add_trace(go.Scatter(x=X_fit.flatten(), y=y_fit, mode="lines", name="Polynomial Regression"))
          future_years_plot = future_years[: forecast_years].flatten()
          fig.add_trace(
              go.Scatter(
                  x=future_years_plot,
                  y=future_enrollment[: forecast_years],
                  mode="markers",
                  name="Forecasted",
              )
          )
          fig.update_layout(
              title=f"{category} Admissions Over Time (Polynomial Regression)",
              xaxis_title="Year",
              yaxis_title="Admissions",
          )
          st.plotly_chart(fig)

    #Faculty of Computer Sciences
    def page_faculty_computer():
        st.header("Faculty of Computer Sciences")
        st.write("Sort by:")
        tab1, tab2, tab3 = st.tabs(["Computer Science", "Data Science", "Mobile Networking"])

        with tab1:
          col1, col2 = st.columns([2, 3])
          with col1:
            st.subheader("Total Number of Admissions Per Year")
            data = {
              "Year": [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013],
              "Enrollment": [136.00, 130.00, 119.00, 116.00, 133.00, 105.00, 99.00, 91.00, 101.00, 86.00]}
            df = pd.DataFrame(data)
            st.dataframe(df)

          with col2:
            X = np.array(data['Year']).reshape(-1, 1)
            y = np.array(data['Enrollment'])
            poly_transformer = PolynomialFeatures(degree=2)
            X_poly = poly_transformer.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            future_years = np.arange(2023, 2031).reshape(-1, 1)
            future_years_poly = poly_transformer.transform(future_years)
            future_enrollment = model.predict(future_years_poly)

            st.subheader("Select the year to forecast:")
            forecast_years = st.slider("Years", min_value=2023, max_value=2030, value=2023, key="admission")
            forecast_index = np.where(future_years.flatten() == forecast_years)[0][0]

            st.subheader(f"Forecasted Enrollment for {forecast_years}:")
            st.header(f"{future_enrollment[forecast_index]:.2f}")

          fig = go.Figure()
          fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Actual'))
          X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
          X_fit_poly = poly_transformer.transform(X_fit)
          y_fit = model.predict(X_fit_poly)
          fig.add_trace(go.Scatter(x=X_fit.flatten(), y=y_fit, mode='lines', name='Polynomial Regression'))
          future_years_plot = future_years[:forecast_years].flatten()
          fig.add_trace(go.Scatter(x=future_years_plot, y=future_enrollment[:forecast_years], mode='markers', name='Forecasted'))
          fig.update_layout(title='Student Enrollment Over Time (Polynomial Regression)',
                            xaxis_title='Year', yaxis_title='Enrollment')
          st.plotly_chart(fig)

        with tab2:
          col1, col2 = st.columns([2, 3])
          with col1:
            st.subheader("Male and Female Admissions Per Year")
            data = {
              "Year": [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013],
              "Male": [109.00, 103.00, 90.00, 93.00, 103.00, 84.00, 86.00, 75.00, 87.00, 74.00],
              "Female": [27.00, 27.00, 29.00, 23.00, 30.00, 21.00, 13.00, 16.00, 14.00, 12.00]}
            df = pd.DataFrame(data)
            st.dataframe(df)

          with col2:
              X = np.array(data["Year"]).reshape(-1, 1)
              poly_transformer = PolynomialFeatures(degree=2)
              X_poly = poly_transformer.fit_transform(X)
              model_male = LinearRegression()
              model_male.fit(X_poly, data["Male"])
              model_female = LinearRegression()
              model_female.fit(X_poly, data["Female"])

              st.subheader("Select the year to forecast:")
              forecast_years = st.slider("Years", min_value=2023, max_value=2030, value=2023, key="gender")

              future_years = np.arange(2023, 2031).reshape(-1, 1)
              future_years_poly = poly_transformer.transform(future_years)
              future_enrollment_male = model_male.predict(future_years_poly)
              future_enrollment_female = model_female.predict(future_years_poly)

              st.subheader(f"Forecasted Admissions for {forecast_years}:")
              st.subheader(f"Male: {future_enrollment_male[forecast_years - 2023]:.2f}")
              st.subheader(f"Female: {future_enrollment_female[forecast_years - 2023]:.2f}")

          fig = go.Figure()
          fig.add_trace(go.Scatter(x=X.flatten(), y=data["Male"], mode="markers", name="Male"))
          fig.add_trace(go.Scatter(x=X.flatten(), y=data["Female"], mode="markers", name="Female"))
          X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
          X_fit_poly = poly_transformer.transform(X_fit)
          y_fit_male = model_male.predict(X_fit_poly)
          y_fit_female = model_female.predict(X_fit_poly)
          fig.add_trace(go.Scatter(x=X_fit.flatten(), y=y_fit_male, mode="lines", name="Male Polynomial Regression"))
          fig.add_trace(go.Scatter(x=X_fit.flatten(), y=y_fit_female, mode="lines", name="Female Polynomial Regression"))
          future_years_plot = future_years[: forecast_years].flatten()
          fig.add_trace(go.Scatter(x=future_years_plot, y=future_enrollment_male[: forecast_years], mode="markers", name="Forecasted Male"))
          fig.add_trace(go.Scatter(x=future_years_plot, y=future_enrollment_female[: forecast_years], mode="markers", name="Forecasted Female"))

          fig.update_layout(title="Male and Female Admissions Over Time (Polynomial Regression)", xaxis_title="Year", yaxis_title="Admissions")
          st.plotly_chart(fig)

        with tab3:
          col1, col2 = st.columns([2, 3])
          with col1:
            st.subheader("Local and International Admissions Per Year")
            data = {
              "Year": [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013],
              "Local": [83, 80, 73, 89, 102, 81, 79, 72, 80, 68],
              "International": [53, 50, 46, 27, 31, 24, 20, 19, 21, 18]}
            df = pd.DataFrame(data)
            st.dataframe(df)

          with col2:
              st.subheader("Select the category for forecast:")
              category = st.selectbox("Category", ["Local", "International"])

              X = np.array(data["Year"]).reshape(-1, 1)
              if category == "Local":
                  y = np.array(data["Local"])
              else:
                  y = np.array(data["International"])

              poly_transformer = PolynomialFeatures(degree=2)
              X_poly = poly_transformer.fit_transform(X)
              model = LinearRegression()
              model.fit(X_poly, y)

              st.subheader("Select the year to forecast:")
              forecast_years = st.slider("Years", min_value=2023, max_value=2030, value=2023, key="Ethnicity")

              future_years = np.arange(2023, 2031).reshape(-1, 1)
              future_years_poly = poly_transformer.transform(future_years)
              future_enrollment = model.predict(future_years_poly)

              st.subheader(f"Forecasted {category} Admissions for {forecast_years}:")
              st.header(f"{future_enrollment[forecast_years - 2023]:.2f}")

          fig = go.Figure()
          fig.add_trace(go.Scatter(x=X.flatten(), y=data["Local"], mode="markers", name="Local"))
          fig.add_trace(go.Scatter(x=X.flatten(), y=data["International"], mode="markers", name="International"))
          X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
          X_fit_poly = poly_transformer.transform(X_fit)
          y_fit = model.predict(X_fit_poly)
          fig.add_trace(go.Scatter(x=X_fit.flatten(), y=y_fit, mode="lines", name="Polynomial Regression"))
          future_years_plot = future_years[: forecast_years].flatten()
          fig.add_trace(
              go.Scatter(
                  x=future_years_plot,
                  y=future_enrollment[: forecast_years],
                  mode="markers",
                  name="Forecasted",
              )
          )
          fig.update_layout(
              title=f"{category} Admissions Over Time (Polynomial Regression)",
              xaxis_title="Year",
              yaxis_title="Admissions",
          )
          st.plotly_chart(fig)

    page_names_to_funcs = {
        "Faculty of Business" : page_faculty_business,
        "Faculty of Engineering" : page_faculty_engineering,
        "Faculty of Computer Sciences" : page_faculty_computer
    }

    page_names_to_funcs[selected]()
