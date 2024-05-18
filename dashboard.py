import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
import model


st.set_page_config(page_title="Dashboard!!!", page_icon=":bar_chart:", layout="wide")

df = pd.read_csv("Clean_Dataset.csv", encoding="ISO-8859-1")
df_bk=df.copy()

df_bk=df_bk.drop('Unnamed: 0',axis=1)

# Drop the "flight" column
df_bk = df_bk.drop(columns=['flight'])

# Define mappings
airline_mapping = {'SpiceJet':0, 'AirAsia':1, 'Vistara':2, 'GO_FIRST':3, 'Indigo':4, 'Air_India':5}
source_city_mapping = {'Delhi':0, 'Mumbai':1, 'Banglore':2, 'Kolkata':3, 'Hyderabad':4, 'Chennai':5}
departure_time_mapping = {'Evening':0, 'Early_Morning':1, 'Morning':2, 'Afternoon':3, 'Night':4, 'Late_Night':5}
stops_mapping = {'zero':0, 'one':1, 'two_or_more':2}
arrival_time_mapping = {'Night':0, 'Morning':1, 'Early_Morning':2, 'Afternoon':3, 'Evening':4, 'Late_Night':5}
destination_city_mapping = {'Mumbai':0, 'Bangalore':1, 'Kolkata':2, 'Hyderabad':3, 'Chennai':4, 'Delhi':5}
class_mapping = {'Economy':0, 'Business':1}

# Map values for object type columns
df_bk['airline'] = df_bk['airline'].map(airline_mapping)
df_bk['source_city'] = df_bk['source_city'].map(source_city_mapping)
df_bk['departure_time'] = df_bk['departure_time'].map(departure_time_mapping)
df_bk['stops'] = df_bk['stops'].map(stops_mapping)
df_bk['arrival_time'] = df_bk['arrival_time'].map(arrival_time_mapping)
df_bk['destination_city'] = df_bk['destination_city'].map(destination_city_mapping)
df_bk['class'] = df_bk['class'].map(class_mapping)

# Fill NaN values with 0 for numerical columns
numerical_columns = df_bk.select_dtypes(include=['float64','int64']).columns
df_bk[numerical_columns] = df_bk[numerical_columns].fillna(0)


x=df_bk.drop(['price'],axis=1)
y=df_bk['price']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.fit_transform(x_test)
x_train=pd.DataFrame(x_train)
x_test=pd.DataFrame(x_test) 

from sklearn.linear_model import LinearRegression

classifier = LinearRegression()
classifier.fit(x_train,y_train)

st.title(":rocket: Flight Price Prediction")
st.markdown('<style>div.block-container{padding-top:1.5rem;}</style>', unsafe_allow_html=True)

st.sidebar.header("Choose your filter: ")

# Create for Airline
Airline = st.sidebar.multiselect("Select Airline", df["airline"].unique())
if not Airline:
    df2 = df.copy()
else:
    df2 = df[df["airline"].isin(Airline)]

# Create for source city
Source_city = st.sidebar.multiselect("Pick the Source City", df2["source_city"].unique())
if not Source_city:
    df3 = df2.copy()
else:
    df3 = df2[df2["source_city"].isin(Source_city)]

# Create for destination city
Destination_city = st.sidebar.multiselect("Pick the Destination City", df3["destination_city"].unique())
if not Destination_city:
    df4 = df3.copy()
else:
    df4 = df3[df3["destination_city"].isin(Destination_city)]

# Create for class
Class = st.sidebar.multiselect("Pick the Class", df4["class"].unique())
if not Class:
    df5 = df4.copy()
else:
    df5 = df4[df4["class"].isin(Class)]

# Create for departure time
Departure_time = st.sidebar.multiselect("Pick the Departure Time", df5["departure_time"].unique())
if not Departure_time:
    df6 = df5.copy()
else:
    df6 = df5[df5["departure_time"].isin(Departure_time)]

# Create for arrival time
Arrival_time = st.sidebar.multiselect("Pick the Arrival Time", df6["arrival_time"].unique())
if not Arrival_time:
    df7 = df6.copy()
else:
    df7 = df6[df6["arrival_time"].isin(Arrival_time)]

# Create for stops
Stops = st.sidebar.multiselect("Pick the Stops", df7["stops"].unique())
if not Stops:
    df8 = df7.copy()
else:
    df8 = df7[df7["stops"].isin(Stops)]

# Create for duration
Duration = st.sidebar.multiselect("Pick the Duration", df8["duration"].unique())
if not Duration:
    df9 = df8.copy()
else:
    df9 = df8[df8["duration"].isin(Duration)]

# Create for days of left
Days_Left = st.sidebar.multiselect("Pick the Days Left", df9["days_left"].unique())
if not Days_Left:
    df10 = df9.copy()
else:
    df10 = df9[df9["days_left"].isin(Days_Left)]


# Filter data based on user input
if st.sidebar.button("Predict"):
    if Airline and Source_city and Destination_city and Class and Departure_time and Arrival_time and Stops and Duration and Days_Left:
        # Create input dataframe for prediction
        input_data = {
            "airline": [airline_mapping[a] for a in Airline],
            "source_city": [source_city_mapping[sc] for sc in Source_city],
            "departure_time": [departure_time_mapping[dt] for dt in Departure_time],
            "stops": [stops_mapping[s] for s in Stops],
            "arrival_time": [arrival_time_mapping[at] for at in Arrival_time],
            "destination_city": [destination_city_mapping[dc] for dc in Destination_city],
            "class": [class_mapping[c] for c in Class],
            "duration": [float(d) for d in Duration],  # Assuming duration is a float or convert appropriately
            "days_left": [int(dl) for dl in Days_Left]  # Assuming days_left is an integer or convert appropriately
        }
        input_df = pd.DataFrame(input_data)

        # Check if input_df needs reshaping or reordering to fit model input structure
        # Scaling input data as per model's training
        input_df = sc.transform(input_df)  # Use the same scaler object used during training (make sure to load properly if needed)

        # Predicting using the trained model
        predictions = classifier.predict(input_df)
        predicted_price = predictions[0]  # Assuming we want to display the first prediction

        st.sidebar.success(f"Predicted Flight Price: Rs {predicted_price:2f}")
    else:
        st.sidebar.error("Please select all options to predict the flight prices.")


# Define color sequence
colors = px.colors.qualitative.Set3

# Display counts of different airlines
airline_counts = df10["airline"].value_counts().reset_index()
airline_counts.columns = ["Airline", "Count"]

# Display bar chart for counts of different airlines with different colors
fig2 = px.bar(airline_counts, x='Airline', y='Count', title='Counts of Different Airlines', color='Airline', color_discrete_sequence=colors)
st.plotly_chart(fig2, use_container_width=True)

# Display total counts of economy and business class flights
class_counts = df10["class"].value_counts().reset_index()
class_counts.columns = ["Class", "Count"]

# Create columns for pie chart and table
box_plot_column, details_column = st.columns([3,1])

with box_plot_column:
    # Plot box plot showing flight prices vs. airlines
    st.subheader('Flight Prices vs. Airlines')
    fig4 = px.box(df10, x='airline', y='price', color='airline', title='Flight Prices vs. Airlines',
    labels={'airline': 'Airline', 'price': 'Price'})
    st.plotly_chart(fig4, use_container_width=True)

with details_column:
    # Display pie chart for counts of economy and business class flights
    st.subheader('Distribution of Flights by Class')
    st.plotly_chart(px.pie(class_counts, values='Count', names='Class', color='Class', color_discrete_sequence=colors), use_container_width=True)
    
    # Table showing maximum and minimum prices of economy and business class flights
    st.subheader('Price Range by Class')
    min_max_prices = df10.groupby('class')['price'].agg(['min', 'max']).reset_index()
    st.table(min_max_prices)

# Calculate average price and average duration for each combination of source and destination
avg_prices_durations = df10.groupby(['source_city', 'destination_city']).agg({'price': 'mean', 'duration': 'mean'}).reset_index()

# Create columns
avg_prices, avg_duration = st.columns([2,2])

with avg_prices:
    # Plot graph showing average price vs. source-destination route
    st.subheader('Average Price by Route')
    fig_price = px.bar(avg_prices_durations, x='source_city', y='price', color='destination_city',
                    title='Average Price by Source-Destination Route', barmode='group')
    st.plotly_chart(fig_price, use_container_width=True)

with avg_duration:
    # Plot graph showing average duration vs. source-destination route
    st.subheader('Average Duration by Route')
    fig_duration = px.bar(avg_prices_durations, x='source_city', y='duration', color='destination_city',
                        title='Average Duration by Source-Destination Route', barmode='group')
    st.plotly_chart(fig_duration, use_container_width=True)

# Check if source and destination cities are selected
if Source_city and Destination_city:
    flights_data = []
    for source in Source_city:
        for destination in Destination_city:
            # Filter dataframe based on selected source and destination cities
            filtered_df = df10[(df10['source_city'] == source) & (df10['destination_city'] == destination)]
            
            # Calculate total number of flights
            total_flights = len(filtered_df)
            
            # Append data to flights_data list
            flights_data.append({'Source': source, 'Destination': destination, 'Flights': total_flights})
    
    # Create a DataFrame from flights_data
    flights_df = pd.DataFrame(flights_data)
    
    # Display table with source, destination, and total flights
    if len(flights_df) > 0:
        st.subheader('Flights from Selected Source to Destination')
        st.table(flights_df)
    else:
        st.write("No flights found for the selected source and destination.")
else:
    st.write("Select source and destination to check total flights.")

# Calculate average price and duration for each combination of source and destination
avg_prices_durations = df10.groupby(['source_city', 'destination_city']).agg({'price': 'mean', 'duration': 'mean', 'airline': 'count'}).reset_index()

# Create bubble chart for relationship between price, duration, and number of flights
bubble_fig = px.scatter(avg_prices_durations, x='price', y='duration', size='airline',
                        color='source_city', 
                        title='Bubble Chart: Price vs Duration vs Number of Flights',
                        labels={'price': 'Price', 'duration': 'Duration', 'airline': 'Number of Flights'},
                        hover_name='destination_city')
st.plotly_chart(bubble_fig, use_container_width=True)


