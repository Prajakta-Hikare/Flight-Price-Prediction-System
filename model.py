import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)


df=pd.read_csv('Clean_Dataset.csv')
df.head()

df=df.drop('Unnamed: 0',axis=1)

# Drop the "flight" column
df = df.drop(columns=['flight'])

df.info()

df.describe()

df.shape

df.isnull().sum()

df_bk=df.copy()

# Filter columns with dtype as object
object_columns = df.select_dtypes(include=['object'])

# Iterate over each object column and print unique values
for column in object_columns.columns:
    unique_values = df[column].unique()
    print(f"Unique values of {column}: {unique_values}")

# Define mappings
airline_mapping = {'SpiceJet':0, 'AirAsia':1, 'Vistara':2, 'GO_FIRST':3, 'Indigo':4, 'Air_India':5}
source_city_mapping = {'Delhi':0, 'Mumbai':1, 'Banglore':2, 'Kolkata':3, 'Hyderabad':4, 'Chennai':5}
departure_time_mapping = {'Evening':0, 'Early_Morning':1, 'Morning':2, 'Afternoon':3, 'Night':4, 'Late_Night':5}
stops_mapping = {'zero':0, 'one':1, 'two_or_more':2}
arrival_time_mapping = {'Night':0, 'Morning':1, 'Early_Morning':2, 'Afternoon':3, 'Evening':4, 'Late_Night':5}
destination_city_mapping = {'Mumbai':0, 'Bangalore':1, 'Kolkata':2, 'Hyderabad':3, 'Chennai':4, 'Delhi':5}
class_mapping = {'Economy':0, 'Business':1}

# Map values for object type columns
df['airline'] = df['airline'].map(airline_mapping)
df['source_city'] = df['source_city'].map(source_city_mapping)
df['departure_time'] = df['departure_time'].map(departure_time_mapping)
df['stops'] = df['stops'].map(stops_mapping)
df['arrival_time'] = df['arrival_time'].map(arrival_time_mapping)
df['destination_city'] = df['destination_city'].map(destination_city_mapping)
df['class'] = df['class'].map(class_mapping)

# Fill NaN values with 0 for numerical columns
numerical_columns = df.select_dtypes(include=['float64','int64']).columns
df[numerical_columns] = df[numerical_columns].fillna(0)

# Display the DataFrame
df.head()

df.tail()

x=df.drop(['price'],axis=1)
y=df['price']

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

y_pred = classifier.predict(x_test)

y_pred.shape, y_test.shape

from sklearn import metrics

print('Mean Absolute Error (MAE):',metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error (RMSE):',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2_score:',metrics.r2_score(y_test, y_pred))

# Example input data (replace with your actual input data)
input_data = {
    'airline': ['SpiceJet'],  # Example airline
    'source_city': ['Delhi'],  # Example source city
    'departure_time': ['Evening'],  # Example departure time
    'stops': ['zero'],  # Example number of stops
    'arrival_time': ['Night'],  # Example arrival time
    'destination_city': ['Mumbai'],  # Example destination city
    'class': ['Economy'],  # Example class
    'duration': [2.17],  # Example duration in hours
    'days_left': [1]  # Example days left for booking
}

# Convert input data to DataFrame
input_df = pd.DataFrame(input_data)

# Map values for object type columns
input_df['airline'] = input_df['airline'].map(airline_mapping)
input_df['source_city'] = input_df['source_city'].map(source_city_mapping)
input_df['departure_time'] = input_df['departure_time'].map(departure_time_mapping)
input_df['stops'] = input_df['stops'].map(stops_mapping)
input_df['arrival_time'] = input_df['arrival_time'].map(arrival_time_mapping)
input_df['destination_city'] = input_df['destination_city'].map(destination_city_mapping)
input_df['class'] = input_df['class'].map(class_mapping)

def rufun(input_data):
    # Scale input data
    input_scaled = sc.transform(input_data)

    # Make prediction
    predicted_price = classifier.predict(input_scaled)

    print("Predicted flight price:", predicted_price)



