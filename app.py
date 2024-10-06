import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Title of the app
st.title("Used Car Price Prediction App")

# Load the dataset from CSV
@st.cache_data
def load_data():
    return pd.read_csv("data_used_car_price.csv")

df = load_data()

# Split the dataset into features and target variable
X = df[['engineSize', 'fuelType', 'year', 'transmission']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

# Creating the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', 'passthrough', ['engineSize', 'year']),
        ('cat', OneHotEncoder(), ['fuelType', 'transmission'])
    ])),
    ('model', LinearRegression())
])

# Fit the model on the training data
pipeline.fit(X_train, y_train)

# Create tabs for Input/Prediction and Visualization
tabs = st.tabs(["Input and Prediction", "Visualization"])

# Input and Prediction Tab
with tabs[0]:
    st.subheader("Input Car Details and Predict Price")
    
    # Select box for brand
    brand = st.selectbox("Select Brand", df['brand'].unique(), key="brand_select")
    
    # Select box for model based on selected brand
    model_name = st.selectbox("Select Model", df[df['brand'] == brand]['model'].unique(), key="model_select")
    
    # Get fuel type and transmission options based on selected model
    fuel_type_options = df[df['model'] == model_name]['fuelType'].unique()
    transmission_options = df[df['model'] == model_name]['transmission'].unique()
    
    # Select box for fuel type based on model
    fuel_type = st.selectbox("Fuel Type", fuel_type_options, key="fuel_type_select")
    
    # Select box for transmission based on model
    transmission = st.selectbox("Transmission", transmission_options, key="transmission_select")
    
    # Slider for engine size
    engine_size = st.slider("Engine Size (in liters)", min_value=1.0, max_value=6.6, value=3.0, step=0.1, key="engine_size_slider")
    
    # Slider for year
    year = st.slider("Year of Manufacture", min_value=1996, max_value=2020, value=2010, step=1, key="year_slider")

    # Display input data when the user presses the button
    if st.button("Predict Price"):
        # Prepare input for prediction
        input_data = pd.DataFrame({
            'engineSize': [engine_size],
            'fuelType': [fuel_type],
            'year': [year],
            'transmission': [transmission]
        })
        
        # Predict the price using the pipeline
        predicted_price = pipeline.predict(input_data)
        st.write(f"**Predicted Price for {brand} {model_name}**: ${predicted_price[0]:.2f}")

# Visualization Tab
with tabs[1]:
    st.subheader("Visualize Car Data")

    # Selectbox to choose the brand for average price visualization
    selected_brand = st.selectbox("Select Brand", df['brand'].unique(), key="visualization_brand_select")

    # Filter the dataframe based on the selected brand
    filtered_df = df[df['brand'] == selected_brand]

    # Average Sales by Transmission for the selected brand
    avg_sales_transmission = filtered_df.groupby('transmission', as_index=False)['price'].mean()

    # Bar Plot for Average Sales by Transmission
    fig_avg_sales_transmission = px.bar(avg_sales_transmission, 
                                         x='transmission', 
                                         y='price', 
                                         title=f"Average Price of {selected_brand} by Transmission",
                                         text='price')

    # Display the average sales by transmission plot
    st.plotly_chart(fig_avg_sales_transmission)

    # Average Sales by Fuel Type for the selected brand
    avg_sales_fuelType = filtered_df.groupby('fuelType', as_index=False)['price'].mean()

    # Bar Plot for Average Sales by Fuel Type
    fig_avg_sales_fuelType = px.bar(avg_sales_fuelType, 
                                     x='fuelType', 
                                     y='price', 
                                     title=f"Average Price of {selected_brand} by Fuel Type",
                                     text='price')

    # Display the average sales by fuel type plot
    st.plotly_chart(fig_avg_sales_fuelType)

    # Scatter Plot
    fig_scatter = px.scatter(df, x='year', y='price', color='brand', size='engineSize', hover_name='model',
                             title="Price vs Year (Scatter Plot)")

    # Display the scatter plot
    st.plotly_chart(fig_scatter)