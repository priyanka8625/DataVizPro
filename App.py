import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport 
import streamlit.components.v1 as components  
import time
from sklearn.impute import SimpleImputer
import mymodule as mm
from sklearn.preprocessing import RobustScaler

# App header and description
st.header("Welcome to DataVizPro")
st.markdown('<h3>Created by: Priyanka Sundalam</h3>', unsafe_allow_html=True)
st.markdown("""
<p>Easily explore and understand your data with our simple EDA tool. Just upload your dataset, and our tool will analyze, clean, and visualize it—helping you find patterns and insights quickly. No complex coding required, just clear and easy-to-read summaries and charts to make data analysis effortless!</p>
""", unsafe_allow_html=True)
st.snow()

# File uploader for CSV
st.markdown("<h4>Upload your Dataset here in CSV format</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Display dataset preview
    df = pd.read_csv(uploaded_file)
    st.write("Here's the preview of your uploaded dataset")
    st.write(df.head())
    st.markdown("<h5>This is your analysis before cleaning and preprocessing the data:</h5>", unsafe_allow_html=True)

    # Generate and display EDA report before cleaning
    report = ProfileReport(df, explorative=True)
    loading_bar = st.progress(0)
    for percent_complete in range(0, 101, 10):
        time.sleep(0.8) 
        loading_bar.progress(percent_complete)
    components.html(report.to_html(), width=650, height=1000, scrolling=True)

    # Data cleaning and preprocessing
    st.markdown("<h6>This is your analysis after cleaning and preprocessing the data:</h6>", unsafe_allow_html=True)
    df.drop_duplicates(inplace=True) # Removes duplicated rows
    shape = df.shape
    columns_with_nulls = df.columns[df.isnull().any()].tolist()
    percent_null_dict = (df.isna().sum() / df.shape[0] * 100).to_dict()

    for key, value in percent_null_dict.items():
        if value > 30:
            df.drop(key, axis=1, inplace=True)  # Drop columns with >30% null values
        else:
            dtype = df[key].dtype
            if dtype == 'object':
                imputer = SimpleImputer(strategy='most_frequent')  # Impute categorical columns
            else:
                outliers = mm.detect_outliers_iqr(df[key]).tolist()
                if outliers:
                    imputer = SimpleImputer(strategy='median')  # Impute numerical columns
                    scaler = RobustScaler()  # Scale numerical columns with outliers
                    df[key] = scaler.fit_transform(df[[key]]).ravel()
                else:
                    imputer = SimpleImputer(strategy='mean')  # Impute numerical columns
                    scaler = RobustScaler()  # Scale numerical columns with outliers
                    df[key] = scaler.fit_transform(df[[key]]).ravel()
            df[key] = imputer.fit_transform(df[[key]]).ravel()

    # Generate and display EDA report after cleaning
    report2 = ProfileReport(df, explorative=True)
    loading_bar = st.progress(0)
    for percent_complete in range(0, 101, 10):
        time.sleep(0.8) 
        loading_bar.progress(percent_complete)
    components.html(report2.to_html(), width=650, height=1000, scrolling=True)
    csv = df.to_csv(index=False).encode('utf-8')
    st.markdown("<h6>Download your cleaned dataset:</h6>", unsafe_allow_html=True)
    st.download_button(
    label="Download Cleaned Data as CSV",
    data=csv,
    file_name="cleaned_dataset.csv",
    mime="text/csv"
)
else:
    st.write("Please upload a CSV file to get started.")
