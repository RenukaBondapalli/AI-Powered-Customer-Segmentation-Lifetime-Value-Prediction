"""
AI-Powered Customer Segmentation & Lifetime Value Prediction
------------------------------------------------------------
An interactive Streamlit application that:
1. Performs RFM analysis on customer data.
2. Segments customers using K-Means clustering.
3. Predicts Customer Lifetime Value (LTV) using regression models.
4. Displays insights through visual dashboards.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
import datetime as dt
import plotly.express as px

# 1. App Config
st.set_page_config(
    page_title="Customer Segmentation & LTV",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛍️ Customer Segmentation & LTV Predictor")

# 2. Sidebar
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload your customer CSV", type=["csv"])
k_clusters = st.sidebar.slider("Number of Segments (k)", 2, 10, 4)

# 3. Process Data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Clean
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # RFM
    NOW = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (NOW - x.max()).days,
        'InvoiceNo': 'count',
        'TotalAmount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Segmentation
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

    # LTV Calculation
    customer_ltv = df.groupby('CustomerID').agg({
        'TotalAmount': 'mean',
        'InvoiceNo': 'count'
    }).reset_index()
    customer_ltv['LTV'] = customer_ltv['TotalAmount'] * customer_ltv['InvoiceNo'] * 1

    # LTV Prediction
    X = customer_ltv[['TotalAmount','InvoiceNo']]
    y = customer_ltv['LTV']
    model = xgb.XGBRegressor()
    model.fit(X, y)
    customer_ltv['Predicted_LTV'] = model.predict(X)

    # 4. Tabs for UI
    tabs = st.tabs(["📊 Customer Segmentation", "💰 LTV Prediction"])

    # ----------- SEGMENTATION TAB -----------
    with tabs[0]:
        st.subheader("Segment Overview")
        seg_counts = rfm['Segment'].value_counts().sort_index()
        st.metric("Total Customers", rfm['CustomerID'].nunique())
        st.metric("Segments Created", k_clusters)

        # Pie Chart - Plotly
        fig1 = px.pie(
            rfm, names='Segment', title='Customer Segment Distribution',
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Bar Chart - Average Monetary
        seg_monetary = rfm.groupby('Segment')['Monetary'].mean().reset_index()
        fig2 = px.bar(
            seg_monetary, x='Segment', y='Monetary',
            title='Average Monetary Value per Segment',
            color='Monetary', color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Table of RFM + Segment
        st.subheader("Customer RFM Data with Segments")
        st.dataframe(rfm)

    # ----------- LTV TAB -----------
    with tabs[1]:
        st.subheader("Predicted LTV")
        # Histogram
        fig3 = px.histogram(customer_ltv, x='Predicted_LTV', nbins=20, title='Predicted LTV Distribution')
        st.plotly_chart(fig3, use_container_width=True)

        # Top Customers
        st.subheader("Top 10 Customers by Predicted LTV")
        st.dataframe(customer_ltv.sort_values(by='Predicted_LTV', ascending=False).head(10))

        # Metrics
        st.metric("Average Predicted LTV", f"${customer_ltv['Predicted_LTV'].mean():.2f}")
        st.metric("Maximum Predicted LTV", f"${customer_ltv['Predicted_LTV'].max():.2f}")