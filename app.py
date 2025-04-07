import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Easter Bulb Removal Model Dashboard")

st.markdown("""
This dashboard lets you upload your **Easter Rules Template_Bulb Removal Model(PM).xlsx** file, which contains historical data for 2023, 2024, and 2025.  
The app will combine these sheets into one dataset and display summary insights and visualizations.
""")

# File uploader widget
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

def load_year_data(xls, sheet_name, year):
    # Parse the specified sheet using row 1 as the header
    df = xls.parse(sheet_name, header=1)
    # Define the required columns
    required_cols = [
        'Bulb/Tray Type',
        'Removal Date',
        'Removal DBE',
        'Average Temperature from Removal Date (°F)',
        'Growing Degree Days (#)'
    ]
    
    # Check for missing columns and add them if needed
    for col in required_cols:
        if col not in df.columns:
            if col == 'Growing Degree Days (#)':
                df[col] = 0
            else:
                df[col] = pd.NA  # Fill with NA for other missing columns

    # Select only the required columns
    df = df[required_cols]
    # Add a column for the year
    df['Year'] = year
    # Rename columns for consistency
    df = df.rename(columns={
        'Bulb/Tray Type': 'Bulb Type',
        'Removal DBE': 'DBE',
        'Average Temperature from Removal Date (°F)': 'Avg Temp (°F)',
        'Growing Degree Days (#)': 'Degree Hours >40°F'
    })
    return df

if uploaded_file is not None:
    try:
        # Read the uploaded Excel file
        xls = pd.ExcelFile(uploaded_file)
        
        # Load data from each year
        df_2023 = load_year_data(xls, '2023', 2023)
        df_2024 = load_year_data(xls, '2024', 2024)
        df_2025 = load_year_data(xls, '2025', 2025)
        
        # Combine data from all years
        df_all = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)
        
        st.subheader("Combined Data")
        st.dataframe(df_all)
        
        # Convert 'Removal Date' column to datetime
        df_all['Removal Date'] = pd.to_datetime(df_all['Removal Date'], errors='coerce')
        
        # Summary statistics by Bulb Type
        st.subheader("Summary by Bulb Type")
        summary = df_all.groupby('Bulb Type').agg({
            'DBE': 'mean',
            'Avg Temp (°F)': 'mean',
            'Degree Hours >40°F': 'mean'
        }).reset_index()
        st.dataframe(summary)
        
        # Bar chart: Average DBE by Bulb Type
        st.subheader("Average DBE by Bulb Type")
        fig1 = px.bar(summary, x="Bulb Type", y="DBE", 
                      title="Average Days Before Easter (DBE) by Bulb Type")
        st.plotly_chart(fig1)
        
        # Scatter plot: DBE vs. Average Temperature
        st.subheader("DBE vs. Average Temperature")
        fig2 = px.scatter(df_all, x="DBE", y="Avg Temp (°F)", color="Bulb Type",
                          hover_data=["Year", "Removal Date"],
                          title="Removal DBE vs. Average Temperature")
        st.plotly_chart(fig2)
        
        st.markdown("""
        **Next Steps for the Model:**
        - Compare historical yield outcomes (once available) against DBE and temperature data.
        - Identify optimal removal DBE ranges per bulb type.
        - Use a regression or rule-based model to suggest adjustments.
        """)
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload your Excel file to begin.")