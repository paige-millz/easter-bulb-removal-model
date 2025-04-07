import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta, date

st.title("Easter Bulb Removal Model Dashboard")

st.markdown("""
This dashboard lets you upload your **Easter Rules Template_Bulb Removal Model(PM).xlsx** file containing historical data for 2023, 2024, and 2025.  
It combines these sheets, displays summary insights and visualizations, and then recommends optimal removal dates based on a user‐selected Easter year.
""")

# File uploader widget
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# Function to compute Easter date using the Anonymous Gregorian algorithm
def compute_easter(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)

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
    # Check for missing columns and fill them if needed
    for col in required_cols:
        if col not in df.columns:
            if col == 'Growing Degree Days (#)':
                df[col] = 0
            else:
                df[col] = pd.NA
    df = df[required_cols]
    df['Year'] = year
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
        
        st.subheader("Combined Historical Data")
        st.dataframe(df_all)
        
        # Convert 'Removal Date' to datetime
        df_all['Removal Date'] = pd.to_datetime(df_all['Removal Date'], errors='coerce')
        
        # --- Exclude rows with notes that shouldn't be used in charts ---
        exclude_keywords = ["florel drench hyac", "bonzi tulips"]
        # Use case-insensitive matching
        df_all_filtered = df_all[~df_all["Bulb Type"].str.contains("|".join(exclude_keywords), case=False, na=False)]
        
        # --- KPIs and Summary for DBE by Bulb Type ---
        st.subheader("KPIs for DBE (Excluding Note Rows)")
        num_years = df_all_filtered["Year"].nunique()
        overall_avg_dbe = df_all_filtered["DBE"].mean()
        st.markdown(f"**Total Years:** {num_years}  |  **Overall Average DBE Pull:** {overall_avg_dbe:.1f} days")
        
        # Summary by Bulb Type from filtered data
        summary_filtered = df_all_filtered.groupby('Bulb Type').agg({
            'DBE': 'mean'
        }).reset_index()
        
        st.subheader("Average DBE by Bulb Type")
        fig1 = px.bar(summary_filtered, x="Bulb Type", y="DBE",
                      title="Average Days Before Easter (DBE) by Bulb Type (Filtered)")
        st.plotly_chart(fig1)
        
        # --- Visualization: DBE vs. Average Temperature with Year Filter ---
        st.subheader("DBE vs. Average Temperature (Filter by Year)")
        years = sorted(df_all["Year"].dropna().unique().astype(int).tolist())
        year_options = ["All"] + [str(y) for y in years]
        selected_year = st.selectbox("Select Year", options=year_options)
        if selected_year != "All":
            df_filtered = df_all[df_all["Year"] == int(selected_year)]
        else:
            df_filtered = df_all
        fig2 = px.scatter(df_filtered, x="DBE", y="Avg Temp (°F)", color="Bulb Type",
                          hover_data=["Year", "Removal Date"],
                          title="Removal DBE vs. Average Temperature")
        st.plotly_chart(fig2)
        
        # --- Regression Model Section ---
        st.subheader("Regression Model: Predicting Average Temperature from DBE")
        df_model = df_all.dropna(subset=['DBE', 'Avg Temp (°F)']).copy()
        df_model['DBE'] = pd.to_numeric(df_model['DBE'], errors='coerce')
        df_model['Avg Temp (°F)'] = pd.to_numeric(df_model['Avg Temp (°F)'], errors='coerce')
        df_model = df_model.dropna(subset=['DBE', 'Avg Temp (°F)'])
        
        if not df_model.empty:
            X = df_model['DBE'].values.reshape(-1, 1)
            y = df_model['Avg Temp (°F)'].values
            model = LinearRegression()
            model.fit(X, y)
            
            st.write("**Regression Model Results:**")
            st.write("Intercept:", model.intercept_)
            st.write("Coefficient (slope):", model.coef_[0])
            
            df_model['Predicted Avg Temp (°F)'] = model.predict(X)
            fig3 = px.scatter(df_model, x="DBE", y="Avg Temp (°F)", color="Bulb Type",
                              hover_data=["Year", "Removal Date"],
                              title="DBE vs. Avg Temp with Regression Line")
            fig3.add_scatter(x=df_model['DBE'], y=df_model['Predicted Avg Temp (°F)'],
                             mode='lines', name='Regression Line')
            st.plotly_chart(fig3)
        else:
            st.info("Not enough data to run the regression model.")
        
        # --- Recommended Removal Dates Section ---
        st.subheader("Recommended Removal Dates Based on Easter Date")
        # Let the user input an Easter year (the app will compute Easter automatically)
        easter_year = st.number_input("Select Easter Year", value=2024, step=1)
        computed_easter = compute_easter(int(easter_year))
        easter_date = pd.to_datetime(computed_easter)
        st.write("Computed Easter Date:", easter_date.strftime("%Y-%m-%d"))
        
        # Calculate the historical average DBE per Bulb Type (using the filtered data)
        avg_dbe = df_all_filtered.groupby("Bulb Type")["DBE"].mean().reset_index().rename(columns={"DBE": "Avg DBE"})
        
        def safe_removal_date(dbe, easter):
            if pd.isnull(dbe):
                return pd.NaT
            return easter - pd.Timedelta(days=float(dbe))
        
        avg_dbe["Recommended Removal Date"] = avg_dbe["Avg DBE"].apply(lambda dbe: safe_removal_date(dbe, easter_date))
        
        # Use the regression model (if available) to predict expected average temperature at that DBE
        if not df_model.empty:
            avg_dbe["Predicted Avg Temp (°F)"] = avg_dbe["Avg DBE"].apply(
                lambda dbe: model.intercept_ + model.coef_[0] * dbe if pd.notnull(dbe) else pd.NA)
        else:
            avg_dbe["Predicted Avg Temp (°F)"] = pd.NA
        
        st.write("### Recommended Removal Dates and Expected Temperature")
        st.dataframe(avg_dbe)
        
        st.markdown("""
        **Explanation:**
        - For each bulb type, the app calculates the historical average DBE (Days Before Easter) at which bulbs were removed (excluding non-relevant rows).
        - The recommended removal date is computed by subtracting the average DBE from the computed Easter date.
        - The regression model predicts the expected average temperature on that removal day.
        """)
        
        # --- Calendar View Section ---
        st.subheader("Calendar View of Recommended Removal Dates")
        # Define a two-month window around Easter (1 month before to 1 month after)
        start_date = easter_date - pd.Timedelta(days=30)
        end_date = easter_date + pd.Timedelta(days=30)
        timeline_df = avg_dbe[['Bulb Type', 'Recommended Removal Date']].dropna()
        timeline_df['Recommended Removal Date'] = pd.to_datetime(timeline_df['Recommended Removal Date'])
        fig_calendar = px.scatter(timeline_df, x="Recommended Removal Date", y="Bulb Type",
                                  title="Recommended Removal Dates Calendar View",
                                  labels={"Recommended Removal Date": "Date", "Bulb Type": "Bulb Type"})
        fig_calendar.add_vline(x=easter_date, line_width=2, line_dash="dash", line_color="red",
                               annotation_text="Easter", annotation_position="top right")
        fig_calendar.update_xaxes(range=[start_date, end_date])
        st.plotly_chart(fig_calendar)
        
        # --- Historical Trends by Bulb Type ---
        st.subheader("Historical Trends by Bulb Type")
        bulb_types = sorted(df_all["Bulb Type"].dropna().unique())
        selected_bulb = st.selectbox("Select Bulb Type", options=bulb_types)
        df_bulb = df_all[df_all["Bulb Type"] == selected_bulb].copy()
        trend = df_bulb.groupby("Year").agg({
            'Avg Temp (°F)': 'mean',
            'Degree Hours >40°F': 'mean'
        }).reset_index()
        if not trend.empty:
            fig_trend = px.line(trend, x="Year", y="Avg Temp (°F)", markers=True,
                                title=f"Historical Avg Temp for {selected_bulb}")
            fig_trend.update_layout(yaxis_title="Average Temperature (°F)")
            st.plotly_chart(fig_trend)
            
            fig_trend2 = px.line(trend, x="Year", y="Degree Hours >40°F", markers=True,
                                 title=f"Historical Degree Hours >40°F for {selected_bulb}")
            fig_trend2.update_layout(yaxis_title="Degree Hours >40°F")
            st.plotly_chart(fig_trend2)
        else:
            st.info("No historical trend data available for the selected bulb type.")
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload your Excel file to begin.")
