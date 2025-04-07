import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta, date

st.title("Easter Bulb Removal Model Dashboard")

st.markdown("""
This dashboard lets you:
- Upload your **Easter Rules Template_Bulb Removal Model(PM).xlsx** file containing historical data (2023–2025) for bulb removal.
- Automatically load historical weather data from a combined CSV file (stored in the repo root).
- Visualize KPIs, regression models, and trends.
- **Select a future Easter year** to see recommended removal dates and predicted average temperatures.
- View historical and predicted average temperatures (Feb–Apr) to see the weather patterns driving removal timing.
""")

# ==========================================================
# 1. Bulb Removal Data Integration (Excel)
# ==========================================================
uploaded_excel = st.file_uploader("Upload Bulb Removal Excel File", type=["xlsx"])

# --- Function to compute Easter date using the Anonymous Gregorian algorithm ---
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

# --- Function to load and clean each year's data from Excel ---
def load_year_data(xls, sheet_name, year):
    df = xls.parse(sheet_name, header=1)
    required_cols = [
        'Bulb/Tray Type',
        'Removal Date',
        'Removal DBE',
        'Average Temperature from Removal Date (°F)',
        'Growing Degree Days (#)'
    ]
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

exclude_keywords = ["additional notes", "florel", "bonzi"]

if uploaded_excel is not None:
    try:
        xls = pd.ExcelFile(uploaded_excel)
        df_2023 = load_year_data(xls, '2023', 2023)
        df_2024 = load_year_data(xls, '2024', 2024)
        df_2025 = load_year_data(xls, '2025', 2025)
        df_all = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)
        
        st.subheader("Combined Historical Bulb Removal Data")
        st.dataframe(df_all)
        
        df_all['Removal Date'] = pd.to_datetime(df_all['Removal Date'], errors='coerce')
        df_all_filtered = df_all[~df_all["Bulb Type"].str.contains("|".join(exclude_keywords), case=False, na=False)]
        
        st.subheader("KPIs for DBE (Filtered)")
        num_years = df_all_filtered["Year"].nunique()
        overall_avg_dbe = df_all_filtered["DBE"].mean()
        st.markdown(f"**Total Years:** {num_years}  |  **Overall Average DBE:** {overall_avg_dbe:.1f} days")
        
        summary_filtered = df_all_filtered.groupby('Bulb Type').agg({'DBE': 'mean'}).reset_index()
        st.subheader("Average DBE by Bulb Type")
        fig1 = px.bar(summary_filtered, x="Bulb Type", y="DBE", title="Avg DBE by Bulb Type (Filtered)")
        st.plotly_chart(fig1)
        
        st.subheader("DBE vs. Average Temperature (Filter by Year)")
        years = sorted(df_all['Year'].dropna().unique().astype(int).tolist())
        year_options = ["All"] + [str(y) for y in years]
        selected_year = st.selectbox("Select Year", options=year_options)
        if selected_year != "All":
            df_filtered = df_all[df_all["Year"] == int(selected_year)]
        else:
            df_filtered = df_all
        fig2 = px.scatter(df_filtered, x="DBE", y="Avg Temp (°F)", color="Bulb Type",
                          hover_data=["Year", "Removal Date"],
                          title="DBE vs. Avg Temp")
        st.plotly_chart(fig2)
        
        st.subheader("Regression Model: Predicting Avg Temp from DBE")
        model_choice = st.selectbox("Select Regression Model Type", options=["Overall", "By Year"])
        model = None
        model_year = None
        
        if model_choice == "Overall":
            df_model = df_all.dropna(subset=['DBE', 'Avg Temp (°F)']).copy()
            df_model['DBE'] = pd.to_numeric(df_model['DBE'], errors='coerce')
            df_model['Avg Temp (°F)'] = pd.to_numeric(df_model['Avg Temp (°F)'], errors='coerce')
            df_model = df_model.dropna(subset=['DBE', 'Avg Temp (°F)'])
            
            if not df_model.empty:
                X = df_model['DBE'].values.reshape(-1, 1)
                y = df_model['Avg Temp (°F)'].values
                model = LinearRegression()
                model.fit(X, y)
                
                st.write("**Overall Regression Results:**")
                st.write("Intercept:", model.intercept_)
                st.write("Coefficient:", model.coef_[0])
                
                df_model['Predicted Avg Temp (°F)'] = model.predict(X)
                fig3 = px.scatter(df_model, x="DBE", y="Avg Temp (°F)", color="Bulb Type",
                                  hover_data=["Year", "Removal Date"], title="Regression Fit (Overall)")
                fig3.add_scatter(x=df_model['DBE'], y=df_model['Predicted Avg Temp (°F)'],
                                 mode='lines', name="Regression Line")
                st.plotly_chart(fig3)
            else:
                st.info("Not enough data for overall regression.")
        else:
            selected_reg_year = st.selectbox("Select Year for Regression", options=sorted(df_all['Year'].dropna().unique().astype(int)))
            df_year = df_all[df_all["Year"] == selected_reg_year].dropna(subset=['DBE', 'Avg Temp (°F)']).copy()
            df_year['DBE'] = pd.to_numeric(df_year['DBE'], errors='coerce')
            df_year['Avg Temp (°F)'] = pd.to_numeric(df_year['Avg Temp (°F)'], errors='coerce')
            df_year = df_year.dropna(subset=['DBE', 'Avg Temp (°F)'])
            
            if not df_year.empty:
                X = df_year['DBE'].values.reshape(-1, 1)
                y = df_year['Avg Temp (°°F)'].values
                model_year = LinearRegression()
                model_year.fit(X, y)
                
                st.write(f"**Regression Results for {selected_reg_year}:**")
                st.write("Intercept:", model_year.intercept_)
                st.write("Coefficient:", model_year.coef_[0])
                
                df_year['Predicted Avg Temp (°F)'] = model_year.predict(X)
                fig4 = px.scatter(df_year, x="DBE", y="Avg Temp (°F)", color="Bulb Type",
                                  hover_data=["Year", "Removal Date"],
                                  title=f"Regression Fit for {selected_reg_year}")
                fig4.add_scatter(x=df_year['DBE'], y=df_year['Predicted Avg Temp (°F)'],
                                 mode='lines', name="Regression Line")
                st.plotly_chart(fig4)
            else:
                st.info(f"Not enough data for regression in {selected_reg_year}.")
        
        # ===================================================
        # 2. Historical Weather Data Integration (CSV)
        # ===================================================
        st.subheader("Historical Weather Data Integration")
        try:
            # Load the combined weather CSV from the repo root
            weather_csv_path = "Combined Weather Data.csv"
            df_weather = pd.read_csv(weather_csv_path)
            if "datetime" in df_weather.columns:
                df_weather["datetime"] = pd.to_datetime(df_weather["datetime"], errors="coerce")
            
            # --- Fix: If there's no 'TAVG' column, use 'temp' as the average temperature ---
            if "TAVG" not in df_weather.columns:
                if "temp" in df_weather.columns:
                    df_weather["TAVG"] = df_weather["temp"]
                else:
                    st.warning("Neither 'TAVG' nor 'temp' found in weather data.")
            
            st.write("### Weather Data Preview")
            st.dataframe(df_weather.head(20))
            
            # Calculate Growing Degree Days (GDD) with base 40°F
            base_temp = 40
            if "TAVG" in df_weather.columns:
                df_weather["GDD"] = df_weather["TAVG"].apply(lambda t: max(t - base_temp, 0))
                st.write("### Weather Data with GDD Calculation")
                st.dataframe(df_weather[["datetime", "TAVG", "GDD"]].head(20))
            else:
                st.info("No average temperature column available for GDD calculation.")
        except Exception as e:
            st.error(f"Error loading weather data: {e}")
        
        # ===================================================
        # 3. Predicted Future Average Temperature Forecast
        # ===================================================
        st.subheader("Predicted Future Average Temperature (Feb-Apr) by Year")
        if "TAVG" in df_weather.columns:
            df_weather["Year"] = df_weather["datetime"].dt.year
            avg_temp_year = df_weather.groupby("Year")["TAVG"].mean().reset_index()
            st.write("### Historical Average Temperature (Feb-Apr)")
            st.dataframe(avg_temp_year)
            
            # Fit a regression model on historical average temperature vs. year
            X_hist = avg_temp_year["Year"].values.reshape(-1, 1)
            y_hist = avg_temp_year["TAVG"].values
            reg = LinearRegression()
            reg.fit(X_hist, y_hist)
            
            # Predict future temperatures for a range of years (e.g., 2024-2030)
            future_years = list(range(2024, 2031))
            X_future = np.array(future_years).reshape(-1, 1)
            y_future_pred = reg.predict(X_future)
            
            future_df = pd.DataFrame({"Year": future_years, "Predicted TAVG": y_future_pred})
            
            fig_future = px.scatter(avg_temp_year, x="Year", y="TAVG",
                                    title="Historical & Predicted Average Temperature (Feb-Apr)",
                                    labels={"TAVG": "Average Temperature (°F)"})
            all_years = pd.concat([avg_temp_year, future_df.rename(columns={"Predicted TAVG": "TAVG"})])
            all_years = all_years.sort_values("Year")
            fig_future.add_scatter(x=all_years["Year"], y=all_years["TAVG"], mode="lines", name="Trend")
            st.plotly_chart(fig_future)
        else:
            st.info("Weather data does not include an average temperature column ('TAVG' or 'temp').")
        
        # ===================================================
        # 4. Recommended Removal Dates & Predictions
        # ===================================================
        st.subheader("Recommended Removal Dates Based on Easter")
        easter_year_input = st.number_input("Select Easter Year (can be future)", value=2024, step=1)
        computed_easter = compute_easter(int(easter_year_input))
        easter_date = pd.to_datetime(computed_easter)
        st.write("Computed Easter Date:", easter_date.strftime("%Y-%m-%d"))
        
        avg_dbe = df_all_filtered.groupby("Bulb Type")["DBE"].mean().reset_index().rename(columns={"DBE": "Avg DBE"})
        avg_dbe["Avg DBE"] = pd.to_numeric(avg_dbe["Avg DBE"], errors='coerce')
        
        def safe_removal_date(dbe, easter):
            if pd.isnull(dbe):
                return pd.NaT
            try:
                return easter - pd.Timedelta(days=int(round(dbe)))
            except Exception as e:
                st.write("Error in safe_removal_date:", e)
                return pd.NaT
        
        avg_dbe["Recommended Removal Date"] = avg_dbe["Avg DBE"].apply(lambda dbe: safe_removal_date(dbe, easter_date))
        
        if model_choice == "Overall" and model is not None:
            avg_dbe["Predicted Avg Temp (°F)"] = avg_dbe["Avg DBE"].apply(
                lambda dbe: model.intercept_ + model.coef_[0] * dbe if pd.notnull(dbe) else pd.NA
            )
        elif model_choice == "By Year" and model_year is not None:
            avg_dbe["Predicted Avg Temp (°F)"] = avg_dbe["Avg DBE"].apply(
                lambda dbe: model_year.intercept_ + model_year.coef_[0] * dbe if pd.notnull(dbe) else pd.NA
            )
        else:
            avg_dbe["Predicted Avg Temp (°F)"] = pd.NA
        
        st.write("### Recommended Removal Dates and Expected Temperature")
        st.dataframe(avg_dbe)
        
        st.markdown("""
        **Explanation:**
        - For each bulb type (filtered for valid entries), the app computes the historical average DBE (Days Before Easter).
        - The recommended removal date is derived by subtracting that average DBE from the computed Easter date.
        - The regression model (Overall or By Year) is used to predict the expected average temperature on that removal day.
        - The forecast above shows the expected average temperature trends (Feb-Apr) driving these recommendations.
        """)
        
        # ===================================================
        # 5. Visualizing Removal Timing (Days From Easter)
        # ===================================================
        st.subheader("Recommended Removal Timing (Days From Easter)")
        avg_dbe["Days From Easter"] = (avg_dbe["Recommended Removal Date"] - easter_date).dt.days
        
        fig_offset = px.scatter(
            avg_dbe,
            x="Days From Easter",
            y="Bulb Type",
            color="Bulb Type",
            title="Recommended Removal Timing (Days from Easter)",
            labels={"Days From Easter": "Days from Easter (negative = before Easter)"}
        )
        fig_offset.add_vline(x=0, line_width=2, line_dash="dash", line_color="red",
                             annotation_text="Easter", annotation_position="top right")
        st.plotly_chart(fig_offset)
        
        st.subheader("Removal Timing by Bulb Type (Horizontal Bar)")
        fig_bars = px.bar(
            avg_dbe,
            x="Days From Easter",
            y="Bulb Type",
            orientation="h",
            color="Bulb Type",
            title="Recommended Removal Timing (Days from Easter)",
            labels={"Days From Easter": "Days from Easter (negative = before Easter)"}
        )
        fig_bars.add_vline(x=0, line_width=2, line_dash="dash", line_color="red",
                           annotation_text="Easter", annotation_position="top right")
        st.plotly_chart(fig_bars)
        
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
            fig_trend.update_layout(yaxis_title="Avg Temp (°F)")
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
    st.info("Please upload your Bulb Removal Excel file to begin.")
