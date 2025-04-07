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
You can also choose to view an overall regression model or a model for a specific year, to see how DBE (Days Before Easter) relates to the average temperature.
""")

# File uploader widget
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

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

# --- Function to load and clean each year's data ---
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

# Define keywords to exclude rows not used in charts/KPIs
exclude_keywords = ["additional notes", "florel", "bonzi"]

if uploaded_file is not None:
    try:
        # 1) Read and combine the data
        xls = pd.ExcelFile(uploaded_file)
        df_2023 = load_year_data(xls, '2023', 2023)
        df_2024 = load_year_data(xls, '2024', 2024)
        df_2025 = load_year_data(xls, '2025', 2025)
        df_all = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)

        st.subheader("Combined Historical Data")
        st.dataframe(df_all)

        # Convert 'Removal Date' to datetime
        df_all['Removal Date'] = pd.to_datetime(df_all['Removal Date'], errors='coerce')

        # 2) Exclude unwanted notes
        df_all_filtered = df_all[~df_all["Bulb Type"].str.contains("|".join(exclude_keywords), case=False, na=False)]

        # 3) KPIs and Average DBE Chart
        st.subheader("KPIs for DBE (Filtered)")
        num_years = df_all_filtered["Year"].nunique()
        overall_avg_dbe = df_all_filtered["DBE"].mean()
        st.markdown(f"**Total Years:** {num_years}  |  **Overall Average DBE Pull:** {overall_avg_dbe:.1f} days")

        summary_filtered = df_all_filtered.groupby('Bulb Type').agg({'DBE': 'mean'}).reset_index()
        st.subheader("Average DBE by Bulb Type (Filtered)")
        fig1 = px.bar(
            summary_filtered,
            x="Bulb Type",
            y="DBE",
            title="Average Days Before Easter (DBE) by Bulb Type (Filtered)"
        )
        st.plotly_chart(fig1)

        # 4) DBE vs. Average Temperature with Year Filter
        st.subheader("DBE vs. Average Temperature (Filter by Year)")
        years = sorted(df_all['Year'].dropna().unique().astype(int).tolist())
        year_options = ["All"] + [str(y) for y in years]
        selected_year = st.selectbox("Select Year", options=year_options)
        if selected_year != "All":
            df_filtered = df_all[df_all["Year"] == int(selected_year)]
        else:
            df_filtered = df_all
        fig2 = px.scatter(
            df_filtered,
            x="DBE",
            y="Avg Temp (°F)",
            color="Bulb Type",
            hover_data=["Year", "Removal Date"],
            title="Removal DBE vs. Average Temperature"
        )
        st.plotly_chart(fig2)

        # 5) Regression Model: Overall or By Year
        st.subheader("Regression Model: Predicting Average Temperature from DBE")
        model_choice = st.selectbox("Select Regression Model Type", options=["Overall", "By Year"])

        # Prepare placeholders for models
        model = None
        model_year = None

        # Overall Model
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

                st.write("**Overall Regression Model Results:**")
                st.write("Intercept:", model.intercept_)
                st.write("Coefficient (slope):", model.coef_[0])

                df_model['Predicted Avg Temp (°F)'] = model.predict(X)
                fig3 = px.scatter(
                    df_model,
                    x="DBE",
                    y="Avg Temp (°F)",
                    color="Bulb Type",
                    hover_data=["Year", "Removal Date"],
                    title="Overall: DBE vs. Avg Temp with Regression Line"
                )
                fig3.add_scatter(
                    x=df_model['DBE'],
                    y=df_model['Predicted Avg Temp (°F)'],
                    mode='lines',
                    name='Regression Line'
                )
                st.plotly_chart(fig3)
            else:
                st.info("Not enough data to run the overall regression model.")

        # By Year Model
        else:
            selected_reg_year = st.selectbox("Select Year for Regression", options=sorted(df_all['Year'].dropna().unique().astype(int)))
            df_year = df_all[df_all["Year"] == selected_reg_year].dropna(subset=['DBE', 'Avg Temp (°F)']).copy()
            df_year['DBE'] = pd.to_numeric(df_year['DBE'], errors='coerce')
            df_year['Avg Temp (°F)'] = pd.to_numeric(df_year['Avg Temp (°F)'], errors='coerce')
            df_year = df_year.dropna(subset=['DBE', 'Avg Temp (°F)'])

            if not df_year.empty:
                X = df_year['DBE'].values.reshape(-1, 1)
                y = df_year['Avg Temp (°F)'].values
                model_year = LinearRegression()
                model_year.fit(X, y)

                st.write(f"**Regression Model Results for {selected_reg_year}:**")
                st.write("Intercept:", model_year.intercept_)
                st.write("Coefficient (slope):", model_year.coef_[0])

                df_year['Predicted Avg Temp (°F)'] = model_year.predict(X)
                fig4 = px.scatter(
                    df_year,
                    x="DBE",
                    y="Avg Temp (°F)",
                    color="Bulb Type",
                    hover_data=["Year", "Removal Date"],
                    title=f"Year {selected_reg_year}: DBE vs. Avg Temp with Regression Line"
                )
                fig4.add_scatter(
                    x=df_year['DBE'],
                    y=df_year['Predicted Avg Temp (°F)'],
                    mode='lines',
                    name='Regression Line'
                )
                st.plotly_chart(fig4)
            else:
                st.info(f"Not enough data for the regression model in year {selected_reg_year}.")

        # 6) Recommended Removal Dates Section
        st.subheader("Recommended Removal Dates Based on Easter Date")
        easter_year = st.number_input("Select Easter Year", value=2024, step=1)
        computed_easter = compute_easter(int(easter_year))
        easter_date = pd.to_datetime(computed_easter)
        st.write("Computed Easter Date:", easter_date.strftime("%Y-%m-%d"))

        # Calculate historical average DBE per Bulb Type using filtered data
        avg_dbe = df_all_filtered.groupby("Bulb Type")["DBE"].mean().reset_index().rename(columns={"DBE": "Avg DBE"})
        avg_dbe["Avg DBE"] = pd.to_numeric(avg_dbe["Avg DBE"], errors='coerce')

        def safe_removal_date(dbe, easter):
            """Return the removal date by subtracting dbe days from Easter.
               If dbe is invalid, return NaT."""
            if pd.isnull(dbe):
                return pd.NaT
            try:
                d_int = int(round(dbe))
                return easter - pd.Timedelta(days=d_int)
            except Exception as e:
                st.write("Debug safe_removal_date error:", e)
                return pd.NaT

        avg_dbe["Recommended Removal Date"] = avg_dbe["Avg DBE"].apply(lambda dbe: safe_removal_date(dbe, easter_date))

        # Predict temperature using the chosen regression model
        if model_choice == "Overall" and 'model' in locals() and model is not None:
            avg_dbe["Predicted Avg Temp (°F)"] = avg_dbe["Avg DBE"].apply(
                lambda dbe: (model.intercept_ + model.coef_[0] * dbe) if pd.notnull(dbe) else pd.NA
            )
        elif model_choice == "By Year" and 'model_year' in locals() and model_year is not None:
            avg_dbe["Predicted Avg Temp (°F)"] = avg_dbe["Avg DBE"].apply(
                lambda dbe: (model_year.intercept_ + model_year.coef_[0] * dbe) if pd.notnull(dbe) else pd.NA
            )
        else:
            avg_dbe["Predicted Avg Temp (°F)"] = pd.NA

        st.write("### Recommended Removal Dates and Expected Temperature")
        st.dataframe(avg_dbe)

        st.markdown("""
        **Explanation:**
        - For each bulb type (excluding rows with unwanted notes), the app calculates the historical average DBE (Days Before Easter).
        - The recommended removal date is computed by subtracting the average DBE from the computed Easter date.
        - The regression model (overall or year-specific) predicts the expected average temperature on that removal day.
        """)

        # 7) Calendar View of Recommended Removal Dates
        st.subheader("Calendar View of Recommended Removal Dates")

        try:
            # Define calendar range
            start_date = pd.to_datetime(easter_date) - pd.Timedelta(days=30)
            end_date = pd.to_datetime(easter_date) + pd.Timedelta(days=30)

            # Prepare the data
            timeline_df = avg_dbe[['Bulb Type', 'Recommended Removal Date']].dropna()
            timeline_df['Recommended Removal Date'] = pd.to_datetime(timeline_df['Recommended Removal Date'])

            if timeline_df.empty:
                st.warning("No recommended removal dates available to display on calendar.")
            else:
                fig_calendar = px.scatter(
                    timeline_df,
                    x="Recommended Removal Date",
                    y="Bulb Type",
                    title="Recommended Removal Dates Calendar View",
                    labels={"Recommended Removal Date": "Date", "Bulb Type": "Bulb Type"}
                )

                fig_calendar.add_vline(
                    x=easter_date,
                    line_width=2,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Easter",
                    annotation_position="top right"
                )

                fig_calendar.update_xaxes(range=[start_date, end_date])
                st.plotly_chart(fig_calendar)

        except Exception as e:
            st.error(f"Error processing calendar view: {e}")
        # 8) Historical Trends by Bulb Type
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
