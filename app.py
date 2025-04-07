import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta, date

st.set_page_config(page_title="Easter Bulb Removal Dashboard", layout="wide")
st.title("🌷 Easter Bulb Removal Model Dashboard")

st.markdown("""
This dashboard helps growers predict **optimal bulb removal dates** based on historical data.  
Upload your bulb tracking sheet and select a future Easter year to get **removal recommendations** and expected weather.
""")

# File uploader widget
uploaded_file = st.file_uploader("📤 Upload your Excel file", type=["xlsx"])

# Easter date calculator
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

# Load and clean yearly data
def load_year_data(xls, sheet_name, year):
    df = xls.parse(sheet_name, header=1)
    required_cols = [
        'Bulb/Tray Type', 'Removal Date', 'Removal DBE',
        'Average Temperature from Removal Date (°F)', 'Growing Degree Days (#)'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 if 'Degree' in col else pd.NA
    df = df[required_cols]
    df['Year'] = year
    return df.rename(columns={
        'Bulb/Tray Type': 'Bulb Type',
        'Removal DBE': 'DBE',
        'Average Temperature from Removal Date (°F)': 'Avg Temp (°F)',
        'Growing Degree Days (#)': 'Degree Hours >40°F'
    })

exclude_keywords = ["additional notes", "florel", "bonzi"]

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        df_all = pd.concat([
            load_year_data(xls, '2023', 2023),
            load_year_data(xls, '2024', 2024),
            load_year_data(xls, '2025', 2025)
        ], ignore_index=True)

        df_all['Removal Date'] = pd.to_datetime(df_all['Removal Date'], errors='coerce')
        df_all_filtered = df_all[~df_all["Bulb Type"].str.contains("|".join(exclude_keywords), case=False, na=False)]

        # KPIs
        st.subheader("📊 Historical KPIs and Trends")
        st.markdown(f"""
        **Total Years:** {df_all_filtered["Year"].nunique()}  
        **Overall Avg DBE:** {df_all_filtered["DBE"].mean():.1f} days before Easter
        """)

        fig_dbe = px.bar(
            df_all_filtered.groupby("Bulb Type")["DBE"].mean().reset_index(),
            x="Bulb Type", y="DBE",
            title="Average Days Before Easter (DBE) by Bulb Type"
        )
        st.plotly_chart(fig_dbe, use_container_width=True)

        # DBE vs. Temp
        st.subheader("📈 DBE vs. Avg Temperature")
        selected_year = st.selectbox("Select Year for Comparison", ["All"] + sorted(df_all["Year"].dropna().unique().astype(str).tolist()))
        df_year_filtered = df_all if selected_year == "All" else df_all[df_all["Year"] == int(selected_year)]
        fig_dbe_temp = px.scatter(
            df_year_filtered, x="DBE", y="Avg Temp (°F)", color="Bulb Type",
            hover_data=["Year", "Removal Date"], title="DBE vs. Temperature"
        )
        st.plotly_chart(fig_dbe_temp, use_container_width=True)

        # Regression
        st.subheader("📉 Regression Model: Predicting Temperature")
        model_choice = st.selectbox("Choose Model Type", ["Overall", "By Year"])
        model, model_df = None, None

        if model_choice == "Overall":
            model_df = df_all.dropna(subset=['DBE', 'Avg Temp (°F)'])
        else:
            year = st.selectbox("Choose Year", sorted(df_all["Year"].unique()))
            model_df = df_all[df_all["Year"] == year].dropna(subset=['DBE', 'Avg Temp (°F)'])

        model_df['DBE'] = pd.to_numeric(model_df['DBE'], errors='coerce')
        model_df['Avg Temp (°F)'] = pd.to_numeric(model_df['Avg Temp (°F)'], errors='coerce')
        model_df.dropna(subset=['DBE', 'Avg Temp (°F)'], inplace=True)

        if not model_df.empty:
            X = model_df["DBE"].values.reshape(-1, 1)
            y = model_df["Avg Temp (°F)"].values
            model = LinearRegression().fit(X, y)
            model_df["Predicted"] = model.predict(X)

            fig_reg = px.scatter(model_df, x="DBE", y="Avg Temp (°F)", color="Bulb Type",
                                 title="Regression: DBE vs. Temperature")
            fig_reg.add_scatter(x=model_df["DBE"], y=model_df["Predicted"], mode="lines", name="Regression Line")
            st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.info("Not enough data to fit the regression model.")

        # Removal Date Prediction
        st.subheader("📅 Recommended Removal Dates Based on Easter")
        easter_year = st.number_input("Select Easter Year", value=2024, step=1)
        easter_date = pd.to_datetime(compute_easter(easter_year))
        st.write(f"Calculated Easter Date: `{easter_date.strftime('%Y-%m-%d')}`")

        avg_dbe = df_all_filtered.groupby("Bulb Type")["DBE"].mean().reset_index().rename(columns={"DBE": "Avg DBE"})
        avg_dbe["Recommended Removal Date"] = avg_dbe["Avg DBE"].apply(
            lambda dbe: easter_date - pd.Timedelta(days=int(round(dbe))) if pd.notnull(dbe) else pd.NaT
        )

        # Predict temps
        if model:
            avg_dbe["Predicted Avg Temp (°F)"] = avg_dbe["Avg DBE"].apply(
                lambda dbe: model.intercept_ + model.coef_[0] * dbe if pd.notnull(dbe) else pd.NA
            )

        st.dataframe(avg_dbe)

        # Visual Timeline
        st.subheader("📆 Visual Timeline of Removal Dates")
        try:
            df_timeline = avg_dbe.dropna(subset=["Recommended Removal Date"]).copy()
            df_timeline["End Date"] = df_timeline["Recommended Removal Date"] + pd.Timedelta(days=1)

            fig_timeline = px.timeline(
                df_timeline,
                x_start="Recommended Removal Date",
                x_end="End Date",
                y="Bulb Type",
                color="Predicted Avg Temp (°F)",
                title="Timeline: Bulb Removal Dates & Predicted Temps",
                color_continuous_scale="RdBu_r"
            )
            fig_timeline.add_vline(x=easter_date, line_dash="dash", line_color="red", annotation_text="Easter")
            fig_timeline.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_timeline, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating timeline: {e}")

        # Historical Trends by Bulb Type
        st.subheader("📈 Trends Over Time by Bulb Type")
        selected_bulb = st.selectbox("Choose Bulb Type", sorted(df_all["Bulb Type"].dropna().unique()))
        df_trend = df_all[df_all["Bulb Type"] == selected_bulb].groupby("Year").agg({
            "Avg Temp (°F)": "mean", "Degree Hours >40°F": "mean"
        }).reset_index()

        fig_trend1 = px.line(df_trend, x="Year", y="Avg Temp (°F)", markers=True,
                             title=f"Avg Temp for {selected_bulb}")
        fig_trend2 = px.line(df_trend, x="Year", y="Degree Hours >40°F", markers=True,
                             title=f"Degree Hours >40°F for {selected_bulb}")

        st.plotly_chart(fig_trend1, use_container_width=True)
        st.plotly_chart(fig_trend2, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("👆 Please upload your Excel file to begin.")
