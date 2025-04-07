import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import date

st.set_page_config(page_title="Easter Bulb Removal Model", layout="wide")
st.title("🌱 Easter Bulb Removal Model Dashboard")

st.markdown("""
Upload your **Easter Rules Template_Bulb Removal Model(PM).xlsx** file containing historical data for 2023–2025.  
This dashboard:
- Combines the data
- Analyzes average removal timing (DBE)
- Predicts future removal dates based on Easter
- Estimates temperature using regression models
""")

uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

# Easter algorithm
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

# Excel sheet loader
def load_year_data(xls, sheet_name, year):
    df = xls.parse(sheet_name, header=1)
    needed = [
        'Bulb/Tray Type', 'Removal Date', 'Removal DBE',
        'Average Temperature from Removal Date (°F)', 'Growing Degree Days (#)'
    ]
    for col in needed:
        if col not in df:
            df[col] = 0 if "Degree" in col else pd.NA
    df = df[needed]
    df["Year"] = year
    df = df.rename(columns={
        'Bulb/Tray Type': 'Bulb Type',
        'Removal DBE': 'DBE',
        'Average Temperature from Removal Date (°F)': 'Avg Temp (°F)',
        'Growing Degree Days (#)': 'Degree Hours >40°F'
    })
    return df

exclude_keywords = ["additional notes", "florel", "bonzi"]

if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        df = pd.concat([
            load_year_data(xls, '2023', 2023),
            load_year_data(xls, '2024', 2024),
            load_year_data(xls, '2025', 2025)
        ], ignore_index=True)

        df['Removal Date'] = pd.to_datetime(df['Removal Date'], errors='coerce')
        df_filtered = df[~df["Bulb Type"].str.contains("|".join(exclude_keywords), case=False, na=False)]

        # KPIs
        st.subheader("📊 KPIs & Avg DBE")
        st.markdown(f"- **Years included**: {df_filtered['Year'].nunique()}")
        st.markdown(f"- **Overall Avg DBE**: `{df_filtered['DBE'].mean():.1f}` days before Easter")
        fig1 = px.bar(df_filtered.groupby('Bulb Type')['DBE'].mean().reset_index(),
                      x="Bulb Type", y="DBE", title="Avg DBE by Bulb Type")
        st.plotly_chart(fig1)

        # DBE vs Temp
        st.subheader("📈 DBE vs. Avg Temperature")
        years = sorted(df['Year'].dropna().unique().astype(int))
        selected_year = st.selectbox("Select Year to Filter", ["All"] + [str(y) for y in years])
        df_plot = df if selected_year == "All" else df[df["Year"] == int(selected_year)]
        fig2 = px.scatter(df_plot, x="DBE", y="Avg Temp (°F)", color="Bulb Type",
                          hover_data=["Year", "Removal Date"],
                          title="DBE vs. Temperature")
        st.plotly_chart(fig2)

        # Regression model
        st.subheader("📉 Regression: Predict Avg Temp from DBE")
        model_choice = st.selectbox("Regression Type", ["Overall", "By Year"])
        model = None

        if model_choice == "Overall":
            df_model = df_filtered.dropna(subset=['DBE', 'Avg Temp (°F)']).copy()
        else:
            reg_year = st.selectbox("Choose Year", years)
            df_model = df_filtered[df_filtered["Year"] == reg_year].dropna(subset=['DBE', 'Avg Temp (°F)']).copy()

        if not df_model.empty:
            X = df_model["DBE"].astype(float).values.reshape(-1, 1)
            y = df_model["Avg Temp (°F)"].astype(float).values
            model = LinearRegression().fit(X, y)
            df_model["Predicted"] = model.predict(X)
            st.markdown(f"**Intercept**: `{model.intercept_:.2f}`, **Slope**: `{model.coef_[0]:.2f}`")
            fig3 = px.scatter(df_model, x="DBE", y="Avg Temp (°F)", color="Bulb Type", title="Regression Fit")
            fig3.add_scatter(x=df_model["DBE"], y=df_model["Predicted"], mode='lines', name="Regression Line")
            st.plotly_chart(fig3)

        # Recommended removal
        st.subheader("📌 Recommended Removal Dates")
        easter_year = st.number_input("Select Easter Year", value=2024)
        easter_date = pd.to_datetime(compute_easter(int(easter_year)))
        st.write(f"**Easter Date:** {easter_date.date()}")

        avg_dbe = df_filtered.groupby("Bulb Type")["DBE"].mean().reset_index()
        avg_dbe["Recommended Removal Date"] = avg_dbe["DBE"].apply(lambda d: easter_date - pd.to_timedelta(round(d), unit="D"))
        avg_dbe["Predicted Avg Temp (°F)"] = avg_dbe["DBE"].apply(lambda d: model.intercept_ + model.coef_[0]*d if model else pd.NA)
        st.dataframe(avg_dbe)

        # Timeline View
        st.subheader("📆 Visual Timeline of Pull Dates")
        timeline_df = avg_dbe[["Bulb Type", "Recommended Removal Date"]].dropna().copy()
        timeline_df["Start"] = timeline_df["Recommended Removal Date"]
        timeline_df["End"] = timeline_df["Recommended Removal Date"] + pd.Timedelta(days=1)
        fig_timeline = px.timeline(timeline_df, x_start="Start", x_end="End", y="Bulb Type", color="Bulb Type",
                                   title="Visual Timeline of Removal Dates")
        fig_timeline.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_timeline)

        # Trends by Bulb
        st.subheader("📈 Trends by Bulb Type")
        bulb_options = sorted([b for b in df["Bulb Type"].dropna().unique() if not any(k in b.lower() for k in exclude_keywords)])
        selected_bulb = st.selectbox("Choose Bulb", options=bulb_options)
        df_bulb = df[df["Bulb Type"] == selected_bulb]
        trend_df = df_bulb.groupby("Year")[["Avg Temp (°F)", "Degree Hours >40°F"]].mean().reset_index()
        st.plotly_chart(px.line(trend_df, x="Year", y="Avg Temp (°F)", markers=True, title="Avg Temp Over Years"))
        st.plotly_chart(px.line(trend_df, x="Year", y="Degree Hours >40°F", markers=True, title="Degree Hours >40°F Over Years"))

    except Exception as e:
        st.error(f"🚨 Error: {e}")
else:
    st.info("Upload an Excel file to begin.")
