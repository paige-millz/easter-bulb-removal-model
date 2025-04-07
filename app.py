import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta

# ---------- Helper Functions ----------
def compute_easter(year):
    """Anonymous Gregorian algorithm to compute Easter."""
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

def load_year_data(xls, sheet, year):
    df = xls.parse(sheet, header=1)
    cols = [
        'Bulb/Tray Type',
        'Removal Date',
        'Removal DBE',
        'Average Temperature from Removal Date (°F)',
        'Growing Degree Days (#)'
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[cols]
    df['Year'] = year
    return df.rename(columns={
        'Bulb/Tray Type': 'Bulb Type',
        'Removal DBE': 'DBE',
        'Average Temperature from Removal Date (°F)': 'Avg Temp (°F)',
        'Growing Degree Days (#)': 'Degree Hours >40°F'
    })

# ---------- Streamlit App ----------
st.title("🌷 Easter Bulb Removal Forecasting Dashboard")

st.markdown("Upload your bulb removal tracking file to visualize trends and get future guidance based on Easter.")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

exclude_keywords = ["additional notes", "bonzi", "florel"]

if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        df_all = pd.concat([
            load_year_data(xls, "2023", 2023),
            load_year_data(xls, "2024", 2024),
            load_year_data(xls, "2025", 2025),
        ], ignore_index=True)

        df_all['Removal Date'] = pd.to_datetime(df_all['Removal Date'], errors='coerce')
        df_all_filtered = df_all[~df_all["Bulb Type"].str.contains("|".join(exclude_keywords), case=False, na=False)]

        # --- KPIs ---
        st.header("📊 Historical DBE KPIs")
        avg_dbe_all = df_all_filtered["DBE"].mean()
        years_count = df_all_filtered["Year"].nunique()
        st.markdown(f"**Years in Dataset:** {years_count}  |  **Average DBE:** {avg_dbe_all:.1f} days")

        avg_dbe_by_bulb = df_all_filtered.groupby("Bulb Type")["DBE"].mean().reset_index()
        fig_bar = px.bar(avg_dbe_by_bulb, x="Bulb Type", y="DBE", title="Avg DBE by Bulb Type")
        st.plotly_chart(fig_bar)

        # --- DBE vs Temp by Year ---
        st.header("📉 DBE vs. Temperature by Year")
        year_options = ["All"] + sorted(df_all["Year"].dropna().unique().astype(str).tolist())
        selected_year = st.selectbox("Select Year to Filter", year_options)
        df_vis = df_all if selected_year == "All" else df_all[df_all["Year"] == int(selected_year)]
        fig_scatter = px.scatter(df_vis, x="DBE", y="Avg Temp (°F)", color="Bulb Type", hover_data=["Year", "Removal Date"])
        st.plotly_chart(fig_scatter)

        # --- Regression Model ---
        st.header("📈 Regression: Predict Avg Temp from DBE")
        model_type = st.radio("Regression Type", ["Overall", "By Year"])
        if model_type == "By Year":
            reg_year = st.selectbox("Choose Year", sorted(df_all["Year"].dropna().unique()))
            df_model = df_all[df_all["Year"] == reg_year]
        else:
            df_model = df_all

        df_model = df_model.dropna(subset=["DBE", "Avg Temp (°F)"])
        df_model["DBE"] = pd.to_numeric(df_model["DBE"], errors="coerce")
        df_model["Avg Temp (°F)"] = pd.to_numeric(df_model["Avg Temp (°F)"], errors="coerce")
        df_model = df_model.dropna()

        model = LinearRegression()
        X = df_model["DBE"].values.reshape(-1, 1)
        y = df_model["Avg Temp (°F)"].values
        model.fit(X, y)
        intercept, slope = model.intercept_, model.coef_[0]
        st.markdown(f"**Intercept:** `{intercept:.2f}`  ,  **Slope:** `{slope:.2f}`")

        df_model["Predicted"] = model.predict(X)
        fig_fit = px.scatter(df_model, x="DBE", y="Avg Temp (°F)", color="Bulb Type", title="Regression Fit")
        fig_fit.add_scatter(x=df_model["DBE"], y=df_model["Predicted"], mode="lines", name="Regression Line")
        st.plotly_chart(fig_fit)

        # --- Forecasting Section ---
        st.header("📌 Recommended Removal Dates")
        future_year = st.number_input("Select Easter Year", value=2026, step=1)
        easter_dt = pd.to_datetime(compute_easter(future_year))
        st.markdown(f"**Easter Date:** {easter_dt.strftime('%Y-%m-%d')}")

        forecast = df_all_filtered.groupby("Bulb Type")["DBE"].mean().reset_index()
        forecast["DBE"] = pd.to_numeric(forecast["DBE"], errors="coerce")
        forecast = forecast.dropna(subset=["DBE"])

        forecast["Recommended Removal Date"] = forecast["DBE"].apply(lambda d: easter_dt - pd.Timedelta(days=int(d)))
        forecast["Predicted Avg Temp (°F)"] = forecast["DBE"].apply(lambda d: round(intercept + slope * d, 1))

        st.dataframe(forecast)

        # --- Visual Timeline ---
        st.header("📆 Visual Timeline of Removal Dates")
        try:
            fig_timeline = px.timeline(
                forecast.sort_values("Recommended Removal Date"),
                x_start="Recommended Removal Date",
                x_end="Recommended Removal Date",
                y="Bulb Type",
                color="Predicted Avg Temp (°F)",
                title="Removal Schedule by Bulb Type"
            )
            fig_timeline.add_vline(x=easter_dt, line_color="red", line_dash="dash", annotation_text="Easter")
            st.plotly_chart(fig_timeline)
        except Exception as e:
            st.error(f"Timeline Error: {e}")

        # --- Trend Section ---
        st.header("📈 Trends Over Time by Bulb Type")
        unique_bulbs = sorted([bt for bt in df_all_filtered["Bulb Type"].dropna().unique() if not any(k in bt.lower() for k in exclude_keywords)])
        selected_bulb = st.selectbox("Choose Bulb Type", unique_bulbs)
        df_bt = df_all_filtered[df_all_filtered["Bulb Type"] == selected_bulb]

        trend = df_bt.groupby("Year").agg({
            "Avg Temp (°F)": "mean",
            "Degree Hours >40°F": "mean"
        }).reset_index()

        fig_temp = px.line(trend, x="Year", y="Avg Temp (°F)", markers=True, title=f"Avg Temp for {selected_bulb}")
        fig_deg = px.line(trend, x="Year", y="Degree Hours >40°F", markers=True, title=f"Degree Hours >40°F for {selected_bulb}")

        st.plotly_chart(fig_temp)
        st.plotly_chart(fig_deg)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload your Excel file to get started.")
