# 7) Visual Timeline (Gantt-Style) of Recommended Removal Dates
st.subheader("Visual Timeline of Recommended Removal Dates")

# Build a mini-dataframe from avg_dbe
timeline_df = avg_dbe[["Bulb Type", "Recommended Removal Date"]].dropna().copy()

# We'll create a 'Start' and 'End' so that px.timeline can draw a small bar for each Bulb Type.
# Here, we make the bar length = 1 day. Feel free to adjust to 0.5 days or 2 days, etc.
timeline_df["Start"] = timeline_df["Recommended Removal Date"]
timeline_df["End"] = timeline_df["Recommended Removal Date"] + pd.Timedelta(days=1)

# Create the Gantt-style timeline
fig_timeline = px.timeline(
    timeline_df,
    x_start="Start",
    x_end="End",
    y="Bulb Type",
    color="Bulb Type",
    title="Recommended Removal Dates (Gantt-Style)"
)

# Reverse the Y-axis so the first Bulb Type appears at the top
fig_timeline.update_yaxes(autorange="reversed")

# Add a vertical line to indicate Easter
fig_timeline.add_vline(
    x=easter_date,
    line_width=3,
    line_dash="dash",
    line_color="red",
    annotation_text="Easter",
    annotation_position="top right"
)

st.plotly_chart(fig_timeline)
