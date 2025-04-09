import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# ---------------------------
# CONFIGURATION
# ---------------------------
st.set_page_config(page_title="OnFly Air Cargo Fit Tool", layout="centered")
st.image("OFA_Gold_Black.png", width=200)
st.title("OnFly Air Cargo Fit Tool")
st.markdown("Determine if cargo and mechanics fit into the selected aircraft based on dimensions and weight.")

# ---------------------------
# LOAD DATA
# ---------------------------
# Use the raw GitHub URLs for the CSV files
aircraft_url = "https://raw.githubusercontent.com/paige-millz/onfly-air-cargo-fit-tool/refs/heads/main/AC%20Specifications_Cargo%20Fit%20Tool%20App%20-%20AC%20Specs.csv"
parts_url = "https://raw.githubusercontent.com/paige-millz/onfly-air-cargo-fit-tool/refs/heads/main/historical_parts_Cargo%20Fit%20Tool%20%20-%20historical_parts.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data
def clean_aircraft(df):
    numeric = df.columns.drop("Aircraft", errors="ignore")
    for col in numeric:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("~", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df_aircraft = clean_aircraft(load_data(aircraft_url))
df_parts = load_data(parts_url)

# ---------------------------
# STEP 1: SELECT AIRCRAFT
# ---------------------------
st.header("Step 1: Select an Aircraft")
aircraft_options = df_aircraft["Aircraft"].dropna().unique().tolist()
selected_aircraft = st.selectbox("Choose Aircraft", aircraft_options, index=0)
aircraft_data = df_aircraft[df_aircraft["Aircraft"] == selected_aircraft].iloc[0]

st.subheader("Selected Aircraft Specs")
st.write(f"Door: {aircraft_data['Door Width (in)']}\" W x {aircraft_data['Door Height (in)']}\" H")
st.write(f"Cabin: {aircraft_data['Cabin Length (in)']}\" L x {aircraft_data['Cabin Width (in)']}\" W x {aircraft_data['Cabin Height (in)']}\" H")
st.write(f"Max Payload: {aircraft_data['Max Payload (lbs)']} lbs")

st.markdown("---")

# ---------------------------
# STEP 2: ENTER MULTIPLE PARTS
# ---------------------------
st.header("Step 2: Add Cargo (Part) Details")

if "parts_list" not in st.session_state:
    st.session_state.parts_list = []

with st.form("add_part_form"):
    part_names = df_parts["Part"].dropna().astype(str).unique().tolist()
    part_name = st.selectbox("Choose Existing Part or Enter New Name", ["(New)"] + sorted(part_names))

    part_aircraft = st.selectbox("Select Aircraft for This Part", aircraft_options)

    if part_name == "(New)":
        part_name = st.text_input("New Part Name", value="")
        length = st.number_input("Length (in)", min_value=0.0, step=0.1)
        width = st.number_input("Width (in)", min_value=0.0, step=0.1)
        height = st.number_input("Height (in)", min_value=0.0, step=0.1)
        weight = st.number_input("Weight (lbs)", min_value=0.0, step=0.1)
    else:
        match = df_parts[df_parts["Part"] == part_name]
        if not match.empty:
            record = match.iloc[0]
            part_name = record["Part"]
            length, width, height, weight = record[["Length (in)", "Width (in)", "Height (in)", "Weight (lbs.)"]]

    rotate = st.checkbox("Rotate part for door/cabin check? (swap L/W)")
    if rotate:
        length, width = width, length

    submitted = st.form_submit_button("Add Part")
    if submitted:
        new_part = {
            "Name": part_name,
            "Aircraft": part_aircraft,
            "Length": length,
            "Width": width,
            "Height": height,
            "Weight": weight
        }
        st.session_state.parts_list.append(new_part)

if st.session_state.parts_list:
    st.subheader("Selected Parts")
    parts_df = pd.DataFrame(st.session_state.parts_list)
    st.table(parts_df)

    csv_buffer = io.StringIO()
    parts_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Parts as CSV",
        data=csv_buffer.getvalue(),
        file_name="onfly_parts_list.csv",
        mime="text/csv"
    )

st.markdown("---")

# STEP 3: Mechanics and Payload
st.header("Step 3: Mechanics and Payload")
mechanics = st.checkbox("Include mechanics in payload?")
mech_count = st.number_input("Number of Mechanics", min_value=0, value=0) if mechanics else 0
mech_weight = 180
tool_weight = st.number_input("Total Tool Weight (lbs)", min_value=0.0) if mechanics else 0.0
seat_removed = st.checkbox("Remove a seat?")
seat_weight = aircraft_data["Seat Weight (lbs)"] if seat_removed else 0

# STEP 4: Feasibility Calculations
st.header("Step 4: Feasibility Check")
total_part_weight = sum(p["Weight"] for p in st.session_state.parts_list)
total_weight = total_part_weight + mech_count * mech_weight + tool_weight
available_payload = aircraft_data["Max Payload (lbs)"] + seat_weight

st.write(f"**Total Required Payload:** {total_weight:.1f} lbs")
st.write(f"**Available Payload:** {available_payload:.1f} lbs")
if total_weight <= available_payload:
    st.success("✅ Payload check passed")
else:
    st.error("❌ Over max payload")

# STEP 5: Door & Cabin Fit Checks
st.header("Step 5: Door & Cabin Fit Checks")
door_w, door_h = aircraft_data['Door Width (in)'], aircraft_data['Door Height (in)']
cabin_l, cabin_w, cabin_h = aircraft_data['Cabin Length (in)'], aircraft_data['Cabin Width (in)'], aircraft_data['Cabin Height (in)']

for part in st.session_state.parts_list:
    fits_door = (part["Length"] <= door_w and part["Width"] <= door_h) or (part["Width"] <= door_w and part["Length"] <= door_h)
    fits_cabin = part["Length"] <= cabin_l and part["Width"] <= cabin_w and part["Height"] <= cabin_h
    st.write(f"**{part['Name']}**")
    st.success("✅ Fits through door") if fits_door else st.error("❌ Too big for door")
    st.success("✅ Fits in cabin") if fits_cabin else st.error("❌ Too big for cabin")

# STEP 6: Visualization
st.header("Step 6: Visualization")
if st.session_state.parts_list:
    selected_visual = st.selectbox("Select part to visualize against door", [p["Name"] for p in st.session_state.parts_list])
    selected_data = next(p for p in st.session_state.parts_list if p["Name"] == selected_visual)

    fig, ax = plt.subplots(figsize=(6, 6))
    door_rect = plt.Rectangle((0, 0), door_w, door_h, edgecolor='blue', fill=False, lw=2, label='Door')
    cargo_rect = plt.Rectangle((0, 0), selected_data["Length"], selected_data["Width"],
                               edgecolor='green' if (selected_data["Length"] <= door_w and selected_data["Width"] <= door_h) else 'red',
                               fill=False, lw=2, label='Cargo')
    ax.add_patch(door_rect)
    ax.add_patch(cargo_rect)
    ax.set_xlim(0, max(door_w, selected_data["Length"]) + 10)
    ax.set_ylim(0, max(door_h, selected_data["Width"]) + 10)
    ax.set_aspect('equal')
    ax.set_xlabel("inches")
    ax.set_ylabel("inches")
    ax.set_title(f"{selected_visual} vs Door")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Full Cargo Layout in Cabin")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.set_xlim(0, cabin_l)
    ax2.set_ylim(0, cabin_w)
    ax2.set_title("Cabin Cargo Layout")
    ax2.set_xlabel("Cabin Length (in)")
    ax2.set_ylabel("Cabin Width (in)")
    x_offset = 0
    y_offset = 0
    spacing = 5

    for part in st.session_state.parts_list:
        if x_offset + part["Length"] > cabin_l:
            x_offset = 0
            y_offset += part["Width"] + spacing
        if y_offset + part["Width"] > cabin_w:
            ax2.text(5, cabin_w - 10, "Not all parts fit in cabin!", fontsize=12, color='red')
            break
        rect = plt.Rectangle((x_offset, y_offset), part["Length"], part["Width"],
                             edgecolor='green', facecolor='lightgreen', lw=2)
        ax2.add_patch(rect)
        ax2.text(x_offset + 2, y_offset + 2, part["Name"], fontsize=8)
        x_offset += part["Length"] + spacing

    ax2.set_aspect('equal')
    st.pyplot(fig2)
