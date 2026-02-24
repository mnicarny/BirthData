# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("Provisional Natality Data Dashboard")
st.subheader("Birth Analysis by State and Gender")


def _normalize_colname(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_")


def _keyify(s: str) -> str:
    # Remove non-alphanumeric for robust matching
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _match_required_fields(columns_norm):
    """
    Dynamically match required logical fields to normalized dataframe columns.
    Returns (mapping, missing_fields)
    """
    required_logical = [
        "state_of_residence",
        "month",
        "month_code",
        "year_code",
        "sex_of_infant",
        "births",
    ]

    cols = list(columns_norm)
    cols_keyed = {_keyify(c): c for c in cols}

    mapping = {}
    missing = []

    for logical in required_logical:
        # 1) Exact match
        if logical in cols:
            mapping[logical] = logical
            continue

        # 2) Keyified match (handles underscores, punctuation differences)
        lk = _keyify(logical)
        if lk in cols_keyed:
            mapping[logical] = cols_keyed[lk]
            continue

        # 3) Fallback: find any column whose key equals logical key (in case of collisions)
        candidates = [c for c in cols if _keyify(c) == lk]
        if len(candidates) == 1:
            mapping[logical] = candidates[0]
            continue

        missing.append(logical)

    return mapping, missing


def _with_all_option(options):
    opts = [o for o in options if pd.notna(o)]
    # Keep original types where possible; Streamlit handles mixed types but sorting may fail
    try:
        opts_sorted = sorted(opts)
    except Exception:
        opts_sorted = sorted([str(o) for o in opts])
    return ["All"] + opts_sorted


@st.cache_data(show_spinner=False)
def load_data():
    try:
        df_raw = pd.read_csv("Provisional_Natality_2025_CDC.csv")
    except FileNotFoundError:
        return None, "not_found", None, None

    # Preserve original column names for debugging
    original_columns = list(df_raw.columns)

    # Normalize column names
    df = df_raw.copy()
    df.columns = [_normalize_colname(c) for c in df.columns]

    # Validate required fields
    mapping, missing = _match_required_fields(df.columns)

    if missing:
        return df, "missing_fields", missing, original_columns

    # Convert births to numeric and drop null births
    births_col = mapping["births"]
    df[births_col] = pd.to_numeric(df[births_col], errors="coerce")
    df = df.dropna(subset=[births_col])

    return df, "ok", mapping, original_columns


df, status, aux, original_columns = load_data()

if status == "not_found":
    st.error("Dataset file not found in repository.")
    st.stop()

if status == "missing_fields":
    missing_fields = aux
    st.error(
        "Required logical fields are missing: "
        + ", ".join(missing_fields)
        + ". Please verify the dataset columns."
    )
    st.write("Original columns:")
    st.write(original_columns)
    st.write("Normalized columns:")
    st.write(list(df.columns) if df is not None else [])
    st.stop()

mapping = aux

# Extract matched columns
state_col = mapping["state_of_residence"]
month_col = mapping["month"]
sex_col = mapping["sex_of_infant"]
births_col = mapping["births"]

# Sidebar filters (multiselect only, include All, default All)
st.sidebar.header("Filters")

state_options = _with_all_option(df[state_col].dropna().unique().tolist())
month_options = _with_all_option(df[month_col].dropna().unique().tolist())
sex_options = _with_all_option(df[sex_col].dropna().unique().tolist())

selected_states = st.sidebar.multiselect(
    "State of Residence", options=state_options, default=["All"]
)
selected_months = st.sidebar.multiselect(
    "Month", options=month_options, default=["All"]
)
selected_sex = st.sidebar.multiselect(
    "Gender", options=sex_options, default=["All"]
)

# Filtering logic (do not modify original df)
filtered = df.copy()

if selected_states and "All" not in selected_states:
    filtered = filtered[filtered[state_col].isin(selected_states)]

if selected_months and "All" not in selected_months:
    filtered = filtered[filtered[month_col].isin(selected_months)]

if selected_sex and "All" not in selected_sex:
    filtered = filtered[filtered[sex_col].isin(selected_sex)]

if filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# Aggregation: group by state_of_residence and sex_of_infant, sum births, sort states alphabetically
agg = (
    filtered.groupby([state_col, sex_col], as_index=False)[births_col]
    .sum()
    .sort_values(by=[state_col, sex_col], ascending=[True, True])
)

# Plot
fig = px.bar(
    agg,
    x=state_col,
    y=births_col,
    color=sex_col,
    title="Total Births by State and Gender",
)
fig.update_layout(
    template="plotly_white",
    legend_title_text="Gender",
    xaxis_title="State of Residence",
    yaxis_title="Births",
    margin=dict(l=10, r=10, t=60, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# Show filtered table below chart (clean display, no index)
display_df = filtered.reset_index(drop=True)

# Prefer hiding index if supported; fall back without it.
try:
    st.dataframe(display_df, hide_index=True, use_container_width=True)
except TypeError:
    st.dataframe(display_df, use_container_width=True)
