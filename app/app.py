import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="APL Logistics | Delivery Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        letter-spacing: -0.3px;
    }

    .info-box {
        padding: 14px 18px;
        border-radius: 8px;
        background: #172d45;
        border-left: 4px solid #4da3ff;
        margin-bottom: 15px;
    }

    .warning-box {
        padding: 12px 16px;
        border-radius: 8px;
        background: #3a3714;
        border-left: 4px solid #f2c94c;
        margin: 10px 0;
    }

    .success-box {
        padding: 12px 16px;
        border-radius: 8px;
        background: #123a28;
        border-left: 4px solid #4ade80;
        margin: 10px 0;
    }

    .danger-box {
        padding: 12px 16px;
        border-radius: 8px;
        background: #401b22;
        border-left: 4px solid #ff5c6c;
        margin: 10px 0;
    }

    .small-note {
        color: #aab2bf;
        font-size: 0.85rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.65rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

REQUIRED_COLUMNS = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk",
    "Delivery Status",
    "Shipping Mode",
    "Market",
    "Order Region",
    "Customer Segment"
]


# ============================================================
# DATA LOADING
# ============================================================

def find_csv_files():
    """
    Search common project folders for CSV files.
    """

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    possible_dirs = [
        project_root,
        project_root / "data",
        project_root / "dataset",
        project_root / "datasets",
        current_file.parent,
    ]

    files = []

    for folder in possible_dirs:
        if folder.exists():
            for file in folder.glob("*.csv"):
                if file.is_file():
                    files.append(file)

    # Remove duplicates
    unique_files = []

    for file in files:
        if file not in unique_files:
            unique_files.append(file)

    return unique_files


@st.cache_data(show_spinner=False)
def load_data_from_path(path):

    """
    Robust CSV loader.

    Handles:
    UTF-8
    UTF-8 BOM
    Windows-1252
    Latin-1

    This fixes the UnicodeDecodeError caused by non-UTF-8 CSV files.
    """

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1"
    ]

    last_error = None

    for encoding in encodings:

        try:

            df = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False
            )

            return df

        except UnicodeDecodeError as error:

            last_error = error

    raise ValueError(
        f"Unable to decode CSV file. Last error: {last_error}"
    )


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(df):

    df = df.copy()

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove duplicate records
    df = df.drop_duplicates()

    # Strip spaces from column names
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # Convert important numeric fields
    numeric_columns = [
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Benefit per order",
        "Sales per customer",
        "Late_delivery_risk",
        "Order Item Discount",
        "Order Item Discount Rate",
        "Order Item Product Price",
        "Order Item Profit Ratio",
        "Order Item Quantity",
        "Sales",
        "Order Item Total",
        "Order Profit Per Order",
        "Product Price",
        "Latitude",
        "Longitude"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Strip spaces from text fields
    text_columns = [
        "Delivery Status",
        "Shipping Mode",
        "Market",
        "Order Region",
        "Order Country",
        "Order State",
        "Customer Segment",
        "Category Name",
        "Department Name",
        "Product Name"
    ]

    for col in text_columns:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
            )

    # --------------------------------------------------------
    # Delivery delay gap
    # --------------------------------------------------------

    if (
        "Days for shipping (real)" in df.columns
        and
        "Days for shipment (scheduled)" in df.columns
    ):

        df["Delay Gap"] = (
            df["Days for shipping (real)"]
            -
            df["Days for shipment (scheduled)"]
        )

        df["Delay Gap"] = pd.to_numeric(
            df["Delay Gap"],
            errors="coerce"
        )

        # Delivery classification
        df["Delivery Classification"] = np.select(
            [
                df["Delay Gap"] > 0,
                df["Delay Gap"] == 0,
                df["Delay Gap"] < 0
            ],
            [
                "Delayed",
                "On-time",
                "Early"
            ],
            default="Unknown"
        )

    # --------------------------------------------------------
    # Normalize late risk
    # --------------------------------------------------------

    if "Late_delivery_risk" in df.columns:

        df["Late_delivery_risk"] = (
            pd.to_numeric(
                df["Late_delivery_risk"],
                errors="coerce"
            )
            .fillna(0)
        )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df):

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    return missing


# ============================================================
# KPI FUNCTIONS
# ============================================================

def calculate_kpis(df):

    total = len(df)

    if total == 0:

        return {
            "total": 0,
            "on_time": 0,
            "on_time_rate": 0,
            "delayed": 0,
            "delay_rate": 0,
            "avg_delay": 0,
            "risk_shipments": 0,
            "risk_rate": 0
        }

    delay_gap = df["Delay Gap"]

    on_time = (delay_gap <= 0).sum()

    delayed = (delay_gap > 0).sum()

    # Average positive delay
    positive_delays = delay_gap[delay_gap > 0]

    avg_delay = (
        positive_delays.mean()
        if len(positive_delays) > 0
        else 0
    )

    risk_shipments = (
        df["Late_delivery_risk"] == 1
    ).sum()

    return {
        "total": total,
        "on_time": int(on_time),
        "on_time_rate": on_time / total * 100,
        "delayed": int(delayed),
        "delay_rate": delayed / total * 100,
        "avg_delay": float(avg_delay),
        "risk_shipments": int(risk_shipments),
        "risk_rate": risk_shipments / total * 100
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rate_table(
    df,
    group_col,
    min_count=1
):

    if group_col not in df.columns:
        return pd.DataFrame()

    result = (
        df.groupby(group_col, dropna=False)
        .agg(
            Shipments=("Delay Gap", "size"),
            Delayed=("Delay Gap", lambda x: (x > 0).sum()),
            Avg_Delay=("Delay Gap", lambda x: x[x > 0].mean()),
            High_Risk=("Late_delivery_risk", lambda x: (x == 1).sum())
        )
        .reset_index()
    )

    result = result[
        result["Shipments"] >= min_count
    ]

    result["Delay Rate (%)"] = (
        result["Delayed"]
        /
        result["Shipments"]
        * 100
    )

    result["High-Risk Rate (%)"] = (
        result["High_Risk"]
        /
        result["Shipments"]
        * 100
    )

    result["SLA / On-Time Rate (%)"] = (
        100
        -
        result["Delay Rate (%)"]
    )

    result["Avg_Delay"] = result["Avg_Delay"].fillna(0)

    return result.sort_values(
        "Delay Rate (%)",
        ascending=False
    )


def format_percentage(value):

    return f"{value:.2f}%"


def safe_value(value):

    if pd.isna(value):
        return 0

    return value


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔎 Dashboard Controls")

st.sidebar.caption(
    "Use the filters to explore delivery performance, "
    "delay risk and logistics efficiency."
)


# ============================================================
# FIND DATA
# ============================================================

csv_files = find_csv_files()

if not csv_files:

    st.error(
        "No CSV file was found in the project folders."
    )

    st.info(
        "Place your logistics CSV inside the project "
        "folder or a data/dataset folder."
    )

    st.stop()


# If multiple CSVs exist
if len(csv_files) == 1:

    csv_path = csv_files[0]

else:

    selected_file = st.sidebar.selectbox(
        "Select Dataset",
        csv_files,
        format_func=lambda x: x.name
    )

    csv_path = selected_file


# ============================================================
# LOAD + CLEAN
# ============================================================

try:

    raw_df = load_data_from_path(
        str(csv_path)
    )

except Exception as error:

    st.error(
        f"Unable to load the dataset: {error}"
    )

    st.stop()


df = clean_data(raw_df)


# ============================================================
# VALIDATE
# ============================================================

missing_columns = validate_dataset(df)

if missing_columns:

    st.error(
        "The dataset is missing required columns:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# DATE COLUMN DETECTION
# ============================================================

possible_date_columns = [
    "Order Date",
    "order date",
    "Order_Date",
    "Date",
    "date",
    "Shipping Date",
    "Delivery Date"
]

date_column = None

for col in possible_date_columns:

    if col in df.columns:

        date_column = col
        break


# ============================================================
# SIDEBAR FILTERS
# ============================================================

# ------------------------------------------------------------
# Shipping Mode
# ------------------------------------------------------------

shipping_modes = sorted(
    df["Shipping Mode"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_shipping = st.sidebar.multiselect(
    "Shipping Mode",
    shipping_modes,
    default=shipping_modes
)


# ------------------------------------------------------------
# Market
# ------------------------------------------------------------

markets = sorted(
    df["Market"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_market = st.sidebar.multiselect(
    "Market",
    markets,
    default=markets
)


# ------------------------------------------------------------
# Region
# ------------------------------------------------------------

regions = sorted(
    df["Order Region"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_region = st.sidebar.multiselect(
    "Order Region",
    regions,
    default=regions
)


# ------------------------------------------------------------
# Customer Segment
# ------------------------------------------------------------

segments = sorted(
    df["Customer Segment"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_segment = st.sidebar.multiselect(
    "Customer Segment",
    segments,
    default=segments
)


# ============================================================
# DATE FILTER
# ============================================================

if date_column:

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    valid_dates = df[date_column].dropna()

    if len(valid_dates) > 0:

        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

else:

    st.sidebar.caption(
        "Date range filter unavailable: "
        "the provided dataset does not contain a date field."
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


if selected_shipping:

    filtered_df = filtered_df[
        filtered_df["Shipping Mode"].isin(
            selected_shipping
        )
    ]


if selected_market:

    filtered_df = filtered_df[
        filtered_df["Market"].isin(
            selected_market
        )
    ]


if selected_region:

    filtered_df = filtered_df[
        filtered_df["Order Region"].isin(
            selected_region
        )
    ]


if selected_segment:

    filtered_df = filtered_df[
        filtered_df["Customer Segment"].isin(
            selected_segment
        )
    ]


if (
    date_column
    and
    "date_range" in locals()
    and
    len(date_range) == 2
):

    start_date = pd.Timestamp(
        date_range[0]
    )

    end_date = pd.Timestamp(
        date_range[1]
    ) + pd.Timedelta(days=1)

    filtered_df = filtered_df[
        (filtered_df[date_column] >= start_date)
        &
        (filtered_df[date_column] < end_date)
    ]


# ============================================================
# HANDLE EMPTY FILTER RESULT
# ============================================================

if filtered_df.empty:

    st.warning(
        "No shipments match the selected filters. "
        "Please broaden the filters."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🚚 Delivery Performance, Delay Risk & Logistics Efficiency"
)

st.caption(
    "APL Logistics | Global Supply Chain Operations"
)


st.markdown(
    f"""
    <div class="info-box">
    <b>Current analysis:</b>
    {len(filtered_df):,} shipments after applying the selected filters.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📊 Executive Delivery Overview")

kpis = calculate_kpis(
    filtered_df
)


c1, c2, c3, c4, c5, c6 = st.columns(6)


with c1:

    st.metric(
        "Total Shipments",
        f"{kpis['total']:,}"
    )


with c2:

    st.metric(
        "On-Time Rate",
        f"{kpis['on_time_rate']:.2f}%"
    )


with c3:

    st.metric(
        "Delayed Shipments",
        f"{kpis['delayed']:,}"
    )


with c4:

    st.metric(
        "Delay Rate",
        f"{kpis['delay_rate']:.2f}%"
    )


with c5:

    st.metric(
        "Avg Delay",
        f"{kpis['avg_delay']:.2f} days"
    )


with c6:

    st.metric(
        "High-Risk Shipments",
        f"{kpis['risk_shipments']:,}",
        f"{kpis['risk_rate']:.2f}%"
    )


# ============================================================
# DATA QUALITY / VALIDATION
# ============================================================

with st.expander(
    "🔍 Data & Dashboard Validation",
    expanded=False
):

    original_rows = len(raw_df)
    cleaned_rows = len(df)
    filtered_rows = len(filtered_df)

    missing_values = int(
        filtered_df.isna().sum().sum()
    )

    duplicate_count = int(
        raw_df.duplicated().sum()
    )

    v1, v2, v3, v4 = st.columns(4)

    with v1:
        st.metric(
            "Original Records",
            f"{original_rows:,}"
        )

    with v2:
        st.metric(
            "After Cleaning",
            f"{cleaned_rows:,}"
        )

    with v3:
        st.metric(
            "Filtered Records",
            f"{filtered_rows:,}"
        )

    with v4:
        st.metric(
            "Missing Cells",
            f"{missing_values:,}"
        )

    st.write(
        f"Duplicate records removed during cleaning: "
        f"**{duplicate_count:,}**"
    )

    st.success(
        "Delivery classification calculated using: "
        "Delay Gap = Actual Shipping Days − Scheduled Shipping Days."
    )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "📈 Overview",
        "⚠️ Delay Risk",
        "🚢 Shipping Modes",
        "🌍 Regional & Market",
        "👥 Customer & Product",
        "🎯 Recommendations"
    ]
)


# ============================================================
# TAB 1 — DELIVERY PERFORMANCE OVERVIEW
# ============================================================

with tabs[0]:

    st.header(
        "Delivery Performance Overview"
    )

    st.markdown(
        """
        This section establishes the baseline delivery performance
        using on-time, delayed and early shipments.
        """
    )

    # --------------------------------------------------------
    # Delivery classification
    # --------------------------------------------------------

    classification = (
        filtered_df["Delivery Classification"]
        .value_counts()
        .reindex(
            ["On-time", "Delayed", "Early"],
            fill_value=0
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Delivery Status Distribution"
        )

        st.bar_chart(
            classification
        )

    with col2:

        st.subheader(
            "Delivery Classification"
        )

        classification_df = (
            classification
            .rename("Shipments")
            .reset_index()
        )

        classification_df.columns = [
            "Classification",
            "Shipments"
        ]

        classification_df["Percentage (%)"] = (
            classification_df["Shipments"]
            /
            classification_df["Shipments"].sum()
            * 100
        )

        st.dataframe(
            classification_df.style.format(
                {
                    "Percentage (%)": "{:.2f}%"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Delay gap
    # --------------------------------------------------------

    st.subheader(
        "Delivery Delay Gap Analysis"
    )

    gap_counts = (
        filtered_df["Delay Gap"]
        .round(0)
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        gap_counts
    )

    st.caption(
        "Negative = early delivery | 0 = on-time | "
        "Positive = delayed delivery."
    )

    # --------------------------------------------------------
    # Delivery status
    # --------------------------------------------------------

    if "Delivery Status" in filtered_df.columns:

        st.subheader(
            "Operational Delivery Status"
        )

        status_table = (
            filtered_df["Delivery Status"]
            .value_counts()
            .rename_axis("Delivery Status")
            .reset_index(name="Shipments")
        )

        status_table["Share (%)"] = (
            status_table["Shipments"]
            /
            status_table["Shipments"].sum()
            * 100
        )

        st.dataframe(
            status_table.style.format(
                {
                    "Share (%)": "{:.2f}%"
                }
            ),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 2 — DELAY RISK ANALYSIS
# ============================================================

with tabs[1]:

    st.header(
        "Delay Risk Analysis"
    )

    st.markdown(
        """
        This section identifies the concentration of late-delivery
        risk and examines whether operational delay aligns with the
        recorded `Late_delivery_risk` indicator.
        """
    )

    risk_distribution = (
        filtered_df["Late_delivery_risk"]
        .map({
            0: "Low / No Risk",
            1: "Late Delivery Risk"
        })
        .fillna("Unknown")
        .value_counts()
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Late Delivery Risk Distribution"
        )

        st.bar_chart(
            risk_distribution
        )

    with col2:

        risk_table = (
            filtered_df["Late_delivery_risk"]
            .value_counts()
            .rename_axis("Risk Flag")
            .reset_index(name="Shipments")
        )

        risk_table["Percentage (%)"] = (
            risk_table["Shipments"]
            /
            risk_table["Shipments"].sum()
            * 100
        )

        risk_table["Risk Meaning"] = (
            risk_table["Risk Flag"]
            .map({
                0: "No late-delivery risk",
                1: "Late-delivery risk"
            })
        )

        st.dataframe(
            risk_table.style.format(
                {
                    "Percentage (%)": "{:.2f}%"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Delay gap distribution
    # --------------------------------------------------------

    st.subheader(
        "Delay Gap Distribution"
    )

    histogram_data = (
        filtered_df["Delay Gap"]
        .round(0)
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        histogram_data
    )

    # --------------------------------------------------------
    # Risk vs actual delay
    # --------------------------------------------------------

    st.subheader(
        "Recorded Risk vs Actual Delivery Outcome"
    )

    risk_outcome = pd.crosstab(
        filtered_df["Late_delivery_risk"],
        filtered_df["Delivery Classification"]
    )

    st.dataframe(
        risk_outcome,
        use_container_width=True
    )

    st.caption(
        "Use this comparison to assess how the recorded risk indicator "
        "corresponds with actual delivery outcomes."
    )


# ============================================================
# TAB 3 — SHIPPING MODE COMPARISON
# ============================================================

with tabs[2]:

    st.header(
        "Shipping Mode Comparison"
    )

    st.markdown(
        """
        Compare shipping modes using shipment volume, delay rate,
        average delay, high-risk rate and SLA/on-time compliance.
        """
    )

    mode_table = rate_table(
        filtered_df,
        "Shipping Mode"
    )

    if not mode_table.empty:

        st.subheader(
            "Mode-wise Delay Performance"
        )

        chart_data = (
            mode_table
            .set_index("Shipping Mode")[
                ["Delay Rate (%)"]
            ]
        )

        st.bar_chart(
            chart_data
        )

        st.subheader(
            "Shipping Mode Efficiency Table"
        )

        display_mode = mode_table[
            [
                "Shipping Mode",
                "Shipments",
                "Delayed",
                "Delay Rate (%)",
                "High_Risk",
                "High-Risk Rate (%)",
                "SLA / On-Time Rate (%)",
                "Avg_Delay"
            ]
        ].copy()

        display_mode = display_mode.rename(
            columns={
                "High_Risk": "High-Risk Shipments",
                "Avg_Delay": "Avg Delay (Days)"
            }
        )

        st.dataframe(
            display_mode.style.format(
                {
                    "Delay Rate (%)": "{:.2f}%",
                    "High-Risk Rate (%)": "{:.2f}%",
                    "SLA / On-Time Rate (%)": "{:.2f}%",
                    "Avg Delay (Days)": "{:.2f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # SLA compliance
    # --------------------------------------------------------

    st.subheader(
        "SLA / On-Time Compliance by Shipping Mode"
    )

    if not mode_table.empty:

        sla_data = (
            mode_table
            .set_index("Shipping Mode")[
                ["SLA / On-Time Rate (%)"]
            ]
        )

        st.bar_chart(
            sla_data
        )

    st.info(
        "Shipping Mode Efficiency Index is interpreted here through "
        "delay rate and SLA/on-time performance rather than assigning "
        "an arbitrary score."
    )


# ============================================================
# TAB 4 — REGIONAL & MARKET LOGISTICS ANALYSIS
# ============================================================

with tabs[3]:

    st.header(
        "Regional & Market Logistics Analysis"
    )

    st.markdown(
        """
        This module satisfies the requirement for Regional & Market
        Heatmaps by combining geographic delay visualization,
        market-wise efficiency and region-level delay diagnostics.
        """
    )

    # --------------------------------------------------------
    # Regional delay
    # --------------------------------------------------------

    region_table = rate_table(
        filtered_df,
        "Order Region"
    )

    market_table = rate_table(
        filtered_df,
        "Market"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Regional Delay Rate"
        )

        if not region_table.empty:

            regional_chart = (
                region_table
                .head(20)
                .set_index("Order Region")[
                    ["Delay Rate (%)"]
                ]
            )

            st.bar_chart(
                regional_chart
            )

    with col2:

        st.subheader(
            "Market-wise Logistics Efficiency"
        )

        if not market_table.empty:

            market_chart = (
                market_table
                .set_index("Market")[
                    ["Delay Rate (%)"]
                ]
            )

            st.bar_chart(
                market_chart
            )

    # --------------------------------------------------------
    # Regional table
    # --------------------------------------------------------

    st.subheader(
        "Regional Delay Diagnostics"
    )

    if not region_table.empty:

        region_display = region_table[
            [
                "Order Region",
                "Shipments",
                "Delayed",
                "Delay Rate (%)",
                "High_Risk",
                "High-Risk Rate (%)",
                "SLA / On-Time Rate (%)",
                "Avg_Delay"
            ]
        ].copy()

        region_display = region_display.rename(
            columns={
                "High_Risk": "High-Risk Shipments",
                "Avg_Delay": "Avg Delay (Days)"
            }
        )

        st.dataframe(
            region_display.style.format(
                {
                    "Delay Rate (%)": "{:.2f}%",
                    "High-Risk Rate (%)": "{:.2f}%",
                    "SLA / On-Time Rate (%)": "{:.2f}%",
                    "Avg Delay (Days)": "{:.2f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Market table
    # --------------------------------------------------------

    st.subheader(
        "Market-wise Logistics Efficiency"
    )

    if not market_table.empty:

        market_display = market_table[
            [
                "Market",
                "Shipments",
                "Delayed",
                "Delay Rate (%)",
                "High_Risk",
                "High-Risk Rate (%)",
                "SLA / On-Time Rate (%)",
                "Avg_Delay"
            ]
        ].copy()

        market_display = market_display.rename(
            columns={
                "High_Risk": "High-Risk Shipments",
                "Avg_Delay": "Avg Delay (Days)"
            }
        )

        st.dataframe(
            market_display.style.format(
                {
                    "Delay Rate (%)": "{:.2f}%",
                    "High-Risk Rate (%)": "{:.2f}%",
                    "SLA / On-Time Rate (%)": "{:.2f}%",
                    "Avg Delay (Days)": "{:.2f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Market × Region Heatmap
    # --------------------------------------------------------

    st.subheader(
        "Market × Region Delay Rate Heatmap"
    )

    heatmap = pd.pivot_table(
        filtered_df,
        index="Market",
        columns="Order Region",
        values="Delay Gap",
        aggfunc=lambda x: (x > 0).mean() * 100
    )

    if not heatmap.empty:

        st.dataframe(
            heatmap.style.format(
                "{:.2f}%"
            ).background_gradient(
                axis=None
            ),
            use_container_width=True
        )

        st.caption(
            "Each cell represents the delayed-shipment percentage "
            "for that Market × Region combination."
        )

    # --------------------------------------------------------
    # Geographic visualization
    # --------------------------------------------------------

    st.subheader(
        "Geographic Delay Visualization"
    )

    if (
        "Latitude" in filtered_df.columns
        and
        "Longitude" in filtered_df.columns
    ):

        map_df = filtered_df[
            [
                "Latitude",
                "Longitude",
                "Delay Gap"
            ]
        ].copy()

        map_df["Latitude"] = pd.to_numeric(
            map_df["Latitude"],
            errors="coerce"
        )

        map_df["Longitude"] = pd.to_numeric(
            map_df["Longitude"],
            errors="coerce"
        )

        map_df = map_df.dropna(
            subset=[
                "Latitude",
                "Longitude"
            ]
        )

        # Avoid rendering an extremely large number of points
        if len(map_df) > 5000:

            map_df = map_df.sample(
                5000,
                random_state=42
            )

        if not map_df.empty:

            st.map(
                map_df,
                latitude="Latitude",
                longitude="Longitude",
                size=8
            )

            st.caption(
                "Geographic distribution of filtered customer/order "
                "locations. The map shows location concentration; "
                "delay intensity is analyzed separately through "
                "the regional and market tables above."
            )

    else:

        st.info(
            "Latitude and Longitude fields are not available, "
            "so geographic visualization cannot be displayed."
        )

    # --------------------------------------------------------
    # High-risk hotspots
    # --------------------------------------------------------

    st.subheader(
        "🔥 High-Risk Market × Shipping Mode Hotspots"
    )

    hotspot = (
        filtered_df
        .groupby(
            ["Shipping Mode", "Market"],
            dropna=False
        )
        .agg(
            Shipments=("Delay Gap", "size"),
            Delayed=("Delay Gap", lambda x: (x > 0).sum()),
            High_Risk=("Late_delivery_risk", lambda x: (x == 1).sum()),
            Avg_Delay=("Delay Gap", lambda x: x[x > 0].mean())
        )
        .reset_index()
    )

    hotspot["Delay Rate (%)"] = (
        hotspot["Delayed"]
        /
        hotspot["Shipments"]
        * 100
    )

    hotspot["High-Risk Rate (%)"] = (
        hotspot["High_Risk"]
        /
        hotspot["Shipments"]
        * 100
    )

    hotspot["Avg_Delay"] = (
        hotspot["Avg_Delay"]
        .fillna(0)
    )

    # Minimum sample size prevents misleading tiny groups
    hotspot = hotspot[
        hotspot["Shipments"] >= 30
    ]

    hotspot = hotspot.sort_values(
        [
            "High-Risk Rate (%)",
            "Shipments"
        ],
        ascending=[
            False,
            False
        ]
    )

    hotspot_display = hotspot.head(15).copy()

    if not hotspot_display.empty:

        hotspot_display = hotspot_display[
            [
                "Shipping Mode",
                "Market",
                "Shipments",
                "Delayed",
                "Delay Rate (%)",
                "High_Risk",
                "High-Risk Rate (%)",
                "Avg_Delay"
            ]
        ]

        hotspot_display = hotspot_display.rename(
            columns={
                "High_Risk": "High-Risk Shipments",
                "Avg_Delay": "Avg Delay (Days)"
            }
        )

        st.dataframe(
            hotspot_display.style.format(
                {
                    "Delay Rate (%)": "{:.2f}%",
                    "High-Risk Rate (%)": "{:.2f}%",
                    "Avg Delay (Days)": "{:.2f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No Market × Shipping Mode combination has at least "
            "30 shipments under the current filters."
        )


# ============================================================
# TAB 5 — CUSTOMER & PRODUCT ANALYSIS
# ============================================================

with tabs[4]:

    st.header(
        "Customer Segment & Product Analysis"
    )

    st.markdown(
        """
        This section supports the methodology requirement to examine
        customer-segment impact and identify product/category areas
        associated with delivery risk.
        """
    )

    # --------------------------------------------------------
    # Customer segment
    # --------------------------------------------------------

    segment_table = rate_table(
        filtered_df,
        "Customer Segment"
    )

    st.subheader(
        "Customer Segment Delivery Performance"
    )

    if not segment_table.empty:

        segment_chart = (
            segment_table
            .set_index("Customer Segment")[
                ["Delay Rate (%)"]
            ]
        )

        st.bar_chart(
            segment_chart
        )

        segment_display = segment_table[
            [
                "Customer Segment",
                "Shipments",
                "Delayed",
                "Delay Rate (%)",
                "High_Risk",
                "High-Risk Rate (%)",
                "SLA / On-Time Rate (%)",
                "Avg_Delay"
            ]
        ].copy()

        segment_display = segment_display.rename(
            columns={
                "High_Risk": "High-Risk Shipments",
                "Avg_Delay": "Avg Delay (Days)"
            }
        )

        st.dataframe(
            segment_display.style.format(
                {
                    "Delay Rate (%)": "{:.2f}%",
                    "High-Risk Rate (%)": "{:.2f}%",
                    "SLA / On-Time Rate (%)": "{:.2f}%",
                    "Avg Delay (Days)": "{:.2f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Category analysis
    # --------------------------------------------------------

    if "Category Name" in filtered_df.columns:

        category_table = rate_table(
            filtered_df,
            "Category Name",
            min_count=10
        )

        st.subheader(
            "Product Category Delay Diagnostics"
        )

        if not category_table.empty:

            category_display = category_table[
                [
                    "Category Name",
                    "Shipments",
                    "Delayed",
                    "Delay Rate (%)",
                    "High_Risk",
                    "High-Risk Rate (%)"
                ]
            ].copy()

            category_display = category_display.rename(
                columns={
                    "High_Risk": "High-Risk Shipments"
                }
            )

            st.dataframe(
                category_display.head(20).style.format(
                    {
                        "Delay Rate (%)": "{:.2f}%",
                        "High-Risk Rate (%)": "{:.2f}%"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # Customer-level drill down
    # --------------------------------------------------------

    if "Customer Id" in filtered_df.columns:

        st.subheader(
            "Customer Risk Drill-Down"
        )

        customer = (
            filtered_df
            .groupby("Customer Id")
            .agg(
                Shipments=("Delay Gap", "size"),
                Delayed=("Delay Gap", lambda x: (x > 0).sum()),
                High_Risk=("Late_delivery_risk", lambda x: (x == 1).sum()),
                Avg_Delay=("Delay Gap", lambda x: x[x > 0].mean())
            )
            .reset_index()
        )

        customer["Delay Rate (%)"] = (
            customer["Delayed"]
            /
            customer["Shipments"]
            * 100
        )

        customer["High-Risk Rate (%)"] = (
            customer["High_Risk"]
            /
            customer["Shipments"]
            * 100
        )

        customer["Avg_Delay"] = (
            customer["Avg_Delay"]
            .fillna(0)
        )

        # Only show customers with sufficient shipment history
        customer = customer[
            customer["Shipments"] >= 3
        ]

        customer = customer.sort_values(
            [
                "High-Risk Rate (%)",
                "Shipments"
            ],
            ascending=[
                False,
                False
            ]
        )

        customer_display = customer.head(20).copy()

        if not customer_display.empty:

            customer_display = customer_display[
                [
                    "Customer Id",
                    "Shipments",
                    "Delayed",
                    "Delay Rate (%)",
                    "High_Risk",
                    "High-Risk Rate (%)",
                    "Avg_Delay"
                ]
            ]

            customer_display = customer_display.rename(
                columns={
                    "High_Risk": "High-Risk Shipments",
                    "Avg_Delay": "Avg Delay (Days)"
                }
            )

            st.dataframe(
                customer_display.style.format(
                    {
                        "Delay Rate (%)": "{:.2f}%",
                        "High-Risk Rate (%)": "{:.2f}%",
                        "Avg Delay (Days)": "{:.2f}"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Customer drill-down uses a minimum of 3 shipments "
                "to reduce misleading extreme percentages from "
                "single-order customers."
            )


# ============================================================
# TAB 6 — KEY FINDINGS & RECOMMENDATIONS
# ============================================================

with tabs[5]:

    st.header(
        "🎯 Key Findings & Operational Recommendations"
    )

    st.markdown(
        """
        Recommendations below are generated from the observed
        delivery gaps, risk segmentation, shipping-mode performance,
        regional/market analysis and customer-segment diagnostics.
        """
    )

    # ========================================================
    # CALCULATE MAIN FINDINGS
    # ========================================================

    # Highest delay shipping mode
    mode_table = rate_table(
        filtered_df,
        "Shipping Mode"
    )

    if not mode_table.empty:

        highest_mode = mode_table.iloc[0]

    else:

        highest_mode = None


    # Highest risk market
    market_table = rate_table(
        filtered_df,
        "Market"
    )

    if not market_table.empty:

        highest_market = market_table.iloc[0]

    else:

        highest_market = None


    # Highest risk region
    region_table = rate_table(
        filtered_df,
        "Order Region"
    )

    if not region_table.empty:

        highest_region = region_table.iloc[0]

    else:

        highest_region = None


    # Highest risk segment
    segment_table = rate_table(
        filtered_df,
        "Customer Segment"
    )

    if not segment_table.empty:

        highest_segment = segment_table.iloc[0]

    else:

        highest_segment = None


    # ========================================================
    # ACTION PRIORITY MATRIX
    # ========================================================

    st.subheader(
        "📌 Action Priority Matrix"
    )

    priority_data = []

    if highest_mode is not None:

        priority_data.append(
            [
                "🔴 Critical",
                "High-risk shipping modes",
                (
                    f"{highest_mode['Shipping Mode']} — "
                    f"{highest_mode['High-Risk Rate (%)']:.2f}% "
                    f"high-risk rate across "
                    f"{int(highest_mode['Shipments']):,} shipments"
                ),
                "Investigate operational bottlenecks immediately"
            ]
        )

    if highest_market is not None:

        priority_data.append(
            [
                "🟠 High",
                "Market-level delay exposure",
                (
                    f"{highest_market['Market']} — "
                    f"{highest_market['Delay Rate (%)']:.2f}% "
                    f"delay rate across "
                    f"{int(highest_market['Shipments']):,} shipments"
                ),
                "Prioritize regional operational review"
            ]
        )

    if highest_region is not None:

        priority_data.append(
            [
                "🟡 Medium",
                "Regional delay hotspots",
                (
                    f"{highest_region['Order Region']} — "
                    f"{highest_region['Delay Rate (%)']:.2f}% "
                    f"delay rate across "
                    f"{int(highest_region['Shipments']):,} shipments"
                ),
                "Investigate capacity and route bottlenecks"
            ]
        )

    if highest_segment is not None:

        priority_data.append(
            [
                "🟢 Monitor",
                "Customer segment exposure",
                (
                    f"{highest_segment['Customer Segment']} — "
                    f"{highest_segment['Delay Rate (%)']:.2f}% "
                    f"delay rate across "
                    f"{int(highest_segment['Shipments']):,} shipments"
                ),
                "Monitor SLA impact by customer segment"
            ]
        )

    priority_df = pd.DataFrame(
        priority_data,
        columns=[
            "Priority",
            "Focus Area",
            "Evidence",
            "Recommended Action"
        ]
    )

    st.dataframe(
        priority_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PRIORITY 1
    # ========================================================

    st.markdown(
        "### 🔴 Priority 1 — Reduce High-Risk Delays"
    )

    if highest_mode is not None:

        st.markdown(
            f"""
            <div class="danger-box">

            <b>Observation:</b>
            {highest_mode['High-Risk Rate (%)']:.2f}% of shipments
            in <b>{highest_mode['Shipping Mode']}</b> are flagged
            as high-risk, based on {int(highest_mode['Shipments']):,}
            shipments.

            <br><br>

            <b>Action:</b>
            Investigate this shipping mode for capacity,
            scheduling and operational bottlenecks.

            <br><br>

            <b>Measure:</b>
            Track high-risk rate, delay rate and average delay
            before and after intervention.

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PRIORITY 2
    # ========================================================

    st.markdown(
        "### 🟠 Priority 2 — Review Shipping Mode Bottlenecks"
    )

    if not mode_table.empty:

        worst_mode = mode_table.iloc[0]

        st.markdown(
            f"""
            <div class="warning-box">

            <b>Observation:</b>
            {worst_mode['Shipping Mode']} has the highest observed
            delay rate among the shipping modes in the current
            filtered dataset.

            <br><br>

            <b>Evidence:</b>
            {worst_mode['Delay Rate (%)']:.2f}% delay rate across
            {int(worst_mode['Shipments']):,} shipments.

            <br><br>

            <b>Action:</b>
            Compare scheduled shipping duration with actual duration
            and investigate consistently underperforming modes.

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PRIORITY 3
    # ========================================================

    st.markdown(
        "### 🟡 Priority 3 — Target Regional & Market Hotspots"
    )

    if (
        highest_market is not None
        and
        highest_region is not None
    ):

        st.markdown(
            f"""
            <div class="warning-box">

            <b>Observation:</b>
            The highest observed market delay rate is in
            <b>{highest_market['Market']}</b>, while the highest
            observed regional delay rate is in
            <b>{highest_region['Order Region']}</b>.

            <br><br>

            <b>Evidence:</b>
            Market delay rate:
            {highest_market['Delay Rate (%)']:.2f}%
            across {int(highest_market['Shipments']):,} shipments.

            <br>
            Regional delay rate:
            {highest_region['Delay Rate (%)']:.2f}%
            across {int(highest_region['Shipments']):,} shipments.

            <br><br>

            <b>Action:</b>
            Use the Market × Region heatmap to identify combinations
            with both sufficient shipment volume and elevated delay risk.

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PRIORITY 4
    # ========================================================

    st.markdown(
        "### 🟢 Priority 4 — Improve Planning Accuracy"
    )

    st.markdown(
        f"""
        <div class="success-box">

        <b>Observation:</b>
        The current dataset contains both scheduled and actual
        shipping duration, allowing planned-versus-actual performance
        to be evaluated directly.

        <br><br>

        <b>Action:</b>
        Review scheduled shipping times against observed delivery
        performance and identify consistently underestimated
        or overestimated durations.

        <br><br>

        <b>Measure:</b>
        Monitor average delay gap and SLA/on-time rate over time
        when date information becomes available.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # INTERPRETATION NOTE
    # ========================================================

    st.subheader(
        "⚠️ Interpretation Note"
    )

    st.markdown(
        """
        <div class="warning-box">

        Extreme delay or risk percentages should always be interpreted
        alongside shipment counts.

        A very high percentage from a small group can be unstable and
        should not automatically be treated as a major operational
        problem.

        Therefore, this dashboard applies minimum shipment thresholds
        when identifying high-risk Market × Shipping Mode hotspots.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # KEY FINDINGS
    # ========================================================

    st.subheader(
        "📌 Key Findings"
    )

    finding1 = (
        f"Overall delay rate: "
        f"{kpis['delay_rate']:.2f}% "
        f"({kpis['delayed']:,} delayed shipments)."
    )

    finding2 = (
        f"Overall high-risk rate: "
        f"{kpis['risk_rate']:.2f}% "
        f"({kpis['risk_shipments']:,} high-risk shipments)."
    )

    finding3 = (
        f"Average delay among delayed shipments: "
        f"{kpis['avg_delay']:.2f} days."
    )

    st.markdown(
        f"""
        - {finding1}
        - {finding2}
        - {finding3}
        """
    )

    if highest_mode is not None:

        st.markdown(
            f"- Highest shipping-mode delay rate: "
            f"**{highest_mode['Shipping Mode']}** "
            f"({highest_mode['Delay Rate (%)']:.2f}%, "
            f"{int(highest_mode['Shipments']):,} shipments)."
        )

    if highest_market is not None:

        st.markdown(
            f"- Highest market delay rate: "
            f"**{highest_market['Market']}** "
            f"({highest_market['Delay Rate (%)']:.2f}%, "
            f"{int(highest_market['Shipments']):,} shipments)."
        )

    if highest_region is not None:

        st.markdown(
            f"- Highest regional delay rate: "
            f"**{highest_region['Order Region']}** "
            f"({highest_region['Delay Rate (%)']:.2f}%, "
            f"{int(highest_region['Shipments']):,} shipments)."
        )


    # ========================================================
    # PREDICTIVE MODEL NOTE
    # ========================================================

    st.subheader(
        "🔮 Predictive Analytics — Project Scope"
    )

    st.info(
        "A predictive model is NOT included because the supplied "
        "project requirements define this phase as a diagnostic "
        "delivery-performance analysis. The project conclusion "
        "describes predictive or optimization-based models as a "
        "future extension after this analytical foundation."
    )


# ============================================================
# FOOTER / DATASET INFORMATION
# ============================================================

st.divider()

st.caption(
    f"Dataset: {csv_path.name} | "
    f"Displayed records: {len(filtered_df):,} | "
    f"Dashboard purpose: Diagnostic logistics intelligence"
)

st.caption(
    "Delay Gap = Days for shipping (real) − "
    "Days for shipment (scheduled)."
)