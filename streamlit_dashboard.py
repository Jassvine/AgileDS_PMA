import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(page_title="Agile DS PMA Dashboard", layout="wide")


@st.cache_data
def load_and_prepare_data(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("train.csv") as f:
            df = pd.read_csv(f)

    df = df.copy()

    # Clean target and core numeric columns.
    df["Time_taken(min)"] = (
        df["Time_taken(min)"]
        .astype(str)
        .str.replace("(min) ", "", regex=False)
    )
    df["Time_taken(min)"] = pd.to_numeric(df["Time_taken(min)"], errors="coerce")

    df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
    df["Delivery_person_Ratings"] = pd.to_numeric(
        df["Delivery_person_Ratings"], errors="coerce"
    )
    df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")

    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")

    order_ts = pd.to_datetime(df["Time_Orderd"].astype(str), format="%H:%M:%S", errors="coerce")
    pickup_ts = pd.to_datetime(
        df["Time_Order_picked"].astype(str), format="%H:%M:%S", errors="coerce"
    )

    # Feature engineering (Sprint requirement).
    df["order_hour"] = order_ts.dt.hour
    df["pickup_hour"] = pickup_ts.dt.hour
    df["prep_time_min"] = (pickup_ts - order_ts).dt.total_seconds() / 60
    df.loc[df["prep_time_min"] < 0, "prep_time_min"] += 24 * 60
    df["order_dayofweek"] = df["Order_Date"].dt.dayofweek
    df["is_weekend"] = df["order_dayofweek"].isin([5, 6]).astype(int)

    # Distance feature from coordinates if not already present.
    if "distance_km" not in df.columns:
        lat1 = np.radians(df["Restaurant_latitude"])
        lon1 = np.radians(df["Restaurant_longitude"])
        lat2 = np.radians(df["Delivery_location_latitude"])
        lon2 = np.radians(df["Delivery_location_longitude"])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        df["distance_km"] = 6371 * c

    traffic_map = {"low": 1, "medium": 2, "high": 3, "jam": 4}
    traffic_norm = df["Road_traffic_density"].astype(str).str.strip().str.lower()
    df["traffic_score"] = traffic_norm.map(traffic_map).fillna(0)
    df["distance_traffic_interaction"] = df["distance_km"] * df["traffic_score"]

    df = df.dropna(subset=["Time_taken(min)"]).copy()
    return df


@st.cache_resource
def train_best_model(df: pd.DataFrame):
    feature_cols = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Vehicle_condition",
        "multiple_deliveries",
        "distance_km",
        "order_hour",
        "prep_time_min",
        "order_dayofweek",
        "is_weekend",
        "Road_traffic_density",
        "Weatherconditions",
        "Type_of_order",
        "Type_of_vehicle",
        "City",
    ]
    target_col = "Time_taken(min)"

    model_df = df[feature_cols + [target_col]].copy()

    X = model_df[feature_cols]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Vehicle_condition",
        "multiple_deliveries",
        "distance_km",
        "order_hour",
        "prep_time_min",
        "order_dayofweek",
        "is_weekend",
    ]
    categorical_features = [
        "Road_traffic_density",
        "Weatherconditions",
        "Type_of_order",
        "Type_of_vehicle",
        "City",
    ]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
    }

    defaults = {
        "age": float(df["Delivery_person_Age"].median()),
        "rating": float(df["Delivery_person_Ratings"].median()),
        "vehicle_condition": int(pd.to_numeric(df["Vehicle_condition"], errors="coerce").median()),
        "multiple_deliveries": int(pd.to_numeric(df["multiple_deliveries"], errors="coerce").median()),
        "distance": float(df["distance_km"].median()),
        "order_hour": int(pd.to_numeric(df["order_hour"], errors="coerce").median()),
        "prep_time": float(pd.to_numeric(df["prep_time_min"], errors="coerce").median()),
        "order_dayofweek": int(pd.to_numeric(df["order_dayofweek"], errors="coerce").median()),
    }

    return model, metrics, defaults


st.title("Agile DS PMA Decision Support Dashboard")
st.caption("Model-driven delivery time insights using the best Q2 model (Random Forest).")

zip_path = "PMA dataset.zip"
df = load_and_prepare_data(zip_path)
model, model_metrics, defaults = train_best_model(df)

st.sidebar.header("Interactive Controls")
selected_cities = st.sidebar.multiselect(
    "Filter by City",
    options=sorted(df["City"].dropna().astype(str).unique()),
    default=sorted(df["City"].dropna().astype(str).unique())[:2],
)
selected_traffic = st.sidebar.multiselect(
    "Filter by Road Traffic",
    options=sorted(df["Road_traffic_density"].dropna().astype(str).unique()),
    default=sorted(df["Road_traffic_density"].dropna().astype(str).unique()),
)
hour_range = st.sidebar.slider("Order Hour Range", 0, 23, (8, 22))
show_weekends_only = st.sidebar.checkbox("Show weekends only", value=False)

filtered = df.copy()
if selected_cities:
    filtered = filtered[filtered["City"].astype(str).isin(selected_cities)]
if selected_traffic:
    filtered = filtered[filtered["Road_traffic_density"].astype(str).isin(selected_traffic)]
filtered = filtered[filtered["order_hour"].between(hour_range[0], hour_range[1], inclusive="both")]
if show_weekends_only:
    filtered = filtered[filtered["is_weekend"] == 1]

c1, c2, c3 = st.columns(3)
c1.metric("Filtered Rows", f"{len(filtered):,}")
c2.metric("Avg Delivery Time (min)", f"{filtered['Time_taken(min)'].mean():.2f}")
c3.metric("Model R2", f"{model_metrics['R2']:.3f}")

st.subheader("Visualization 1: Delivery Time Distribution")
fig1 = px.histogram(
    filtered.dropna(subset=["Time_taken(min)"]),
    x="Time_taken(min)",
    nbins=30,
    marginal="violin",
    color_discrete_sequence=["#1f77b4"],
    labels={"Time_taken(min)": "Time Taken (min)"},
    title="Distribution of Delivery Times",
)
fig1.update_layout(bargap=0.05)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Visualization 2: Average Delivery Time by Traffic Density")
traffic_avg = (
    filtered.groupby("Road_traffic_density")["Time_taken(min)"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "avg_time", "count": "count"})
    .sort_values("avg_time", ascending=False)
)
fig2 = px.bar(
    traffic_avg,
    x="Road_traffic_density",
    y="avg_time",
    color="Road_traffic_density",
    text=traffic_avg["avg_time"].round(1),
    hover_data={"count": True, "avg_time": ":.2f"},
    color_discrete_sequence=px.colors.sequential.Viridis_r,
    labels={"Road_traffic_density": "Traffic Density", "avg_time": "Avg Time (min)", "count": "Orders"},
    title="Average Delivery Time by Road Traffic Density",
)
fig2.update_traces(textposition="outside")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Visualization 3: Distance vs Delivery Time")
scatter_df = (
    filtered[["distance_km", "Time_taken(min)", "Road_traffic_density", "City", "Weatherconditions"]]
    .dropna()
    .sample(n=min(2500, len(filtered)), random_state=42)
)
fig3 = px.scatter(
    scatter_df,
    x="distance_km",
    y="Time_taken(min)",
    color="Road_traffic_density",
    hover_data={"City": True, "Weatherconditions": True, "distance_km": ":.2f", "Time_taken(min)": True},
    labels={"distance_km": "Distance (km)", "Time_taken(min)": "Time Taken (min)", "Road_traffic_density": "Traffic"},
    title="Distance vs Delivery Time (coloured by Traffic Density)",
    opacity=0.55,
    trendline="ols",
)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Predictive Output: Estimate Delivery Time")
with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        in_age = st.slider("Delivery Person Age", 18, 60, int(np.clip(defaults["age"], 18, 60)))
        in_rating = st.slider("Delivery Rating", 1.0, 5.0, float(np.clip(defaults["rating"], 1.0, 5.0)), 0.1)
        in_vehicle_condition = st.slider("Vehicle Condition", 0, 3, int(np.clip(defaults["vehicle_condition"], 0, 3)))

    with col2:
        in_multiple = st.slider("Multiple Deliveries", 0, 3, int(np.clip(defaults["multiple_deliveries"], 0, 3)))
        in_distance = st.slider("Distance (km)", 0.5, 25.0, float(np.clip(defaults["distance"], 0.5, 25.0)), 0.1)
        in_order_hour = st.slider("Order Hour", 0, 23, int(np.clip(defaults["order_hour"], 0, 23)))

    with col3:
        in_prep = st.slider("Prep Time (min)", 0.0, 120.0, float(np.clip(defaults["prep_time"], 0.0, 120.0)), 1.0)
        in_day = st.selectbox("Day of Week", options=list(range(7)), index=int(np.clip(defaults["order_dayofweek"], 0, 6)))
        in_city = st.selectbox("City", options=sorted(df["City"].dropna().astype(str).unique()))

    in_traffic = st.selectbox("Road Traffic Density", options=sorted(df["Road_traffic_density"].dropna().astype(str).unique()))
    in_weather = st.selectbox("Weather", options=sorted(df["Weatherconditions"].dropna().astype(str).unique()))
    in_order_type = st.selectbox("Type of Order", options=sorted(df["Type_of_order"].dropna().astype(str).unique()))
    in_vehicle_type = st.selectbox("Type of Vehicle", options=sorted(df["Type_of_vehicle"].dropna().astype(str).unique()))

    submitted = st.form_submit_button("Predict Delivery Time")

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "Delivery_person_Age": in_age,
                "Delivery_person_Ratings": in_rating,
                "Vehicle_condition": in_vehicle_condition,
                "multiple_deliveries": in_multiple,
                "distance_km": in_distance,
                "order_hour": in_order_hour,
                "prep_time_min": in_prep,
                "order_dayofweek": in_day,
                "is_weekend": int(in_day in [5, 6]),
                "Road_traffic_density": in_traffic,
                "Weatherconditions": in_weather,
                "Type_of_order": in_order_type,
                "Type_of_vehicle": in_vehicle_type,
                "City": in_city,
            }
        ]
    )

    predicted_time = float(model.predict(input_df)[0])
    st.success(f"Predicted delivery time: {predicted_time:.2f} minutes")

    if predicted_time > 35:
        st.warning("Decision Hint: High estimated delay. Consider rider reassignment or priority routing.")
    elif predicted_time > 25:
        st.info("Decision Hint: Moderate delay risk. Monitor traffic and prep-time closely.")
    else:
        st.success("Decision Hint: Low delay risk. Current configuration is acceptable.")

st.subheader("Model Performance (Best Q2 Model)")
st.write(pd.DataFrame([model_metrics]).round(4))
