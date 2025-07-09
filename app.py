import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from haversine import haversine, Unit
import plotly.graph_objects as go

# ------------------------
# Streamlit Setup
# ------------------------
st.set_page_config(page_title="IDW Interpolation", layout="wide")
st.title("Inverse Distance Weighting of Measured K*h Values")
st.markdown(
    "Performs IDW interpolation on latitude/longitude using haversine distances (km). "
    "Interpolation is confined to the perimeter of the data."
)

# ------------------------
# Original Data
# ------------------------
data = {
    'UWI': [
        '101/01-29-011-06W2/00', '101/04-22-010-05W2/00', '101/05-23-010-10W2/00',
        '101/11-25-010-08W2/00', '101/13-07-009-05W2/00', '101/16-29-010-06W2/00',
        '111/03-27-011-05W2/00', '111/04-08-011-07W2/00', '111/04-12-012-07W2/00',
        '111/04-18-009-09W2/00', '111/04-21-009-07W2/00', '121/03-03-010-07W2/00',
        '121/05-29-010-11W2/00', '121/12-05-011-05W2/00', '121/14-24-012-08W2/00',
        '131/02-08-012-04W2/00', '131/13-21-009-07W2/00', '141/08-27-012-06W2/00',
        '141/13-18-010-08W2/00', '141/15-26-011-06W2/00', '141/16-02-011-08W2/00',
        '191/04-28-009-10W2/00', '191/16-26-010-06W2/00'
    ],
    'Lat': [
        49.931383, 49.829286, 49.833671, 49.855245, 49.724227, 49.85674,
        49.931954, 49.888158, 49.974766, 49.72755, 49.741497, 49.785373,
        49.847617, 49.880096, 50.015431, 49.977522, 49.75371, 50.02486,
        49.827066, 49.944846, 49.885641, 49.75847, 49.855041
    ],
    'Long': [
        -102.781818, -102.613696, -103.269454, -102.968955, -102.681063, -102.776664,
        -102.610789, -102.934808, -102.843135, -103.222192, -102.904647, -102.878683,
        -103.471582, -102.663714, -102.977483, -102.515151, -102.907216, -102.735345,
        -103.086901, -102.718421, -102.985106, -103.313913, -102.70758
    ],
    'Kh': [
        1.39, 4.77, 0.8, 1.29, 0.7, 6.07, 0.42, 1.18, 0.67, 0.22, 1.79, 0.53,
        0.53, 0.96, 0.99, 1.7, 0.8, 0.38, 0.92, 2.64, 0.52, 0.24, 0.93
    ]
}
df = pd.DataFrame(data)

# ------------------------
# Sidebar inputs
# ------------------------
st.sidebar.header("Target Location")
target_lat = st.sidebar.number_input("Latitude", value=49.85, format="%.6f")
target_lon = st.sidebar.number_input("Longitude", value=-102.9, format="%.6f")
power = st.sidebar.slider("IDW Power", 1, 10, 2)

# ------------------------
# Convex hull
# ------------------------
points = df[["Long", "Lat"]].values
hull = ConvexHull(points)
hull_path = Path(points[hull.vertices])

# ------------------------
# IDW interpolation
# ------------------------
def idw_interpolation(x, y, power):
    dists = np.array([
        haversine((y, x), (lat, lon), unit=Unit.KILOMETERS)
        for lat, lon in zip(df["Lat"], df["Long"])
    ])
    if np.any(dists == 0):
        return df.loc[dists == 0, "Kh"].iloc[0]
    weights = 1 / dists ** power
    return np.sum(weights * df["Kh"]) / np.sum(weights)

# ------------------------
# Compute target
# ------------------------
if not hull_path.contains_point((target_lon, target_lat)):
    st.warning("Target is outside convex hull.")
    interpolated_value = None
else:
    interpolated_value = idw_interpolation(target_lon, target_lat, power)
    st.success(f"Interpolated value: {interpolated_value:.2f}")

# ------------------------
# Additional points (no Kh)
# ------------------------
new_points = pd.DataFrame({
    'UWI': [
        '101/02-18-010-05W2/00', '101/06-21-010-05W2/00', '101/12-31-010-05W2/00',
        '101/01-32-010-05W2/00', '101/12-34-010-08W2/00', '101/08-26-010-10W2/00',
        '101/08-07-011-05W2/00', '102/14-14-011-06W2/00', '105/14-17-011-06W2/00',
        '103/15-17-011-06W2/00', '101/14-19-011-06W2/00', '101/16-22-011-06W2/00',
        '103/14-23-011-06W2/00', '101/14-27-011-06W2/00', '101/14-29-011-06W2/00',
        '101/01-30-011-06W2/00', '101/04-31-011-06W2/00', '102/04-34-011-06W2/00',
        '101/05-34-011-06W2/00', '101/08-36-011-06W2/00', '101/14-06-011-07W2/00',
        '101/09-11-011-07W2/00', '101/12-11-011-07W2/00', '101/02-36-011-07W2/00',
        '101/03-13-011-08W2/00'
    ],
    'Lat': [
        49.814771, 49.832132, 49.866327, 49.857341, 49.868215, 49.846982,
        49.893545, 49.914881, 49.916131, 49.915046, 49.927481, 49.930120,
        49.929340, 49.959624, 49.944613, 49.932108, 49.946346, 49.945439,
        49.948868, 49.949333, 49.886972, 49.894420, 49.897549, 49.947061,
        49.902719
    ],
    'Long': [
        -102.667045, -102.627060, -102.680538, -102.638395, -103.020105, -103.249499,
        -102.665668, -102.726501, -102.793379, -102.787880, -102.818935, -102.733293,
        -102.722465, -102.749376, -102.790980, -102.807662, -102.824224, -102.754334,
        -102.754400, -102.689783, -102.950880, -102.849813, -102.867676, -102.831193,
        -102.976637
    ]
})

def interpolate_new_points(df_new):
    interpolated = []
    for _, row in df_new.iterrows():
        if hull_path.contains_point((row
