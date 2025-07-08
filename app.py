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
# Data
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
# Build grid for contour
# ------------------------
grid_lon = np.linspace(df["Long"].min(), df["Long"].max(), 200)
grid_lat = np.linspace(df["Lat"].min(), df["Lat"].max(), 200)
grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
grid_z = np.full_like(grid_lon_mesh, np.nan, dtype=float)

grid_points = np.column_stack((grid_lon_mesh.ravel(), grid_lat_mesh.ravel()))
mask = hull_path.contains_points(grid_points).reshape(grid_lon_mesh.shape)

for i in range(grid_lat_mesh.shape[0]):
    for j in range(grid_lat_mesh.shape[1]):
        if mask[i, j]:
            grid_z[i, j] = idw_interpolation(grid_lon_mesh[i, j], grid_lat_mesh[i, j], power)

# ------------------------
# Plotly plot with hover
# ------------------------
fig = go.Figure()

# Add contour
fig.add_trace(go.Contour(
    x=grid_lon,
    y=grid_lat,
    z=grid_z,
    colorscale='Inferno',
    contours=dict(start=np.nanmin(grid_z), end=np.nanmax(grid_z), size=0.1),
    colorbar=dict(title="Interpolated Value"),
    hovertemplate='Lon: %{x}<br>Lat: %{y}<br>Interp: %{z:.2f}<extra></extra>'
))

# Add scatter for data points
fig.add_trace(go.Scatter(
    x=df["Long"],
    y=df["Lat"],
    mode='markers',
    marker=dict(color='white', size=8, line=dict(color='black', width=1)),
    name='Data Points',
    text=df.apply(lambda row: f"UWI: {row['UWI']}<br>Lat: {row['Lat']:.5f}<br>Long: {row['Long']:.5f}<br>Kh: {row['Kh']}", axis=1),
    hoverinfo='text'
))

# Add target point
fig.add_trace(go.Scatter(
    x=[target_lon],
    y=[target_lat],
    mode='markers',
    marker=dict(color='cyan', size=10, symbol='x'),
    name='Target',
    hovertemplate=f"Target Location<br>Lat: {target_lat}<br>Lon: {target_lon}<br>Interp: {interpolated_value:.2f}" if interpolated_value else "Outside convex hull"
))

fig.update_layout(
    title="IDW Interpolation Map with Hover Labels",
    xaxis_title="Longitude",
    yaxis_title="Latitude",
    plot_bgcolor='dimgray',
    paper_bgcolor='dimgray',
    font_color='white',
    width=1200,
    height=700
)

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5
    )
)

st.plotly_chart(fig, use_container_width=False)
