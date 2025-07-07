import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from haversine import haversine, Unit

# Page config
st.set_page_config(page_title="IDW Interpolation App", layout="wide")
st.title("Inverse Weighted Distance Interpolation of K*h Values")
st.markdown(
    "This app performs IDW interpolation directly on lat/lon using haversine distances (km). "
    "It uses all points within the convex hull of your data (no radius cutoff)."
)

# Define the dataset
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

# Sidebar inputs
st.sidebar.header("Input Well Coordinates")
target_lat = st.sidebar.number_input("Latitude", value=49.85, format="%.6f")
target_lon = st.sidebar.number_input("Longitude", value=-102.9, format="%.6f")
power = st.sidebar.slider("IDW Power", min_value=1, max_value=10, value=2)

# Extract data
lats = df["Lat"].values
lons = df["Long"].values
values = df["Kh"].values

# Convex hull setup
points = np.column_stack((lons, lats))
hull = ConvexHull(points)
hull_path = Path(points[hull.vertices])

# IDW interpolation at target point
if not hull_path.contains_point((target_lon, target_lat)):
    st.warning("⚠️ Point is outside convex hull. Interpolation skipped.")
    interpolated_value = None
else:
    dists = np.array([haversine((target_lat, target_lon), (lat, lon), unit=Unit.KILOMETERS)
                      for lat, lon in zip(lats, lons)])
    dists[dists == 0] = 1e-6
    weights = 1 / dists**power
    interpolated_value = np.sum(weights * values) / np.sum(weights)
    st.success(f"✅ Interpolated value at (Lat: {target_lat}, Lon: {target_lon}): {interpolated_value:.2f}")

# Create grid for map
grid_lon = np.linspace(min(lons), max(lons), 200)
grid_lat = np.linspace(min(lats), max(lats), 200)
grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
grid_z = np.full_like(grid_lon_mesh, np.nan)

# Mask grid to convex hull
grid_points = np.column_stack((grid_lon_mesh.ravel(), grid_lat_mesh.ravel()))
mask = hull_path.contains_points(grid_points).reshape(grid_lon_mesh.shape)

# Interpolation over grid
for i in range(grid_lat_mesh.shape[0]):
    for j in range(grid_lat_mesh.shape[1]):
        if not mask[i, j]:
            continue
        gx, gy = grid_lon_mesh[i, j], grid_lat_mesh[i, j]
        dists = np.array([haversine((gy, gx), (lat, lon), unit=Unit.KILOMETERS)
                          for lat, lon in zip(lats, lons)])
        dists[dists == 0] = 1e-6
        weights = 1 / dists**power
        grid_z[i, j] = np.sum(weights * values) / np.sum(weights)

# Plot
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
ax.set_facecolor('dimgray')  # set dark gray background

contour = ax.contourf(
    grid_lon_mesh, grid_lat_mesh, grid_z,
    cmap='inferno', levels=200
)

# Wells & target
ax.scatter(lons, lats, c='white', edgecolor='k', label='Data Points')
ax.scatter(target_lon, target_lat, color='cyan', marker='x', s=100, label='Target Location')

plt.colorbar(contour, ax=ax, label='Interpolated Value')
ax.set_title("IDW Interpolation Map (Haversine, inside Convex Hull)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

st.pyplot(fig)
