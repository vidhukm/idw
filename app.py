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

# Data
data = {
    'UWI': [...],  # your UWI list
    'Lat': [...],  # your latitude list
    'Long': [...], # your longitude list
    'Kh': [...]    # your Kh list
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
