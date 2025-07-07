import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path

# Set page configuration
st.set_page_config(page_title="IDW Interpolation App", layout="wide")

# Title
st.title("Inverse Weighted Distance Interpolation of K*h Values")
st.markdown("This app performs Inverse Distance Weighting (IDW) interpolation based on measured bottomhole formation permeability and thickness.")

# Define dataset
data = {...}  # keep your existing data dict here

df = pd.DataFrame(data)

# Sidebar inputs
st.sidebar.header("Input Well Coordinates")
target_lat = st.sidebar.number_input("Latitude", value=49.85, format="%.6f")
target_lon = st.sidebar.number_input("Longitude", value=-102.9, format="%.6f")
power = st.sidebar.slider("IDW Power (controls influence drop-off)", min_value=1, max_value=10, value=2)
max_reach_km = st.sidebar.number_input("Max Reach Distance (km)", value=10.0, min_value=0.0, step=0.5)

# Distance in km function
def distance_km(lon1, lat1, lon2, lat2):
    mean_lat_rad = np.radians((lat1 + lat2) / 2.0)
    delta_lon = (lon1 - lon2) * 111.32 * np.cos(mean_lat_rad)
    delta_lat = (lat1 - lat2) * 111.32
    return np.sqrt(delta_lon**2 + delta_lat**2)

lats = df['Lat'].values
lons = df['Long'].values
values = df['Kh'].values

# Exact match
exact_match = df[(df['Lat'] == target_lat) & (df['Long'] == target_lon)]
if not exact_match.empty:
    matched_value = exact_match['Kh'].values[0]
    st.success(f"✅ Exact match found. Value at (Lat: {target_lat}, Lon: {target_lon}) is {matched_value}")
    interpolated_value = matched_value
else:
    # Convex hull bounds
    points = np.column_stack((lons, lats))
    hull = ConvexHull(points)
    hull_path = Path(points[hull.vertices])

    distances = distance_km(lons, lats, target_lon, target_lat)
    within_reach = distances <= max_reach_km

    if not hull_path.contains_point((target_lon, target_lat)):
        st.warning("⚠️ Point is outside convex hull. Interpolation skipped.")
        interpolated_value = None
    elif not np.any(within_reach):
        st.warning("⚠️ No data points within max reach distance. Interpolation skipped.")
        interpolated_value = None
    else:
        filtered_distances = distances[within_reach]
        filtered_values = values[within_reach]
        if np.any(filtered_distances == 0):
            interpolated_value = filtered_values[filtered_distances == 0][0]
        else:
            weights = 1 / filtered_distances**power
            interpolated_value = np.sum(weights * filtered_values) / np.sum(weights)
        st.success(f"✅ Interpolated value at (Lat: {target_lat}, Lon: {target_lon}) is {interpolated_value:.2f}")

# Generate contour map
grid_lon = np.linspace(min(lons), max(lons), 200)
grid_lat = np.linspace(min(lats), max(lats), 200)
grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
grid_z = np.full_like(grid_lon_mesh, np.nan)

# Compute IDW over grid
for i in range(grid_lon_mesh.shape[0]):
    for j in range(grid_lon_mesh.shape[1]):
        gx, gy = grid_lon_mesh[i, j], grid_lat_mesh[i, j]
        dists = distance_km(lons, lats, gx, gy)
        within = dists <= max_reach_km
        if not np.any(within):
            continue  # remains NaN
        nearby_dists = dists[within]
        nearby_vals = values[within]
        if np.any(nearby_dists == 0):
            grid_z[i, j] = nearby_vals[nearby_dists == 0][0]
        else:
            w = 1 / nearby_dists**power
            grid_z[i, j] = np.sum(w * nearby_vals) / np.sum(w)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
cmap = plt.cm.cividis
contour = ax.contourf(grid_lon_mesh, grid_lat_mesh, grid_z, cmap=cmap, levels=20)
scatter = ax.scatter(lons, lats, c=values, edgecolor='k', cmap='viridis', label='Data Points')
ax.scatter(target_lon, target_lat, color='red', marker='x', s=100, label='Target Location')
plt.colorbar(contour, ax=ax, label='Interpolated Value')
ax.set_title("IDW Interpolation (km scaled distances) with Target Location")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
st.pyplot(fig)
