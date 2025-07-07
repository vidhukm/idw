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

# Create DataFrame
df = pd.DataFrame(data)

# Sidebar inputs
st.sidebar.header("Input Well Coordinates")
target_lat = st.sidebar.number_input("Latitude", value=49.85, format="%.6f")
target_lon = st.sidebar.number_input("Longitude", value=-102.9, format="%.6f")
power = st.sidebar.slider("IDW Power (controls reach)", min_value=1, max_value=10, value=2)
max_reach = st.sidebar.number_input("Max Reach Distance (degrees)", value=0.1, min_value=0.0, step=0.01)

# Extract coordinates and values
lats = df['Lat'].values
lons = df['Long'].values
values = df['Kh'].values

# Precompute cos of average latitude for distance scaling
lat_avg = np.mean(lats)
cos_lat = np.cos(np.radians(lat_avg))

def scaled_distance(lon1, lat1, lon2, lat2):
    return np.sqrt((cos_lat * (lon1 - lon2))**2 + (lat1 - lat2)**2)

# Check for exact match
exact_match = df[(df['Lat'] == target_lat) & (df['Long'] == target_lon)]
if not exact_match.empty:
    matched_value = exact_match['Kh'].values[0]
    st.success(f"✅ Exact match found. Value at (Lat: {target_lat}, Lon: {target_lon}) is {matched_value}")
    interpolated_value = matched_value
else:
    # Define bounds (still in raw lon/lat degrees)
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    # Check if point is within convex hull
    points = np.column_stack((lons, lats))
    hull = ConvexHull(points)
    hull_path = Path(points[hull.vertices])

    if not (min_lon <= target_lon <= max_lon and min_lat <= target_lat <= max_lat):
        st.warning("⚠️ Point is outside the designated area. Interpolation skipped.")
        interpolated_value = None
    else:
        distances = scaled_distance(lons, lats, target_lon, target_lat)
        within_reach = distances <= max_reach

        if not np.any(within_reach):
            st.warning("⚠️ No data points within the max reach distance. Interpolation skipped.")
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
grid_z = np.zeros_like(grid_lon_mesh)

# Compute IDW over grid with scaled distances and max reach
for i in range(grid_lon_mesh.shape[0]):
    for j in range(grid_lon_mesh.shape[1]):
        gx, gy = grid_lon_mesh[i, j], grid_lat_mesh[i, j]
        dists = scaled_distance(lons, lats, gx, gy)
        within_reach = dists <= max_reach

        if not np.any(within_reach):
            grid_z[i, j] = np.nan  # Mark as no data
        else:
            filtered_dists = dists[within_reach]
            filtered_values = values[within_reach]

            if np.any(filtered_dists == 0):
                grid_z[i, j] = filtered_values[filtered_dists == 0][0]
            else:
                w = 1 / filtered_dists**power
                grid_z[i, j] = np.sum(w * filtered_values) / np.sum(w)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
contour = ax.contourf(grid_lon_mesh, grid_lat_mesh, grid_z, cmap='cividis', levels=20)
scatter = ax.scatter(lons, lats, c=values, edgecolor='k', cmap='viridis', label='Data Points')
ax.scatter(target_lon, target_lat, color='red', marker='x', s=100, label='Target Location')
plt.colorbar(contour, ax=ax, label='Interpolated Value')
ax.set_title("IDW Interpolation with Target Location")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
st.pyplot(fig)
