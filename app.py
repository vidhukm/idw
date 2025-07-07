import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from pyproj import Transformer

# Set page configuration
st.set_page_config(page_title="IDW Interpolation App", layout="wide")

# Title
st.title("Inverse Weighted Distance Interpolation of K*h Values")
st.markdown("This app performs Inverse Distance Weighting (IDW) interpolation based on measured bottomhole formation permeability and thickness.")

# Define the dataset
data = {
    'UWI': [...],  # Keep as is
    'Lat': [...],  # Keep as is
    'Long': [...],  # Keep as is
    'Kh': [...]  # Keep as is
}

# Create DataFrame
df = pd.DataFrame(data)

# Sidebar inputs
st.sidebar.header("Input Well Coordinates")
target_lat = st.sidebar.number_input("Latitude", value=49.85, format="%.6f")
target_lon = st.sidebar.number_input("Longitude", value=-102.9, format="%.6f")
power = st.sidebar.slider("IDW Power (controls reach)", min_value=1, max_value=10, value=2)

# Coordinate transformer: WGS84 to UTM Zone 13N (covers SK)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32613", always_xy=True)

# Transform all lat/lon to projected x/y
df["X"], df["Y"] = transformer.transform(df["Long"].values, df["Lat"].values)
target_x, target_y = transformer.transform(target_lon, target_lat)

# Extract coordinates and values
x = df['X'].values
y = df['Y'].values
values = df['Kh'].values

# Check for exact match
exact_match = df[(df['Lat'] == target_lat) & (df['Long'] == target_lon)]
if not exact_match.empty:
    matched_value = exact_match['Kh'].values[0]
    st.success(f"✅ Exact match found. Value at (Lat: {target_lat}, Lon: {target_lon}) is {matched_value}")
    interpolated_value = matched_value
else:
    # Define bounds
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)

    # Check if point is within convex hull
    points = np.column_stack((x, y))
    hull = ConvexHull(points)
    hull_path = Path(points[hull.vertices])

    if not hull_path.contains_point((target_x, target_y)):
        st.warning("⚠️ Point is outside the designated area. Interpolation skipped.")
        interpolated_value = None
    else:
        distances = np.sqrt((x - target_x)**2 + (y - target_y)**2)
        if np.any(distances == 0):
            interpolated_value = values[distances == 0][0]
        else:
            weights = 1 / distances**power
            interpolated_value = np.sum(weights * values) / np.sum(weights)
        st.success(f"✅ Interpolated value at (Lat: {target_lat}, Lon: {target_lon}) is {interpolated_value:.2f}")

# Generate grid for contouring
grid_x = np.linspace(min(x), max(x), 200)
grid_y = np.linspace(min(y), max(y), 200)
grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x, grid_y)
grid_z = np.zeros_like(grid_x_mesh)

# Compute IDW over grid
for i in range(grid_x_mesh.shape[0]):
    for j in range(grid_x_mesh.shape[1]):
        gx, gy = grid_x_mesh[i, j], grid_y_mesh[i, j]
        dists = np.sqrt((x - gx)**2 + (y - gy)**2)
        if np.any(dists == 0):
            grid_z[i, j] = values[dists == 0][0]
        else:
            w = 1 / dists**power
            grid_z[i, j] = np.sum(w * values) / np.sum(w)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
contour = ax.contourf(grid_x_mesh, grid_y_mesh, grid_z, cmap='cividis', levels=20)
scatter = ax.scatter(x, y, c=values, edgecolor='k', cmap='viridis', label='Data Points')
ax.scatter(target_x, target_y, color='red', marker='x', s=100, label='Target Location')
plt.colorbar(contour, ax=ax, label='Interpolated Value')
ax.set_title("IDW Interpolation in Projected Coordinates")
ax.set_xlabel("Easting (meters)")
ax.set_ylabel("Northing (meters)")
ax.legend()
st.pyplot(fig)
