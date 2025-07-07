import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from scipy.spatial import ConvexHull
from matplotlib.path import Path

# Set page configuration
st.set_page_config(page_title="Kriging Interpolation App", layout="wide")

# Title
st.title("Kriging Interpolation of Kh Values")
st.markdown("This app performs Kriging interpolation based on manually defined well data.")

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
st.sidebar.header("Input Coordinates")
target_lat = st.sidebar.number_input("Latitude", value=49.85, format="%.6f")
target_lon = st.sidebar.number_input("Longitude", value=-102.9, format="%.6f")

# Extract coordinates and values
lats = df['Lat'].values
lons = df['Long'].values
values = df['Kh'].values

# Check for exact match
exact_match = df[(df['Lat'] == target_lat) & (df['Long'] == target_lon)]
if not exact_match.empty:
    matched_value = exact_match['Kh'].values[0]
    st.success(f"✅ Exact match found. Value at (Lat: {target_lat}, Lon: {target_lon}) is {matched_value}")
    interpolated_value = matched_value
else:
    # Check if point is within convex hull
    points = np.column_stack((lons, lats))
    hull = ConvexHull(points)
    hull_path = Path(points[hull.vertices])

    if not hull_path.contains_point((target_lon, target_lat)):
        st.warning("⚠️ Point is outside the convex hull. Interpolation skipped.")
        interpolated_value = None
    else:
        OK = OrdinaryKriging(
            lons, lats, values,
            variogram_model='spherical',
            verbose=False,
            enable_plotting=False
        )
        z, ss = OK.execute('points', [target_lon], [target_lat])
        interpolated_value = z[0]
        st.success(f"✅ Interpolated value at (Lat: {target_lat}, Lon: {target_lon}) is {interpolated_value:.2f}")

# Generate contour map
grid_lon = np.linspace(min(lons), max(lons), 500)
grid_lat = np.linspace(min(lats), max(lats), 250)
grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

OK_grid = OrdinaryKriging(
    lons, lats, values,
    variogram_model='spherical',
    verbose=False,
    enable_plotting=False
)
z_grid, ss_grid = OK_grid.execute('grid', grid_lon, grid_lat)

fig, ax = plt.subplots(figsize=(12, 6))
contour = ax.contourf(grid_lon_mesh, grid_lat_mesh, z_grid, cmap='cividis')
scatter = ax.scatter(lons, lats, c=values, edgecolor='k', cmap='viridis', label='Data Points')
ax.scatter(target_lon, target_lat, color='red', marker='x', s=100, label='Target Location')
plt.colorbar(contour, ax=ax, label='Interpolated Value')
ax.set_title("Kriging Interpolation with Target Location")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
st.pyplot(fig)
