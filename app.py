# Save this as app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from scipy.spatial import ConvexHull
from matplotlib.path import Path

# Data
data = {...}  # Use your existing data dictionary here
df = pd.DataFrame(data)

# Sidebar input
st.sidebar.title("Input Coordinates")
target_lat = st.sidebar.number_input("Latitude", value=49.85)
target_lon = st.sidebar.number_input("Longitude", value=-102.9)

# Interpolation logic
lats = df['Lat'].values
lons = df['Long'].values
values = df['Kh'].values

points = np.column_stack((lons, lats))
hull = ConvexHull(points)
hull_path = Path(points[hull.vertices])

if not hull_path.contains_point((target_lon, target_lat)):
    st.warning("⚠️ Point is outside the convex hull. Interpolation skipped.")
    interpolated_value = None
else:
    OK = OrdinaryKriging(lons, lats, values, variogram_model='spherical', verbose=False, enable_plotting=False)
    z, ss = OK.execute('points', [target_lon], [target_lat])
    interpolated_value = z[0]
    st.success(f"✅ Interpolated value: {interpolated_value:.2f}")

# Plot
grid_lon = np.linspace(min(lons), max(lons), 500)
grid_lat = np.linspace(min(lats), max(lats), 250)
grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

OK_grid = OrdinaryKriging(lons, lats, values, variogram_model='spherical', verbose=False, enable_plotting=False)
z_grid, ss_grid = OK_grid.execute('grid', grid_lon, grid_lat)

fig, ax = plt.subplots(figsize=(10, 5))
contour = ax.contourf(grid_lon_mesh, grid_lat_mesh, z_grid, cmap='cividis')
scatter = ax.scatter(lons, lats, c=values, edgecolor='k', cmap='viridis')
ax.scatter(target_lon, target_lat, color='red', marker='x', s=100, label='Target')
plt.colorbar(contour, ax=ax, label='Interpolated Value')
ax.set_title("Kriging Interpolation")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
st.pyplot(fig)
