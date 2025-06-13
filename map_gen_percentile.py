import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import colors
import os
from datetime import datetime

# Configuration
DAILY_DATA_PATH = "~/Desktop/Academia Sinica/EQPred-ConvLSTM/dataset/daily_groundwater.csv"
TRIANGULATION_PATH = "~/Desktop/Academia Sinica/EQPred-ConvLSTM/dataset/Triangulation.csv"
OUTPUT_DIR = "groundwater_maps_global_percentile_scaled"
GRID_SIZE = 128
START_DATE = "2009/4/1"
END_DATE = "2023/3/31"
LOWER_PERCENTILE = 0.5    # 5th percentile for global vmin
UPPER_PERCENTILE = 99.5   # 95th percentile for global vmax

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Loading (Same as Before) ---
def standardize_id(station_id):
    if isinstance(station_id, str):
        return station_id.replace('.0', '') if '.0' in station_id else station_id
    elif isinstance(station_id, float):
        return str(int(station_id))
    else:
        return str(station_id)

# Load triangulation
tri_df = pd.read_csv(TRIANGULATION_PATH)
station_coords = {}
triangles = []
for _, row in tri_df.iterrows():
    triangle = []
    for v in ['v1', 'v2', 'v3']:
        sid = standardize_id(row[f'{v}_ID'])
        if sid not in station_coords:
            station_coords[sid] = (row[f'{v}_x'], row[f'{v}_y'])
        triangle.append(sid)
    triangles.append(triangle)

# Load groundwater data
daily_df = pd.read_csv(DAILY_DATA_PATH)
daily_df['Date'] = pd.to_datetime(daily_df[['Year', 'Month', 'Day']])
daily_df.columns = [standardize_id(c) if c not in ['Year', 'Month', 'Day', 'Date'] else c for c in daily_df.columns]

# Match station IDs
tri_stations = set(station_coords.keys())
daily_stations = set(daily_df.columns) - {'Year', 'Month', 'Day', 'Date'}
common_stations = sorted(tri_stations & daily_stations)

# Build triangulation
station_to_idx = {s: i for i, s in enumerate(common_stations)}
valid_triangles = [[station_to_idx[v] for v in tri] for tri in triangles if all(v in station_to_idx for v in tri)]
x_coords = [station_coords[s][0] for s in common_stations]
y_coords = [station_coords[s][1] for s in common_stations]
triangulation = mtri.Triangulation(x_coords, y_coords, triangles=np.array(valid_triangles))

# --- Global Percentile Calculation ---
# Collect ALL valid values across the entire date range
start_date = datetime.strptime(START_DATE, "%Y/%m/%d")
end_date = datetime.strptime(END_DATE, "%Y/%m/%d")
target_dates = daily_df[(daily_df['Date'] >= start_date) & (daily_df['Date'] <= end_date)]

all_values = []
for _, row in target_dates.iterrows():
    z = row[common_stations].values.astype(float)
    all_values.extend(z[~np.isnan(z)])  # Ignore NaN values

# Compute global percentiles
vmin, vmax = np.percentile(all_values, [LOWER_PERCENTILE, UPPER_PERCENTILE])
print(f"Global color range: [{vmin:.2f}, {vmax:.2f}] (from {LOWER_PERCENTILE}th–{UPPER_PERCENTILE}th percentiles)")
norm = colors.Normalize(vmin=vmin, vmax=vmax)  # Fixed for all maps

# --- Generate Maps with Global Scaling ---
xi = np.linspace(min(x_coords), max(x_coords), GRID_SIZE)
yi = np.linspace(min(y_coords), max(y_coords), GRID_SIZE)
grid_x, grid_y = np.meshgrid(xi, yi)

for _, row in target_dates.iterrows():
    date_str = row['Date'].strftime("%Y-%m-%d")
    try:
        z = row[common_stations].values.astype(float)
        z_mean = np.nanmean(z)
        z_filled = np.nan_to_num(z, nan=z_mean)

        # Interpolate
        interpolator = mtri.LinearTriInterpolator(triangulation, z_filled)
        grid_z = interpolator(grid_x, grid_y)
        grid_z = np.nan_to_num(grid_z, nan=z_mean)

        # Plot (using GLOBAL norm)
        plt.figure(figsize=(1, 1))
        plt.contourf(grid_x, grid_y, grid_z, levels=1000, cmap='Greys', norm=norm)
        plt.axis('off')
        plt.margins(0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save
        output_path = os.path.join(OUTPUT_DIR, f"{date_str}.png")
        plt.savefig(output_path, dpi=128, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error on {date_str}: {e}")
        continue

print("All maps saved with GLOBAL percentile scaling.")