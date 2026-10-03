import numpy as np
import pandas as pd

# =========================
# make_grid()
# =========================

def make_grid(
    elevation_min,
    elevation_max,
    shoreline_len,
    grid_resolution,
    tides
):
    
    
    tidal_min = tides["tidal_min"]
    tidal_max = tides["tidal_max"]
    max_exposure = tides["max_exposure"]
   
    eps = 1E-10
    
    # -------------------------
    # 1. Create the grid
    # -------------------------
    n_shoreline = int((eps + shoreline_len) // grid_resolution)
    n_elevation = int((eps + elevation_max - elevation_min) // grid_resolution)
    shoreline_coords = grid_resolution * (np.arange(n_shoreline) + 0.5)
    elevation_coords = elevation_min + grid_resolution * (np.arange(n_elevation) + 0.5)
    
    shoreline_grid, elevation_grid = np.meshgrid(
        shoreline_coords, elevation_coords, indexing="ij"
    )
    
    # -------------------------
    # 2. Compute exposure (tidal-based)
    # -------------------------
    exposure_grid = np.zeros_like(elevation_grid)

    # Linear intertidal exposure
    intertidal_mask = (elevation_grid > tidal_min) & (elevation_grid < tidal_max)
    exposure_grid[intertidal_mask] = (
        (elevation_grid[intertidal_mask] - tidal_min)
        / (tidal_max - tidal_min)
    ) * max_exposure

    # Fully exposed
    exposure_grid[elevation_grid >= tidal_max] = max_exposure

    # Below tidal_min remains 0
    return shoreline_grid, elevation_grid, exposure_grid
    
# =========================
# calc_number_grid()
# =========================

def calc_number_grid(
    agents_df, 
    grid, 
    stage
):

    df = agents_df[
        (agents_df["Stage"] == stage) &
        (agents_df["Shore_idx"].notna()) &
        (agents_df["Elev_idx"].notna())
    ]

    if df.empty:
        return np.zeros_like(grid)

    shore_idx = df["Shore_idx"].astype(int).to_numpy()
    elev_idx = df["Elev_idx"].astype(int).to_numpy()

    number_grid = np.zeros_like(grid)
    np.add.at(number_grid, (shore_idx, elev_idx), 1)

    return number_grid

# =========================
# calc_temp_grid()
# =========================

def calc_temp_grid(
    elevation_grid,
    air_temp,
    water_temp,
    agents_df,
    mussel_effect,
    tides
):

    enable_mussel_temp_effect = mussel_effect["enable_mussel_temp_effect"]
    mussel_temp_mode = mussel_effect["mussel_temp_mode"]
    threshold_temp_shift = mussel_effect["threshold_temp_shift"]
    max_cell_threshold = mussel_effect["max_cell_threshold"]
    tidal_min = tides["tidal_min"]
       
    # -------------------------
    # 1. Base temperature by elevation
    # -------------------------
    # At or below tidal minimum = water temperature
    # Above tidal minimum = air temperature
    temp_grid = np.where(
        elevation_grid > tidal_min,
        air_temp,
        water_temp
    ).astype(float)

    # -------------------------
    # 2. Apply mussel temperature effect
    # -------------------------
    if enable_mussel_temp_effect:
        direction = np.sign(water_temp - temp_grid)
           
        if mussel_temp_mode == "threshold":
            # Count adults per cell
            adult_count_grid = calc_number_grid(agents_df, temp_grid, "Adult")

            threshold = (adult_count_grid > max_cell_threshold).astype(float)
            temp_grid = temp_grid + threshold_temp_shift * direction * threshold       

        else:
            raise ValueError("Only 'threshold' mussel_temp_mode is implemented")

        # Clamp temperature between air and water
        temp_min = min(air_temp, water_temp)
        temp_max = max(air_temp, water_temp)
        temp_grid = np.clip(temp_grid, temp_min, temp_max)

    return temp_grid

