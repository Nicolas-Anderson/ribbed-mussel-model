import numpy as np
import pandas as pd

# =========================
# maturation
# =========================

def maturation(agents_df, maturation_size, space_limitation=None):

    space_limitation = space_limitation or {}
    use_space_limit = space_limitation.get("enabled", False)
    max_adults = space_limitation.get("max_adults_per_cell")

    eligible_mask = (
        (agents_df["Stage"] == "Juvenile") &
        (agents_df["Size"] >= maturation_size)
    )

    eligible = agents_df.loc[eligible_mask].copy()

    if eligible.empty:
        return agents_df, pd.DataFrame(columns=agents_df.columns)

    # No space limitation: all eligible juveniles mature
    if not use_space_limit:
        matured = eligible.copy()
        matured["Stage"] = "Adult"
        return agents_df.loc[~eligible_mask].copy(), matured

    if max_adults is None:
        raise ValueError(
            "max_adults_per_cell must be provided when space limitation is enabled."
        )

    adult_counts = (
        agents_df[agents_df["Stage"] == "Adult"]
        .groupby(["Shore_idx", "Elev_idx"])
        .size()
    )

    mature_indices = []

    for cell, group in eligible.groupby(["Shore_idx", "Elev_idx"]):

        n_adults = adult_counts.get(cell, 0)
        vacancies = max(0, max_adults - n_adults)
        n_mature = min(len(group), vacancies)

        if n_mature > 0:
            selected = np.random.choice(
                group.index,
                size=n_mature,
                replace=False
            )
            mature_indices.extend(selected)

    matured = agents_df.loc[mature_indices].copy()
    matured["Stage"] = "Adult"

    # All maturation-eligible juveniles leave the juvenile population and those not selected because of space limitation are lost.
    agents_df = agents_df.loc[~eligible_mask].copy()

    return agents_df, matured

def settle(
    agents_df,
    settlement_params,
    water_temp,
    shoreline_grid,
    elevation_grid,
    tidal_min,
    tidal_max
):

    settlement_exp = settlement_params["settlement_exp"]
    min_size = settlement_params["min_size"]
    max_size = settlement_params["max_size"]
    settlement_probability = settlement_params["settlement_probability"]

    # -------------------------
    # Identify larvae eligible for settlement
    # -------------------------
    eligible_mask = (
        (agents_df["Stage"] == "Larval") &
        (
            ((agents_df["Development"] > 1.0) & (agents_df["Size"] > min_size)) | (agents_df["Size"] > max_size))
    )

    n_eligible = eligible_mask.sum()

    if n_eligible > 0:

        # ---------------------------------
        # Decide who successfully settles
        # ---------------------------------
        rand = np.random.rand(n_eligible)
        success_mask = rand < settlement_probability

        # Subset eligible individuals
        eligible_df = agents_df.loc[eligible_mask].copy()

        settlers_df = eligible_df.loc[success_mask].copy()
        n_settlers = len(settlers_df)

        # ---------------------------------
        # Assign settlement locations (only for successful settlers)
        # ---------------------------------
        if n_settlers > 0:

            # Sample shoreline index
            shore_idx = np.random.choice(
                np.arange(shoreline_grid.shape[0]),
                size=n_settlers
            )

            # Intertidal elevation indices
            elev_coords = elevation_grid[0, :]
            intertidal_mask = (
                (elev_coords >= tidal_min) &
                (elev_coords <= tidal_max)
            )
            intertidal_indices = np.where(intertidal_mask)[0]

            # Weighted elevation sampling
            intertidal_elevs = elev_coords[intertidal_indices]
            weights = np.exp(-settlement_exp * intertidal_elevs)
            probs = weights / weights.sum()

            elevation_idx = np.random.choice(
                intertidal_indices,
                size=n_settlers,
                p=probs
            )

            # Extract coordinates
            shoreline_vals = shoreline_grid[shore_idx, elevation_idx]
            elevation_vals = elevation_grid[shore_idx, elevation_idx]

            # Build juvenile dataframe
            settlers_df["Stage"] = "Juvenile"
            settlers_df["Development"] = 0.0
            settlers_df["Shoreline"] = shoreline_vals
            settlers_df["Elevation"] = elevation_vals
            settlers_df["Shore_idx"] = shore_idx
            settlers_df["Elev_idx"] = elevation_idx

        else:
            settlers_df = pd.DataFrame(columns=agents_df.columns)

        # ---------------------------------
        # Remove ALL eligible larvae (success + failure)
        # ---------------------------------
        agents_df = agents_df.loc[~eligible_mask]

    else:
        settlers_df = pd.DataFrame(columns=agents_df.columns)

    return agents_df, settlers_df

def larval_supply(
    larval_supply_params, 
    water_temp, 
    next_agent_id):
    
    larval_temp_min = larval_supply_params["larval_temp_min"]
    larvae_base_rate = larval_supply_params["larvae_base_rate"]

    if water_temp <= larval_temp_min or larvae_base_rate <= 0:
        return pd.DataFrame(columns=[
            "Agent_ID","Stage","Shoreline","Elevation","Age","Size","Development","Shore_idx","Elev_idx"
        ]), next_agent_id

    n_new = int(larvae_base_rate)
    new_ids = np.arange(next_agent_id, next_agent_id + n_new)
    next_agent_id += n_new

    spawned = pd.DataFrame({
        "Agent_ID": new_ids,
        "Stage": "Larval",
        "Shoreline": np.nan,
        "Elevation": np.nan,
        "Age": 0,
        "Size": 0.100,
        "Development": 0.0,
        "Shore_idx": -1,
        "Elev_idx": -1
    })

    return spawned, next_agent_id
