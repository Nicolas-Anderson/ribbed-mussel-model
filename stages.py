import numpy as np
import pandas as pd

# =========================
# adult_surv()
# =========================
    
    
def adult_surv(agents_df, adult_daily_survival, adult_start_day, day=None):

    # Identify adults
    adults_mask = agents_df["Stage"] == "Adult"
    n_adults = adults_mask.sum()
    adults = agents_df[adults_mask].copy()

    # If no adults, return unchanged
    if n_adults == 0:
        return adults

    # Survival delay
    if day is not None and day < adult_start_day:
        return adults

    # Survival draw
    survive_adults = np.random.rand(n_adults) < adult_daily_survival
    adults = adults.loc[survive_adults]

    return adults

# =========================
# Juvenile_dev
# =========================

def juvenile_dev(
    agents_df,
    juvenile_survival,
    juvenile_daily_growth,
    elevation_grid,
    exposure_grid,
    temp_grid,
    max_exposure
):
    
    base_juv_surv = juvenile_survival["base_juv_surv"]
    duration_juv_surv = juvenile_survival["duration_juv_surv"]
    length_juv_surv = juvenile_survival["length_juv_surv"]
    temp_juv_surv = juvenile_survival["temp_juv_surv"]
   
    # -------------------------
    # Identify juveniles
    # -------------------------
    juveniles = agents_df[agents_df["Stage"] == "Juvenile"].copy()

    # Filter to those with valid spatial indices
    juveniles = juveniles[
        juveniles["Shore_idx"].notna() & juveniles["Elev_idx"].notna()
    ].copy()

    if juveniles.empty:
        return pd.DataFrame(columns=agents_df.columns)

    # -------------------------
    # Local environment for survival
    # -------------------------
    shoreline_idx = juveniles["Shore_idx"].astype(int).to_numpy()
    elevation_idx = juveniles["Elev_idx"].astype(int).to_numpy()

    local_exposure = exposure_grid[shoreline_idx, elevation_idx]
    local_temp = temp_grid[shoreline_idx, elevation_idx]

    
    # Logistic survival model
    linear_pred = (
        base_juv_surv
        + duration_juv_surv * local_exposure
        + length_juv_surv * juveniles["Size"].values
        + temp_juv_surv * local_temp
    ).astype(float)

    survival_probs = 1 / (1 + np.exp(-linear_pred))
    
    # Survival draw
    survive_juveniles = np.random.rand(len(juveniles)) < survival_probs

    # Keep only surviving juveniles
    juveniles = juveniles.loc[survive_juveniles].copy()

    # -------------------------
    # Juvenile growth (submergence-dependent)
    # -------------------------
    if not juveniles.empty:
        # Recompute indices for survivors
        shoreline_idx = juveniles["Shore_idx"].astype(int).to_numpy()
        elevation_idx = juveniles["Elev_idx"].astype(int).to_numpy()

        # Local exposure of survivors
        local_exposure = exposure_grid[shoreline_idx, elevation_idx]

        # Submergence proportion (0 = fully emerged, 1 = fully submerged)
        submergence = 1.0 - (local_exposure / max_exposure)
        submergence = np.clip(submergence, 0.0, 1.0)

        # Growth scaled by submergence
        growth_increment = juvenile_daily_growth * submergence
        juveniles["Size"] += growth_increment

    return juveniles
    
# =========================
# Larva_dev
# =========================

def larva_dev(
    agents_df,
    larval_survival,
    larval_growth,
    larval_development,
    water_temp,
    food
):
    # Survival parameters
    mu0 = larval_survival["mu0"]
    b = larval_survival["b"]
    c = larval_survival["c"]
    K = larval_survival["K"]
    
    # Growth parameters
    larval_growth_base = larval_growth["larval_growth_base"]
    watertemp_coef = larval_growth["watertemp_coef"]
    food_coef = larval_growth["food_coef"]
        
    # Development parameters
    dev_coef = larval_development["development_coef"]
    dev_exp = larval_development["development_exp"]

    # -------------------------
    # Identify larvae
    # -------------------------
    larvae_mask = agents_df["Stage"] == "Larval"
    n_larvae = larvae_mask.sum()

    if n_larvae == 0:
        return pd.DataFrame(columns=agents_df.columns)

    larvae = agents_df.loc[larvae_mask].copy()
    
    # -------------------------
    # Larval development
    # -------------------------
    dev_increment = (dev_coef * np.exp(dev_exp * water_temp))
    larvae["Development"] += dev_increment
    
    # -------------------------
    # Larval survival
    # -------------------------
    WaterTemp_Optimal = b / (2 * c)
    mu_T = np.exp(c * (water_temp - WaterTemp_Optimal) ** 2)
    mu_F = 1 / (1 + food / (K))
    survival_prob = 1 - (mu0 * mu_T * mu_F)

    survive_larvae = np.random.rand(n_larvae) < survival_prob
    larvae = larvae.loc[survive_larvae].copy()

    # -------------------------
    # Larval growth
    # -------------------------
    if not larvae.empty:
        growth_increment = (
            larval_growth_base
            + watertemp_coef * water_temp
            + food_coef * food
        ) 

        larvae["Size"] += growth_increment

    return larvae
