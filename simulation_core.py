# =========================================================
# simulation_core.py
# =========================================================

import pandas as pd

from grids import make_grid, calc_temp_grid
from init_agents import init_adults
from transitions import maturation, settle, larval_supply
from stages import adult_surv, juvenile_dev, larva_dev


# step and simulate function
# -------------------------
def step(df_prev, config, grids, water_temp, air_temp, food, day, next_agent_id):
    
    shoreline_grid, elevation_grid, exposure_grid = grids
    agent_param = config["agent_params"]
    mussel_effect = config["mussel_effect"]
    tides = config["tides"]
    
    pieces = []  

    # -------------------------
    # Temperature grid
    # -------------------------
    temp_grid = calc_temp_grid(
        elevation_grid,
        air_temp,
        water_temp,
        df_prev,
        mussel_effect,
        tides
    )
    
    # -------------------------
    # Update Day and Age for existing agents
    # -------------------------
    df_prev["Day"] = day
    df_prev["Age"] = df_prev["Age"].fillna(0) + 1
    
    # -------------------------
    # Adult survival
    # -------------------------
    ddf = adult_surv(df_prev, agent_param["adult"]["adult_daily_survival"], agent_param["adult"]["adult_start_day"], day=day)
    ddf["Day"] = day
    pieces.append(ddf)

    # -------------------------
    # Juvenile → Adult maturation
    # -------------------------
    df_prev, ddf = maturation(
        df_prev,
        agent_param["maturation"]["maturation_size"]
    )
    pieces.append(ddf)

    # -------------------------
    # Juvenile survival + growth
    # -------------------------
    ddf = juvenile_dev(
        df_prev,
        agent_param["juvenile"]["survival"],
        agent_param["juvenile"]["growth"]["juvenile_daily_growth"],
        elevation_grid,
        exposure_grid,
        temp_grid,
        tides["max_exposure"]
    )
    pieces.append(ddf)

    # -------------------------
    # Settlement (Larvae → Juveniles)
    # -------------------------
    df_prev, ddf = settle(
        df_prev,
        agent_param["settlement"],
        water_temp,
        shoreline_grid,
        elevation_grid,
        tides["tidal_min"],
        tides["tidal_max"]
    )
    pieces.append(ddf)

    # -------------------------
    # Larval survival + growth + development
    # -------------------------
    ddf = larva_dev(
        df_prev,
        agent_param["larva"]["survival"],
        agent_param["larva"]["growth"],
        agent_param["settlement"]["development"],  
        water_temp,
        food
    )
    pieces.append(ddf)

    # -------------------------
    # Larval supply
    # -------------------------
    ddf, next_agent_id = larval_supply(agent_param["larval_supply"], water_temp, next_agent_id)
    ddf["Day"] = day
    pieces.append(ddf)

    # Remove empty dataframes and concat once
    pieces = [p for p in pieces if not p.empty]
    df = pd.concat(pieces, ignore_index=True)

    return df, next_agent_id

def simulate(config):

    food_df = pd.read_csv(config["food_filepath"], index_col="day")
    temp_df = pd.read_csv(config["temp_filepath"], index_col="day")

    grids = make_grid(
        config["elevation_min"],
        config["elevation_max"],
        config["shoreline_len"],
        config["technical"]["grid_resolution"],
        config["tides"]
    )

    shoreline_grid, elevation_grid, exposure_grid = grids

    # Initialize adults
    df, next_agent_id = init_adults(
        shoreline_grid, 
        elevation_grid, 
        config["initial_conditions"],
        config["tides"],
        start_id=0)

    # Store all daily outputs
    all_days = [df.copy()]  # day 0 / initial adults

    for day in range(1, config["sim_days"] + 1):

        air_temp = temp_df.loc[day, "ATMP"]
        water_temp = temp_df.loc[day, "WTMP"]
        food = food_df.loc[day, "cells"]

        # Run one day
        df, next_agent_id = step(df, config, grids, water_temp, air_temp, food, day, next_agent_id)

        # Store only the new day's output
        all_days.append(df.copy())

    # Concatenate once
    results = pd.concat(all_days, ignore_index=True)

    return results