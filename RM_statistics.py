import numpy as np
import pandas as pd
import warnings
from scipy.optimize import curve_fit, OptimizeWarning

warnings.simplefilter("ignore", OptimizeWarning)


def compute_stats(df, config):

    def exp_func(z, a, b):
        return a * np.exp(-b * z)

    grid_resolution = config["technical"]["grid_resolution"]
    elevation_min = config["elevation_min"]

    results = []

    for stage, df_stage in df.groupby("Stage"):

        # =================================================
        # abundance 
        # =================================================
        abundance = len(df_stage)

        # =================================================
        # handle empty stage
        # =================================================
        if df_stage.empty:
            results.append({
                "stage": stage,
                "abundance": 0,
                "density_coef_model": np.nan,
                "density_exp_model": np.nan,
                "agg_coef_model": np.nan,
                "agg_exp_model": np.nan,
            })
            continue

        # =================================================
        # spatial grid
        # =================================================
        n_shore = df_stage["Shore_idx"].max() + 1
        n_elev = df_stage["Elev_idx"].max() + 1

        full_grid = pd.MultiIndex.from_product(
            [range(n_shore), range(n_elev)],
            names=["Shore_idx", "Elev_idx"]
        ).to_frame(index=False)

        obs = (
            df_stage
            .groupby(["Shore_idx", "Elev_idx"])
            .size()
            .reset_index(name="Abundance")
        )

        cell_counts = full_grid.merge(
            obs,
            on=["Shore_idx", "Elev_idx"],
            how="left"
        )

        cell_counts["Abundance"] = cell_counts["Abundance"].fillna(0)

        cell_counts["z"] = (
            elevation_min +
            (cell_counts["Elev_idx"] + 0.5) * grid_resolution
        )

        summary_df = (
            cell_counts
            .groupby("z")
            .agg(
                Realized_Mean=("Abundance", "mean"),
                Realized_Var=("Abundance", "var")
            )
            .reset_index()
        )

        summary_df["Realized_VarMean"] = (
            summary_df["Realized_Var"] /
            summary_df["Realized_Mean"]
        )

        summary_df = summary_df.replace([np.inf, -np.inf], np.nan).dropna()

        # =================================================
        # density fit
        # =================================================
        try:
            density_params, _ = curve_fit(
                exp_func,
                summary_df["z"],
                summary_df["Realized_Mean"]
            )
            density_coef_model, density_exp_model = density_params
        except Exception:
            density_coef_model = np.nan
            density_exp_model = np.nan

        # =================================================
        # aggregation fit
        # =================================================
        try:
            agg_params, _ = curve_fit(
                exp_func,
                summary_df["z"],
                summary_df["Realized_VarMean"]
            )
            agg_coef_model, agg_exp_model = agg_params
        except Exception:
            agg_coef_model = np.nan
            agg_exp_model = np.nan

        results.append({
            "stage": stage,
            "abundance": abundance,
            "density_coef_model": density_coef_model,
            "density_exp_model": density_exp_model,
            "agg_coef_model": agg_coef_model,
            "agg_exp_model": agg_exp_model,
        })

    return pd.DataFrame(results)