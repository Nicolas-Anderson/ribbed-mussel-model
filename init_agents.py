# Using the multinomial and the expected adult coefficient as the number of individuals in the landscape

import numpy as np
import pandas as pd


def init_adults(
    shoreline_grid,
    elevation_grid,
    initial_conditions,
    tides,
    start_id=0
):
    """
    Two-layer adult initialization
    ==============================
    
    LAYER 1: Density across elevation
    ---------------------------------
    Adults are first distributed among elevation bands using a deterministic
    allocation. An exponential decay function controls how abundance changes
    with elevation while preserving the total number of adults (N).
    
    LAYER 2: Aggregation along the shoreline
    ----------------------------------------
    Within each elevation band, individuals are distributed among shoreline
    cells using a Dirichlet-Multinomial distribution. This separates the
    large-scale density gradient (Layer 1) from the fine-scale spatial
    aggregation (Layer 2).
    
    For an elevation containing N_e adults distributed across S shoreline
    cells,
    
        p ~ Dirichlet(α, ..., α)
        X ~ Multinomial(N_e, p)
    
    which is equivalent to
    
        X ~ Dirichlet-Multinomial(N_e, α).
    
    The expected number and variance of individuals in a given elevation in each shoreline cell is
    
        E[X_i] = 
            N_e / S
            
        Var(X_i) =
            (N_e / S) * (1 - 1 / S) * (N_e + S * α) / (1 + S * α).
    
    The concentration parameter (α) controls the degree of aggregation:
    
        α → ∞
            Approaches a multinomial distribution with nearly uniform
            allocation among shoreline cells.
        α = 1
            Produces moderate spatial heterogeneity.
        α < 1
            Produces increasingly aggregated (clustered) distributions.
    
    The Index of Dispersion metric (ID, variance/mean ratio) is converted to the
    corresponding Dirichlet concentration parameter (α), allowing users
    to specify aggregation in biologically interpretable units while the
    model samples from the mathematically equivalent Dirichlet-Multinomial
    distribution.

    Thus, 
    α = (N_e - (S*ID)/(S-1)) / (S * ((S*ID)/(S-1) - 1))

    """

    # -------------------------------------------------
    # Parameters
    # -------------------------------------------------
    adult_density_coef = initial_conditions["adult_density_coef"]      # adults per shoreline cell at elevation = 0
    adult_density_exp = initial_conditions["adult_density_exp"]

    agg_coef = initial_conditions["aggregation_coef"]  # Index of dispersion (variance/mean of adults) starting at elevation = 0
    agg_exp = initial_conditions["aggregation_exp"]

    tidal_min = tides["tidal_min"]
    tidal_max = tides["tidal_max"]

    n_shore, n_elev = shoreline_grid.shape
    elev_coords = elevation_grid[0, :]

    intertidal_mask = (elev_coords >= tidal_min) & (elev_coords <= tidal_max)
    intertidal_indices = np.where(intertidal_mask)[0]

    agent_rows = []
    next_id = start_id

    # -------------------------------------------------
    # Density across elevation (deterministic)
    # -------------------------------------------------
    
    N_elev_int = []
    
    for elev in intertidal_indices:
    
        z = elev_coords[elev]
    
        # expected density at this elevation within a cell
        density = adult_density_coef * np.exp(-adult_density_exp * z)
    
        # expected number of adults across all shoreline cells
        expected = density * n_shore
    
        # deterministic integer abundance
        N_elev_int.append(int(round(expected)))
    
    N_elev_int = np.array(N_elev_int)
    
    # Total abundance emerges from the density function
    N_total = N_elev_int.sum()

    # -------------------------------------------------
    # Shoreline allocation with burn-in that selects the values closest to the initial conditions
    # -------------------------------------------------
    max_attempts = 1000 # number of attempts
    
    S = len(range(n_shore))  # shoreline cells per elevation
    
    for idx, elev in enumerate(intertidal_indices):

        N_e = int(N_elev_int[idx]) # ignore elevations where the number of individuals is non-positive.
        if N_e <= 0:
            continue

        z = elev_coords[elev]

        # Aggregation decreases exponentially with elevation. The desired variance-to-mean ratio (ID) is constrained to be
        # at least the multinomial baseline (ID = 1), since values below this cannot be represented by a Dirichlet-Multinomial.
        ID_target = max(1.0, agg_coef * np.exp(-agg_exp * z))

        # Baseline variance-to-mean ratio for a multinomial allocation across S shoreline cells.
        # This provides the reference point for converting the desired aggregation into a Dirichlet concentration parameter.
        ID_min = 1.0 - (1.0 / S)

        accepted = False

        for _ in range(max_attempts):
            
            #Desired aggregation expressed relative to the multinomial baseline.
            ratio = ID_target / ID_min
            
            # -------------------------------------------------
            # Regime 1: degenerate system (no inference possible)
            # -------------------------------------------------
            # With one or fewer individuals there is no meaningful partitioning among shoreline cells, so the Dirichlet-Multinomial is not identifiable. 
            #We therefore use the multinomial limit (α → ∞), resulting in a random allocation with no inferred aggregation.
            if N_e <= 1:
                alpha = 1e6  # α → ∞ limit (no aggregation structure and the distributions of the one individual is thus random)
            
            # -------------------------------------------------
            # Regime 2: identifiable system
            # -------------------------------------------------
            else:
                # A finite population cannot exhibit unlimited overdispersion. 
                # As individuals become maximally clustered, additional increases in the theoretical aggregation parameter cannot be distinguished.
                # Therefore, the requested aggregation is bounded by the maximum observable value for a sample of size N_e.
                ratio_eff = min(ratio, N_e - 1.0)
            
                # Prevent exact singularity at ratio = 1
                # (this corresponds to the pure multinomial limit)
                ratio_eff = max(ratio_eff, 1.0 + 1e-8)
            
                # Invert the analytical relatinoship between the Dirichlet-Multinomial variance and the symmetric Dirichlet concentration parameter (α).
                alpha = ((N_e - ratio_eff) / (ratio_eff - 1.0)) / S 

            # -------------------------------------------------
            # Sample
            # -------------------------------------------------
            pi = np.random.dirichlet([alpha] * S) # use the Dirichlet to determine the probability vector with length S
            counts = np.random.multinomial(N_e, pi) # use the multinomical to set the number of individuals in each cell

            # -------------------------------------------------
            # Check realized aggregations
            # -------------------------------------------------
            mean = counts.mean()
            var = counts.var(ddof=1)
            ID_realized = var / mean if mean > 0 else 0

            # tolerance
            if ID_target > 0:
                err = abs(ID_realized - ID_target) / ID_target
            else:
                err = 0

            if err < 0.01: #set the range of expected error
                accepted = True
                break

        if not accepted:
            # fallback: accept last draw
            pass

        # -------------------------------------------------
        # Create agents
        # -------------------------------------------------

        for shore in range(n_shore):

            count = int(counts[shore])
            if count <= 0:
                continue

            shoreline = shoreline_grid[shore, elev]
            elevation = elevation_grid[shore, elev]

            for _ in range(count):
                agent_rows.append({
                    "Agent_ID": next_id,
                    "Stage": "Adult",
                    "Shoreline": shoreline,
                    "Elevation": elevation,
                    "Age": np.nan,
                    "Size": np.nan,
                    "Development": np.nan,
                    "Shore_idx": shore,
                    "Elev_idx": elev,
                    "Day": 0
                })
                next_id += 1

    agents_df = pd.DataFrame(agent_rows)

    return agents_df, next_id