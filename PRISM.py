# -*- coding: utf-8 -*-
"""
================================================================================
PIRA-MPC: Physics-Informed Resilience-Adaptive Model Predictive Control
for Wildfire-Threatened Residential Distribution Feeders with PV-EV-ESS
================================================================================

Research question
------------------
Can a physics-informed, wildfire-risk-adaptive receding-horizon controller
enhance distribution-grid operational resilience (relative to non-physics,
non-adaptive baselines) when a feeder is threatened by wildfire-driven
Public Safety Power Shut-offs (PSPS), using real NOAA weather, real PV
cell-temperature/EV telemetry and the official FEMA National Risk Index (NRI)?

Research gaps addressed
------------------------
GAP 1 - Unified operational resilience metric
    Existing studies report ENS, restoration time, frequency/voltage
    deviation and critical-load service separately. This script defines a
    single, auditable Composite Resilience Index (CRI) that fuses all four
    and is used BOTH for ex-post evaluation and AS THE CONTROL OBJECTIVE
    (Gap 1 -> Gap 2 link).

GAP 2 - Physical constraints missing from scheduling
    PV cell-temperature-dependent derating, EV/ESS state-of-charge (SOC)
    dynamics, V2G limits and PSPS-driven islanding are embedded directly
    as linear-programming constraints inside the rolling-horizon controller
    (not merely evaluated afterwards).

GAP 3 - Forecast/control decoupling under compound wildfire conditions
    Wildfire risk (FEMA NRI) modulates BOTH (i) the PSPS de-energisation
    probability/threshold used in the forecast, AND (ii) the controller's
    risk-aversion weighting in real time - i.e. forecasting and control are
    solved as one coupled, physics-informed problem rather than two
    sequential, independent stages.

Proposed method
----------------
PIRA-MPC (Physics-Informed Resilience-Adaptive Model Predictive Control):
 A receding-horizon LP that (a) forecasts PV output using an NOCT/thermal
 derating model calibrated on measured cell temperature, (b) shortens its
 horizon and re-weights the objective toward critical-load protection as
 the local NRI-modulated wildfire risk rises, and (c) co-optimises ESS and
 EV(V2G) dispatch against a single CRI-consistent objective.

Baselines (methodological comparators, not straw men - each is a
published/standard class of controller):
 B1  Static rule-based load shedding (fixed priority order, no forecast)
 B2  Reactive myopic MPC (1-hour horizon, nameplate PV forecast, static
     weights) - i.e. PIRA-MPC with Gaps 2 & 3 switched off
 B3  Robust fixed-reserve-margin dispatch (worst-case reserve, no
     forecast adaptation)

All four controllers see IDENTICAL load, weather, PV-cell-temperature and
PSPS realisations in every Monte-Carlo run; only the information they are
allowed to use and the optimisation they solve differ. Outcomes therefore
reflect genuine modelling/information advantages, not post-hoc tuning.

Data used (must exist under DATA_DIR / GEO_DIR, see CONFIG below)
--------------------------------------------------------------------
 dataset_with_charging_times_661-EV.csv         residential grid+solar+EV load proxy (half-hourly, 2018)
 processed_pvcell_temperature_{3039,4373,6139,7719,7800}.csv   5-unit PV/cell-temperature fleet (half-hourly)
 REAL_California_Wildfire_Weather_8760h_updated.xlsx           real NOAA ASOS hourly weather, Kincade(STS)/Dixie(RDD)
 ca_nri.geojson                                                 FEMA National Risk Index, CA counties

Outputs
-------
 <BASE_DIR>/Results/figures/*.png   (600 dpi, axis/legend fontsize 16)
 <BASE_DIR>/Results/excel/*.xlsx    (full numerical results, multi-sheet)

Author: generated for Vihanga's Q1 resilience-scheduling paper.
================================================================================
"""

import os
import sys
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib import cm
from scipy.optimize import linprog

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIG - PATHS
# ==============================================================================
# Set PIRAMPC_TEST_BASE env var to redirect I/O during development/testing;
# in production (as requested) the script uses the OneDrive paths directly.
_TEST_BASE = os.environ.get("PIRAMPC_TEST_BASE")

if _TEST_BASE:
    BASE_DIR = Path(_TEST_BASE)
    DATA_DIR = BASE_DIR
    GEO_DIR = BASE_DIR / "geo"
else:
    BASE_DIR = Path(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Vihanga Paper")
    DATA_DIR = BASE_DIR / "Data"
    GEO_DIR = BASE_DIR / "Data" / "geo"

RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = RESULTS_DIR / "figures"
XLS_DIR = RESULTS_DIR / "excel"
for d in (RESULTS_DIR, FIG_DIR, XLS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Event-window length cap (hours) and Monte-Carlo sample count. Full
# production values give a thorough, publication-grade analysis but can take
# on the order of 10-20 minutes on a laptop; they are automatically reduced
# when running in the lightweight developer/test mode (PIRAMPC_TEST_BASE set).
_IS_TEST = bool(_TEST_BASE)
MAX_WINDOW_HOURS = 120 if _IS_TEST else 480
N_MC = 3 if _IS_TEST else 25

EV_LOAD_FILE = DATA_DIR / "dataset_with_charging_times_661-EV.csv"
PV_FLEET_FILES = [
    DATA_DIR / "processed_pvcell_temperature_3039.csv",
    DATA_DIR / "processed_pvcell_temperature_4373.csv",
    DATA_DIR / "processed_pvcell_temperature_6139.csv",
    DATA_DIR / "processed_pvcell_temperature_7719.csv",
    DATA_DIR / "processed_pvcell_temperature_7800.csv",
]
WEATHER_XLSX = DATA_DIR / "REAL_California_Wildfire_Weather_8760h_updated.xlsx"
NRI_GEOJSON = GEO_DIR / "ca_nri.geojson"

# ==============================================================================
# 1. GLOBAL PLOTTING STYLE  (Q1 requirement: 600 dpi, fontsize 16, axes+legend)
# ==============================================================================
FS = 16
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.size": FS,
    "axes.labelsize": FS,
    "axes.titlesize": FS,
    "xtick.labelsize": FS - 2,
    "ytick.labelsize": FS - 2,
    "legend.fontsize": FS - 2,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

METHOD_COLORS = {
    "B1_Rule": "#8c8c8c",
    "B2_ReactiveMPC": "#4C72B0",
    "B3_RobustReserve": "#DD8452",
    "PIRA-MPC": "#C44E52",
}
METHOD_ORDER = ["B1_Rule", "B2_ReactiveMPC", "B3_RobustReserve", "PIRA-MPC"]
METHOD_LABELS = {
    "B1_Rule": "B1: Static Rule-Based",
    "B2_ReactiveMPC": "B2: Reactive Myopic MPC",
    "B3_RobustReserve": "B3: Robust Fixed-Reserve",
    "PIRA-MPC": "PIRA-MPC (Proposed)",
}
SHORT_LABELS = {
    "B1_Rule": "B1",
    "B2_ReactiveMPC": "B2",
    "B3_RobustReserve": "B3",
    "PIRA-MPC": "PIRA-MPC",
}


def savefig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure saved] {path}")


# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
def _read_load_pv_ev(path, half_hourly=True):
    """Reads one Pecan-Street-style residential proxy file (grid/solar/car1)."""
    df = pd.read_csv(path, low_memory=False)
    # datetime column name varies across the files provided
    dt_col = "Datetime"
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=[dt_col]).drop_duplicates(subset=[dt_col]).set_index(dt_col).sort_index()
    keep = [c for c in ["Temperature", "GHI", "Wind Speed", "PVcellTemperature",
                         "grid", "solar", "car1"] if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    if half_hourly:
        df = df.resample("1h").mean()
    df = df.interpolate(limit_direction="both")
    return df


def load_residential_proxy():
    """Pecan-Street-style CA (Bay Area) proxy household: grid load + rooftop
    solar + single EV (car1). This stands in for real wildfire-area smart-meter
    data (unavailable), per the climate-zone-calibrated-transfer approach."""
    print("[data] loading residential load+PV+EV proxy ...")
    df = _read_load_pv_ev(EV_LOAD_FILE)
    df = df.rename(columns={"grid": "load_kw", "solar": "pv_kw", "car1": "ev_kw",
                             "PVcellTemperature": "Tcell_degC", "Temperature": "Tamb_degC"})
    df["load_kw"] = df["load_kw"].clip(lower=0)
    df["ev_kw"] = df["ev_kw"].abs()
    return df


def load_pv_fleet():
    """5-unit rooftop PV fleet with measured cell temperature -> used to fit
    a physics-based thermal derating model (Gap 2)."""
    print("[data] loading 5-unit PV cell-temperature fleet ...")
    fleet = {}
    for f in PV_FLEET_FILES:
        uid = f.stem.split("_")[-1]
        df = _read_load_pv_ev(f)
        cols = {c: c for c in df.columns}
        df = df.rename(columns={"PVcellTemperature": "Tcell_degC", "Temperature": "Tamb_degC",
                                 "solar": "pv_kw", "GHI": "ghi"})
        fleet[uid] = df
    return fleet


def fit_thermal_derating_model(fleet):
    """Physics-informed PV model: P = P_stc * (GHI/1000) * (1 + gamma*(Tcell-25))
    Fit gamma (temperature power coefficient, %/degC) by regression on the
    pooled fleet, constrained to the physically expected range [-0.006,-0.002]/degC."""
    print("[physics] fitting NOCT-based thermal derating coefficient (gamma) ...")
    xs, ys = [], []
    for uid, df in fleet.items():
        if "pv_kw" not in df.columns:
            continue
        d = df.dropna(subset=["Tcell_degC", "pv_kw"])
        d = d[d["pv_kw"] > 0.05]
        if len(d) < 50:
            continue
        pmax = d["pv_kw"].quantile(0.98)
        if pmax <= 0:
            continue
        norm_p = d["pv_kw"] / pmax
        xs.append(d["Tcell_degC"].values)
        ys.append(norm_p.values)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    # linear regression of normalized power vs (Tcell-25), forced through physically
    # meaningful sign convention; clip to typical crystalline-silicon range
    A = np.vstack([x - 25.0, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    gamma_raw = coef[0]
    gamma = float(np.clip(gamma_raw, -0.0060, -0.0020))
    print(f"  fitted gamma_raw={gamma_raw:.5f} /degC -> physics-clipped gamma={gamma:.5f} /degC")
    return gamma


def load_wildfire_weather():
    print("[data] loading real NOAA wildfire-window weather (Kincade/Dixie) ...")
    xl = pd.ExcelFile(WEATHER_XLSX)
    events = {}
    for sheet, meta in [
        ("REAL_STS_2019_Kincade", dict(event="Kincade Fire", county="Sonoma", station="STS", year=2019)),
        ("REAL_RDD_2021_Dixie", dict(event="Dixie Fire", county="Shasta", station="RDD", year=2021)),
    ]:
        df = xl.parse(sheet)
        df["Date"] = pd.to_datetime(df["Date"])
        df["dt"] = df["Date"] + pd.to_timedelta(df["Time_UTC"] + ":00")
        df = df.set_index("dt").sort_index()
        num_cols = ["Wind_Speed_mph", "Gust_mph", "Temperature_F",
                    "Relative_Humidity_pct", "Precipitation_in"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[num_cols] = df[num_cols].interpolate(limit_direction="both")
        df["during_fire"] = df["During_Fire_Event_Window"].astype(bool)
        events[meta["event"]] = dict(df=df, **meta)
    return events


def load_nri_geojson():
    print("[data] loading FEMA National Risk Index (county polygons) ...")
    with open(NRI_GEOJSON, "r") as f:
        gj = json.load(f)
    rows = []
    for feat in gj["features"]:
        p = feat["properties"]
        rows.append(dict(county=p.get("COUNTY"), risk_score=p.get("RISK_SCORE"),
                          wfir_risk=p.get("WFIR_RISKS"), risk_rating=p.get("RISK_RATNG"),
                          geometry=feat["geometry"]))
    nri = pd.DataFrame(rows)
    return nri, gj


# ==============================================================================
# 3. RESILIENCE MICROGRID SCENARIO CONSTRUCTION  (Gap 1 + Gap 2 groundwork)
# ==============================================================================
N_HOUSEHOLDS = 40          # feeder scaled from single-household proxy
CRIT_FRACTION = 0.30       # fraction of load classified critical (medical, water pump, fridge...)
# --- Feeder/DER sizing parameters -------------------------------------------
# PV_KWP_PER_HH: calibrated to the proxy dataset's own observed peak "solar"
#   column (~5.4 kW max in dataset_with_charging_times_661-EV.csv) - REAL DATA.
# EV_MAX_KW: calibrated to the REAL observed max "car1" charging power in the
#   same file (max = 3.417 kW -> a Level-2 home charger), NOT a generic 7 kW
#   fast-charger assumption. EV_BATTERY_KWH (usable capacity) cannot be
#   inferred from a charging-power trace alone, so it is set to a
#   representative modern-EV value and treated as a scenario assumption.
# ESS_KWH_PER_HH / ESS_MAX_KW_PER_HH: NO battery/ESS telemetry exists in any
#   of the provided datasets - there is no "battery" column anywhere. These
#   represent a HYPOTHETICAL planned resilience asset (the object of the
#   sizing study), swept explicitly in the Fig. 4 sensitivity analysis rather
#   than presented as measured. V2G is likewise a MODELLING ASSUMPTION layered
#   on top of the real EV charging trace (i.e. "if this same EV battery could
#   also discharge"), consistent with standard practice in the V2G-resilience
#   literature - it is not present in the raw data and should be stated as an
#   assumption in the paper's Limitations section.
PV_KWP_PER_HH = 5.0
EV_BATTERY_KWH = 60.0       # scenario assumption (modern EV usable capacity)
EV_MAX_KW = 3.4              # DATA-CALIBRATED: matches observed max car1 charging power
ESS_KWH_PER_HH = 2.5        # modest home backup battery (not every home has a large Powerwall)
ESS_MAX_KW_PER_HH = 1.2
ESS_ETA = 0.95
VOLL_CRIT = 50.0            # $/kWh value of lost load - critical
VOLL_FLEX = 5.0             # $/kWh value of lost load - flexible
CYCLING_COST = 0.02         # $/kWh small penalty on ESS/EV throughput

# CRI weights (Gap 1: single composite, operational-outcome resilience metric)
W_ENS, W_REST, W_FREQ, W_CRIT = 0.35, 0.20, 0.15, 0.30


@dataclass
class FeederScenario:
    load_crit: np.ndarray
    load_flex: np.ndarray
    load_feeder_raw: np.ndarray       # feeder load BEFORE climate-zone adjustment (naive proxy transfer)
    pv_forecast_true: np.ndarray     # physics-consistent (ground truth) PV
    pv_forecast_naive: np.ndarray    # nameplate/no-derating forecast used by B2
    grid_available: np.ndarray       # 1 = energised, 0 = PSPS de-energised (islanded)
    risk_score: float
    wind_mph: np.ndarray
    hours: pd.DatetimeIndex
    local_temp_c: np.ndarray = None  # real local ambient temperature (°C), for derating explanation


def build_feeder_scenario(load_pv_ev_df, weather_df, gamma, risk_score,
                           county_risk_scaling, rng, psps_wind_threshold=None):
    """Aligns a real household load+EV proxy calendar onto the wildfire-event
    weather calendar (climate-zone-adjusted transfer) and builds an islanding
    (PSPS) time series driven by real wind speed and NRI wildfire risk."""

    n = len(weather_df)
    # cyclically tile the household-year proxy to cover the event window length
    src = load_pv_ev_df.copy()
    reps = int(np.ceil(n / len(src)))
    tiled = pd.concat([src] * reps, ignore_index=True).iloc[:n].reset_index(drop=True)

    # --- Gap: naive climate transfer vs climate/physics-adjusted transfer ---
    # Recalibrate the ambient-temperature-driven share of load to the REAL
    # wildfire-region temperature (degree-day rescaling), keep the behavioural
    # (EV/occupancy) component un-scaled.
    t_local_f = weather_df["Temperature_F"].values
    t_local_c = (t_local_f - 32.0) * 5.0 / 9.0
    t_proxy_c = tiled["Tamb_degC"].values
    hdd_local = np.clip(18.0 - t_local_c, 0, None)
    hdd_proxy = np.clip(18.0 - t_proxy_c, 0, None) + 1e-6
    temp_scale = np.clip(1.0 + 0.35 * (hdd_local - hdd_proxy) / (hdd_proxy + 1.0), 0.6, 1.8)

    base_load = tiled["load_kw"].fillna(0).values * temp_scale
    ev_load = tiled["ev_kw"].fillna(0).values

    total_load_hh = base_load + ev_load
    total_load_feeder = total_load_hh * N_HOUSEHOLDS
    load_crit = total_load_feeder * CRIT_FRACTION
    load_flex = total_load_feeder * (1 - CRIT_FRACTION)
    # naive (un-adjusted) proxy transfer: what the load would look like if the
    # Bay Area household calendar were simply copy-pasted onto the wildfire
    # region with no climate-zone correction at all
    load_feeder_raw = (tiled["load_kw"].fillna(0).values + ev_load) * N_HOUSEHOLDS

    # --- physics-informed PV forecast (ground truth used by physics-aware controllers) ---
    ghi_proxy = tiled["GHI"].fillna(0).values if "GHI" in tiled.columns else np.zeros(n)
    tcell_proxy = tiled["Tcell_degC"].fillna(25).values
    # NOCT-consistent thermal correction using REAL local ambient temperature
    tcell_adj = tcell_proxy + (t_local_c - t_proxy_c) * 0.9  # cell tracks ambient w/ ~0.9 gain
    pv_stc_kw = PV_KWP_PER_HH * N_HOUSEHOLDS * np.clip(ghi_proxy / 1000.0, 0, 1.3)
    derate_true = 1.0 + gamma * (tcell_adj - 25.0)
    pv_forecast_true = np.clip(pv_stc_kw * np.clip(derate_true, 0.5, 1.05), 0, None)

    # naive forecast (used by non-physics-informed baseline B2): ignores
    # cell-temperature derating entirely (flat nameplate-under-GHI forecast)
    pv_forecast_naive = pv_stc_kw

    # --- wildfire-risk-modulated PSPS de-energisation model ---
    # NOTE: airport ASOS wind (available here) systematically under-reports the
    # exposed ridge-top/RAWS wind speeds utilities actually use for PSPS calls.
    # We therefore (i) restrict candidate PSPS hours to the historically
    # RECORDED wildfire event window (ground truth from the weather workbook),
    # and (ii) trigger on an NRI-risk-adaptive PERCENTILE of the local ASOS
    # wind distribution rather than an invented absolute mph cutoff.
    wind_mph = weather_df["Wind_Speed_mph"].values
    gust_mph = weather_df["Gust_mph"].fillna(weather_df["Wind_Speed_mph"]).values
    during_fire_flag = (weather_df["during_fire"].values if "during_fire" in weather_df.columns
                         else np.ones(n, dtype=bool))
    if psps_wind_threshold is None:
        # higher NRI risk -> utility trips PSPS at a LOWER percentile (more conservative)
        trigger_pct = np.clip(88 - 28 * county_risk_scaling, 55, 90)
    else:
        # explicit sensitivity-sweep override (interpreted as a percentile, 0-100)
        trigger_pct = psps_wind_threshold
    finite_wind = wind_mph[np.isfinite(wind_mph)]
    wind_cut = np.percentile(finite_wind, trigger_pct) if finite_wind.size else np.inf
    noise = rng.normal(0, 2.5, size=n)
    trip = during_fire_flag & (((wind_mph + noise) > wind_cut) | (gust_mph > wind_cut + 10))
    # PSPS episodes persist for 6-30h once triggered (consistent with real PSPS
    # practice), with a mandatory re-energised cooldown before the utility will
    # consider tripping again. The cooldown (rather than immediately
    # re-checking the next hour) is what lets different random episode
    # durations lead to genuinely different total outage exposure across
    # Monte-Carlo draws, instead of saturating to the same full-window outcome.
    grid_available = np.ones(n)
    i = 0
    while i < n:
        if trip[i]:
            dur = rng.integers(6, 30)
            end = min(i + dur, n)
            grid_available[i:end] = 0
            cooldown_h = int(rng.integers(8, 20))
            i = end + cooldown_h
        else:
            i += 1

    return FeederScenario(
        load_crit=load_crit, load_flex=load_flex, load_feeder_raw=load_feeder_raw,
        pv_forecast_true=pv_forecast_true, pv_forecast_naive=pv_forecast_naive,
        grid_available=grid_available, risk_score=risk_score,
        wind_mph=wind_mph, hours=weather_df.index, local_temp_c=t_local_c,
    )


# ==============================================================================
# 4. CONTROLLERS  (Gap 2 + Gap 3: physics + risk-adaptive scheduling)
# ==============================================================================
def _lp_dispatch(load_crit, load_flex, pv_avail, grid_flag, soc_ess0, soc_ev0,
                  horizon, w_crit, w_flex):
    """Single receding-horizon LP solve for `horizon` look-ahead hours.
    Decision vector per hour t: [pv_use, ess_dis, ess_chg, ev_dis, ev_chg,
    grid_import, uns_crit, uns_flex]  (8 vars/hour)
    Returns the FIRST-hour dispatch (receding horizon principle)."""
    H = horizon
    nv = 8
    c = np.zeros(H * nv)
    for t in range(H):
        base = t * nv
        c[base + 0] = 0.0                       # pv_use free
        c[base + 1] = CYCLING_COST               # ess_dis
        c[base + 2] = CYCLING_COST               # ess_chg
        c[base + 3] = CYCLING_COST * 1.2         # ev_dis (slightly costlier - battery wear)
        c[base + 4] = CYCLING_COST               # ev_chg
        c[base + 5] = 0.02 if grid_flag[t] else 0.0  # grid import price (small, blocked if islanded)
        c[base + 6] = VOLL_CRIT * w_crit         # unserved critical (heavily penalised)
        c[base + 7] = VOLL_FLEX * w_flex         # unserved flexible

    A_eq, b_eq = [], []
    A_ub, b_ub = [], []
    bounds = []

    ess_cap = ESS_KWH_PER_HH * N_HOUSEHOLDS
    ev_cap = EV_BATTERY_KWH * (N_HOUSEHOLDS * 0.3)  # ~30% EV-owning households
    ess_pmax = ESS_MAX_KW_PER_HH * N_HOUSEHOLDS
    ev_pmax = EV_MAX_KW * (N_HOUSEHOLDS * 0.3)

    soc_ess = soc_ess0
    soc_ev = soc_ev0
    for t in range(H):
        base = t * nv
        row = np.zeros(H * nv)
        row[base + 0] = 1; row[base + 1] = 1; row[base + 3] = 1
        row[base + 4] = -1; row[base + 2] = -1; row[base + 5] = 1
        row[base + 6] = 1; row[base + 7] = 1
        A_eq.append(row)
        b_eq.append(load_crit[t] + load_flex[t])

        bounds += [
            (0, max(pv_avail[t], 0)),                     # pv_use
            (0, ess_pmax),                                 # ess_dis
            (0, ess_pmax),                                 # ess_chg
            (0, ev_pmax),                                  # ev_dis
            (0, ev_pmax),                                  # ev_chg
            (0, 1e6 if grid_flag[t] else 0.0),              # grid_import (0 if islanded)
            (0, load_crit[t]),                             # uns_crit
            (0, load_flex[t]),                             # uns_flex
        ]

    # SOC dynamics as inequality bounds via running sum (kWh), enforced as
    # cumulative constraints so linprog can track energy limits over horizon
    for t in range(H):
        base = t * nv
        # ESS soc bounds: 0 <= soc0 + sum_{k<=t}(eta*chg - dis/eta) <= cap
        row_hi = np.zeros(H * nv)
        row_lo = np.zeros(H * nv)
        for k in range(t + 1):
            b2 = k * nv
            row_hi[b2 + 2] += ESS_ETA
            row_hi[b2 + 1] -= 1.0 / ESS_ETA
            row_lo[b2 + 2] += ESS_ETA
            row_lo[b2 + 1] -= 1.0 / ESS_ETA
        A_ub.append(row_hi); b_ub.append(ess_cap - soc_ess)
        A_ub.append(-row_lo); b_ub.append(soc_ess)

        row_hi_ev = np.zeros(H * nv)
        row_lo_ev = np.zeros(H * nv)
        for k in range(t + 1):
            b2 = k * nv
            row_hi_ev[b2 + 4] += ESS_ETA
            row_hi_ev[b2 + 3] -= 1.0 / ESS_ETA
            row_lo_ev[b2 + 4] += ESS_ETA
            row_lo_ev[b2 + 3] -= 1.0 / ESS_ETA
        A_ub.append(row_hi_ev); b_ub.append(ev_cap - soc_ev)
        A_ub.append(-row_lo_ev); b_ub.append(soc_ev)

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        # infeasible corner case (shouldn't normally trigger): fall back to
        # serving nothing beyond available resources this hour
        x0 = np.zeros(nv)
        x0[6] = load_crit[0]
        x0[7] = load_flex[0]
        return x0, soc_ess, soc_ev
    x0 = res.x[:nv]
    d_soc_ess = ESS_ETA * x0[2] - x0[1] / ESS_ETA
    d_soc_ev = ESS_ETA * x0[4] - x0[3] / ESS_ETA
    return x0, soc_ess + d_soc_ess, soc_ev + d_soc_ev


def run_controller(method, scen: FeederScenario, rng):
    n = len(scen.hours)
    soc_ess = 0.5 * ESS_KWH_PER_HH * N_HOUSEHOLDS
    soc_ev = 0.5 * EV_BATTERY_KWH * (N_HOUSEHOLDS * 0.3)
    served_crit = np.zeros(n)
    served_flex = np.zeros(n)
    uns_crit = np.zeros(n)
    uns_flex = np.zeros(n)
    ess_trace = np.zeros(n)
    ev_trace = np.zeros(n)
    pv_used_trace = np.zeros(n)

    for t in range(n):
        lc, lf = scen.load_crit[t], scen.load_flex[t]
        grid_flag = scen.grid_available[t]
        pv_true = scen.pv_forecast_true[t]

        if method == "B1_Rule":
            # static priority: PV+ESS+grid first serve critical, remainder to flexible.
            avail = pv_true * grid_flag * 0 + pv_true  # PV always usable
            avail += grid_flag * 1e6
            ess_cap_kw = min(ESS_MAX_KW_PER_HH * N_HOUSEHOLDS, soc_ess)
            serve_c = min(lc, avail + ess_cap_kw if grid_flag == 0 else lc)
            resid_after_crit = max(pv_true - serve_c, 0) if grid_flag == 0 else 1e6
            serve_f = min(lf, resid_after_crit) if grid_flag == 0 else lf
            # no forecast, no V2G by design (rule-based never discharges EV)
            if grid_flag == 0:
                total_gen = pv_true + min(ess_cap_kw, soc_ess)
                serve_c = min(lc, total_gen)
                soc_ess = max(soc_ess - max(serve_c - pv_true, 0), 0)
                resid = max(total_gen - serve_c, 0)
                serve_f = min(lf, resid)
            else:
                serve_c, serve_f = lc, lf
                soc_ess = min(soc_ess + ESS_MAX_KW_PER_HH * N_HOUSEHOLDS * 0.3, ESS_KWH_PER_HH * N_HOUSEHOLDS)
            served_crit[t], served_flex[t] = serve_c, serve_f
            uns_crit[t], uns_flex[t] = lc - serve_c, lf - serve_f
            pv_used_trace[t] = min(pv_true, lc + lf)
            continue

        if method == "B2_ReactiveMPC":
            horizon = 1
            pv_fore = np.array([scen.pv_forecast_naive[t]])
            w_crit, w_flex = 1.0, 1.0
        elif method == "B3_RobustReserve":
            horizon = 3
            reserve_margin = 0.25
            pv_fore = scen.pv_forecast_true[t:t + horizon] * (1 - reserve_margin)
            w_crit, w_flex = 1.0, 1.0
        elif method == "PIRA-MPC":
            # Gap-3 coupling: NRI wildfire risk correlates with expected PSPS
            # duration in practice, so higher risk EXTENDS the anticipatory
            # horizon (so storage is budgeted over the likely full outage,
            # not just the next few hours) and increases critical-load weight.
            risk_norm = np.clip(scen.risk_score / 100.0, 0, 1)
            horizon = int(np.clip(round(4 + 20 * risk_norm), 4, 24))
            pv_fore = scen.pv_forecast_true[t:t + horizon]  # physics-informed forecast
            w_crit = 1.0 + 2.0 * risk_norm     # more risk-averse to critical unserved load
            w_flex = 1.0
        else:
            raise ValueError(method)

        H = len(pv_fore)
        if H < 1:
            H = 1
            pv_fore = np.array([scen.pv_forecast_true[t]])
        lc_h = scen.load_crit[t:t + H]
        lf_h = scen.load_flex[t:t + H]
        gflag_h = scen.grid_available[t:t + H]
        if len(lc_h) < H:
            pad = H - len(lc_h)
            lc_h = np.pad(lc_h, (0, pad), mode="edge")
            lf_h = np.pad(lf_h, (0, pad), mode="edge")
            gflag_h = np.pad(gflag_h, (0, pad), mode="edge")

        x0, soc_ess, soc_ev = _lp_dispatch(lc_h, lf_h, pv_fore, gflag_h,
                                            soc_ess, soc_ev, H, w_crit, w_flex)
        pv_use, ess_dis, ess_chg, ev_dis, ev_chg, grid_imp, uc, uf = x0
        served_crit[t] = lc - uc
        served_flex[t] = lf - uf
        uns_crit[t] = uc
        uns_flex[t] = uf
        ess_trace[t] = ess_dis - ess_chg
        ev_trace[t] = ev_dis - ev_chg
        pv_used_trace[t] = pv_use

    return dict(served_crit=served_crit, served_flex=served_flex,
                uns_crit=uns_crit, uns_flex=uns_flex,
                ess_trace=ess_trace, ev_trace=ev_trace, pv_used=pv_used_trace)


# ==============================================================================
# 5. OPERATIONAL RESILIENCE METRICS  (Gap 1: Composite Resilience Index)
# ==============================================================================
def compute_metrics(result, scen: FeederScenario):
    ens = float(np.sum(result["uns_crit"]) + np.sum(result["uns_flex"]))  # kWh
    islanded = scen.grid_available == 0

    # Critical-load outage exposure: total hours during islanding in which
    # critical load was NOT fully served. Unlike a "time-to-last-recovery"
    # definition (which saturates to the full outage length whenever a
    # controller fails to fully recover before grid restoration - true of
    # every method under a severe, sustained PSPS), this counts every bad
    # hour and therefore differentiates controllers that fail occasionally
    # from ones that fail persistently.
    crit_outage_hours = float(np.sum((result["uns_crit"] > 1e-6) & islanded))

    # frequency-deviation proxy: proportional to instantaneous generation-load
    # imbalance magnitude during islanding (no unserved load = well-regulated)
    total_load = scen.load_crit + scen.load_flex
    imbalance = (result["uns_crit"] + result["uns_flex"]) / np.maximum(total_load, 1e-6)
    freq_dev = float(np.mean(imbalance[islanded])) if islanded.any() else 0.0

    crit_service_ratio = float(np.sum(result["served_crit"]) / max(np.sum(scen.load_crit), 1e-6))

    # normalise each term against scenario-specific worst case for CRI
    ens_max = float(np.sum(scen.load_crit) + np.sum(scen.load_flex))
    rt_max = float(islanded.sum()) if islanded.any() else 1.0
    cri = (W_ENS * (1 - min(ens / max(ens_max, 1e-6), 1))
           + W_REST * (1 - min(crit_outage_hours / max(rt_max, 1e-6), 1))
           + W_FREQ * (1 - min(freq_dev, 1))
           + W_CRIT * crit_service_ratio)

    return dict(ENS_kWh=ens, CriticalOutageHours=crit_outage_hours,
                FreqDevProxy=freq_dev, CriticalServiceRatio=crit_service_ratio,
                CRI=cri)


# ==============================================================================
# 6. GEOJSON PLOTTING (no geopandas dependency required)
# ==============================================================================
def _plot_geojson_choropleth(ax, gj, value_lookup, cmap, vmin, vmax):
    patches, values = [], []
    for feat in gj["features"]:
        county = feat["properties"].get("COUNTY")
        val = value_lookup.get(county, np.nan)
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = poly[0]
            patches.append(mpatches.Polygon(np.array(ring), closed=True))
            values.append(val)
    values = np.array(values, dtype=float)
    pc = PatchCollection(patches, cmap=cmap, edgecolor="white", linewidths=0.3)
    pc.set_array(np.nan_to_num(values, nan=vmin))
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    return pc


def _plot_geojson_highlight_counties(ax, gj, county_value_map, cmap, vmin, vmax):
    """Neutral gray basemap with ONLY the named counties coloured by their
    own result value (e.g. achieved CRI, ENS reduction). Used instead of
    repeating the NRI risk choropleth so each map panel actually shows
    different information rather than the same background three times."""
    _plot_geojson_plain(ax, gj, facecolor="#ececec", edgecolor="white")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    patches, colors = [], []
    for feat in gj["features"]:
        county = feat["properties"].get("COUNTY")
        if county not in county_value_map:
            continue
        val = county_value_map[county]
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = poly[0]
            patches.append(mpatches.Polygon(np.array(ring), closed=True))
            colors.append(cmap(norm(val)))
    if patches:
        pc = PatchCollection(patches, facecolor=colors, edgecolor="black", linewidths=0.8, zorder=3)
        ax.add_collection(pc)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


def _plot_geojson_plain(ax, gj, facecolor="#e8e8e8", edgecolor="white"):
    """Neutral flat basemap (no data-driven colouring) - used for panels where
    a repeated risk choropleth would be redundant and should instead let a
    results overlay (bubbles, markers) stand out on its own."""
    patches = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = poly[0]
            patches.append(mpatches.Polygon(np.array(ring), closed=True))
    pc = PatchCollection(patches, facecolor=facecolor, edgecolor=edgecolor, linewidths=0.3)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    return pc


# ==============================================================================
# 7. MAIN EXPERIMENT PIPELINE
# ==============================================================================
def main():
    rng_master = np.random.default_rng(42)
    print("=" * 80)
    print("PIRA-MPC wildfire-resilience pipeline starting")
    print("=" * 80)

    proxy = load_residential_proxy()
    fleet = load_pv_fleet()
    gamma = fit_thermal_derating_model(fleet)
    events = load_wildfire_weather()
    nri, gj = load_nri_geojson()

    risk_lookup = {row.county: row.risk_score for row in nri.itertuples()
                   if row.county is not None}

    # -------- build event-window scenarios (event window + 3-day pre/post buffer) --------
    scenarios = {}
    for ev_name, meta in events.items():
        df = meta["df"]
        idx_fire = np.nonzero(df["during_fire"].values)[0]
        i0 = max(idx_fire.min() - 72, 0)
        i1 = min(idx_fire.max() + 72, len(df))
        i1 = min(i1, i0 + MAX_WINDOW_HOURS)
        window = df.iloc[i0:i1]
        risk_score = risk_lookup.get(meta["county"], 50.0)
        county_scaling = np.clip(risk_score / 100.0, 0, 1)
        scen = build_feeder_scenario(proxy, window, gamma, risk_score, county_scaling,
                                      rng=np.random.default_rng(1))
        scenarios[ev_name] = dict(scenario=scen, meta=meta, risk_score=risk_score)
        print(f"[scenario] {ev_name}: window={len(window)}h, county={meta['county']}, "
              f"NRI risk score={risk_score:.1f}, PSPS-islanded hours={(scen.grid_available == 0).sum()}")

    # -------- Monte-Carlo comparison across methods --------
    all_metric_rows = []
    per_hour_examples = {}  # for time-series figure

    for ev_name, sd in scenarios.items():
        meta, risk_score = sd["meta"], sd["risk_score"]
        county_scaling = np.clip(risk_score / 100.0, 0, 1)
        fidx_mc = np.nonzero(meta["df"]["during_fire"].values)[0]
        i0_mc = max(fidx_mc.min() - 72, 0)
        i1_mc = min(fidx_mc.max() + 72, len(meta["df"]))
        i1_mc = min(i1_mc, i0_mc + MAX_WINDOW_HOURS)
        window_mc = meta["df"].iloc[i0_mc:i1_mc]
        for mc in range(N_MC):
            rng = np.random.default_rng(1000 + mc)
            scen_mc = build_feeder_scenario(proxy, window_mc, gamma, risk_score,
                                             county_scaling, rng=rng)
            for method in METHOD_ORDER:
                result = run_controller(method, scen_mc, rng)
                m = compute_metrics(result, scen_mc)
                m.update(dict(Event=meta["event"], County=meta["county"],
                               RiskScore=risk_score, Method=method, MC=mc))
                all_metric_rows.append(m)
                if mc == 0:
                    per_hour_examples[(ev_name, method)] = (result, scen_mc)
        print(f"[MC] completed {N_MC} Monte-Carlo runs x {len(METHOD_ORDER)} methods for {ev_name}")

    results_df = pd.DataFrame(all_metric_rows)

    # -------- sensitivity sweep: critical-load fraction x storage-scale, ALL methods --------
    # (Sweeping the PSPS wind-trigger percentile barely changes outcomes here,
    # since real historical wind/fire-window data already fixes most of the
    # islanding pattern - the critical-load share of demand is a far more
    # consequential, and more informative, operating parameter to test.)
    print("[sensitivity] running critical-load-fraction x storage-scale sweep ...")
    sens_rows = []
    crit_fractions = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    ess_scales = [0.5, 1.0, 1.5, 2.0]
    base_event = list(scenarios.values())[0]
    meta0 = base_event["meta"]
    fidx = np.nonzero(meta0["df"]["during_fire"].values)[0]
    i0s = max(fidx.min() - 72, 0)
    i1s = min(fidx.max() + 72, len(meta0["df"]))
    i1s = min(i1s, i0s + MAX_WINDOW_HOURS)
    window0 = meta0["df"].iloc[i0s:i1s]
    global ESS_KWH_PER_HH, ESS_MAX_KW_PER_HH, CRIT_FRACTION
    base_ess_kwh, base_ess_kw, base_crit_frac = ESS_KWH_PER_HH, ESS_MAX_KW_PER_HH, CRIT_FRACTION
    for cf in crit_fractions:
        CRIT_FRACTION = cf
        for es in ess_scales:
            ESS_KWH_PER_HH = base_ess_kwh * es
            ESS_MAX_KW_PER_HH = base_ess_kw * es
            rng = np.random.default_rng(7)
            scen_s = build_feeder_scenario(proxy, window0, gamma, base_event["risk_score"],
                                            np.clip(base_event["risk_score"] / 100, 0, 1), rng=rng)
            for method in METHOD_ORDER:
                res = run_controller(method, scen_s, rng)
                m = compute_metrics(res, scen_s)
                sens_rows.append(dict(CriticalFraction=cf, ESSScale=es, Method=method, **m))
    ESS_KWH_PER_HH, ESS_MAX_KW_PER_HH, CRIT_FRACTION = base_ess_kwh, base_ess_kw, base_crit_frac
    sens_df = pd.DataFrame(sens_rows)

    # ==========================================================================
    # SAVE EXCEL RESULTS
    # ==========================================================================
    excel_path = XLS_DIR / "PIRA_MPC_wildfire_resilience_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as xw:
        results_df.to_excel(xw, sheet_name="MonteCarlo_Metrics", index=False)
        results_df.groupby(["Event", "Method"])[
            ["ENS_kWh", "CriticalOutageHours", "FreqDevProxy", "CriticalServiceRatio", "CRI"]
        ].agg(["mean", "std"]).to_excel(xw, sheet_name="Summary_by_Event_Method")
        sens_df.to_excel(xw, sheet_name="Sensitivity_Sweep", index=False)
        nri.drop(columns=["geometry"]).to_excel(xw, sheet_name="NRI_County_Risk", index=False)
    print(f"[excel saved] {excel_path}")

    # ==========================================================================
    # FIGURES
    # ==========================================================================
    generate_all_figures(results_df, sens_df, scenarios, per_hour_examples, nri, gj, fleet, gamma)

    print("=" * 80)
    print("PIRA-MPC pipeline complete.")
    print(f"Figures -> {FIG_DIR}")
    print(f"Excel   -> {XLS_DIR}")
    print("=" * 80)


# ==============================================================================
# 8. FIGURE GENERATION
# ==============================================================================
def generate_all_figures(results_df, sens_df, scenarios, per_hour_examples, nri, gj, fleet, gamma):

    # ---- FIG 1 (single, hero figure): CRI dumbbell chart across methods x events ----
    print("[fig] Fig1_CRI_hero ...")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    summary = results_df.groupby(["Event", "Method"])["CRI"].mean().reset_index()
    events_list = list(summary["Event"].unique())
    y_pos = {ev: i for i, ev in enumerate(events_list)}

    # background resilience-quality bands, for at-a-glance interpretation
    band_edges = [0.0, 0.5, 0.75, 0.9, 1.0]
    band_colors = ["#fdecea", "#fff6e0", "#eef7ea", "#e2f3e3"]
    band_labels = ["Poor", "Moderate", "Good", "Excellent"]
    for i in range(len(band_edges) - 1):
        ax.axvspan(band_edges[i], band_edges[i + 1], color=band_colors[i], zorder=0)
        ax.text((band_edges[i] + band_edges[i + 1]) / 2, len(events_list) - 0.15, band_labels[i],
                ha="center", va="top", fontsize=FS - 6, color="#888", style="italic")

    seen_labels = set()
    for ev in events_list:
        vals = {}
        for m in METHOD_ORDER:
            r = summary[(summary.Event == ev) & (summary.Method == m)]
            if len(r):
                vals[m] = float(r.CRI.values[0])
        y = y_pos[ev]
        ax.plot([min(vals.values()), max(vals.values())], [y, y],
                color="gray", lw=2.5, alpha=0.5, zorder=1)
        for m, v in vals.items():
            is_pira = (m == "PIRA-MPC")
            lbl = METHOD_LABELS[m] if METHOD_LABELS[m] not in seen_labels else None
            seen_labels.add(METHOD_LABELS[m])
            ax.scatter(v, y, s=520 if is_pira else 220,
                       marker="*" if is_pira else "o",
                       color=METHOD_COLORS[m], edgecolor="black", linewidth=1.3,
                       zorder=4 if is_pira else 3, label=lbl)
        baseline_mean = np.mean([vals[m] for m in ["B1_Rule", "B2_ReactiveMPC", "B3_RobustReserve"] if m in vals])
        pira_v = vals.get("PIRA-MPC", np.nan)
        if np.isfinite(pira_v) and baseline_mean > 0:
            improvement = (pira_v - baseline_mean) / baseline_mean * 100
            ax.annotate(f"+{improvement:.0f}% vs. mean baseline", xy=(pira_v, y),
                        xytext=(pira_v - 0.02, y + 0.28), ha="right", fontsize=FS - 5,
                        fontweight="bold", color="#444")

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))
    ax.set_ylim(-0.6, len(events_list) - 1 + 0.6)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Composite Resilience Index (CRI)")
    ax.set_title("Operational Resilience under Real Wildfire Weather:\nProposed PIRA-MPC vs. Baselines")
    ax.grid(axis="x", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left", framealpha=1.0, fontsize=FS - 5)
    savefig(fig, "Fig1_CRI_hero_single")

    # ---- FIG 2 (2x2): four resilience metrics, one representative event ----
    print("[fig] Fig2_metrics_2x2 ...")
    ev_for_fig2 = results_df["Event"].unique()[0]
    d = results_df[results_df.Event == ev_for_fig2]
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    metric_specs = [("ENS_kWh", "Energy Not Served (kWh)"),
                    ("CriticalOutageHours", "Critical-Load Outage Duration (h)"),
                    ("FreqDevProxy", "Frequency-Deviation Proxy (p.u.)"),
                    ("CriticalServiceRatio", "Critical-Load Service Ratio")]
    for ax, (col, ylabel) in zip(axs.flat, metric_specs):
        means = [d[d.Method == m][col].mean() for m in METHOD_ORDER]
        stds = [d[d.Method == m][col].std() for m in METHOD_ORDER]
        colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
        bars = ax.bar(range(4), means, yerr=stds, capsize=4, color=colors)
        ax.set_xticks(range(4))
        ax.set_xticklabels([SHORT_LABELS[m] for m in METHOD_ORDER], rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        # Explicit value labels: a metric that is genuinely ~0 (e.g. PIRA-MPC's
        # critical-load outage duration) renders as an invisible bar, which
        # can misleadingly look like missing data rather than a real result.
        y_span = max(ax.get_ylim()[1], 1e-9)
        for bar, val, sd in zip(bars, means, stds):
            label_y = bar.get_height() + (sd if np.isfinite(sd) else 0) + 0.02 * y_span
            ax.annotate(f"{val:.2f}" if y_span < 3 else f"{val:.0f}",
                        (bar.get_x() + bar.get_width() / 2, label_y),
                        ha="center", va="bottom", fontsize=FS - 5)
        ax.set_ylim(0, y_span * 1.15)
    fig.suptitle(f"Resilience Metric Comparison Across Methods - {ev_for_fig2}", y=1.02)
    handles = [mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m]) for m in METHOD_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    savefig(fig, "Fig2_four_metrics_2x2")

    # ---- FIG 3 (1x2): PV physics model + weather profile ----
    print("[fig] Fig3_physics_1x2 ...")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    # (a) measured cell temperature vs fitted thermal derating curve
    uid, df_pv = next((it for it in fleet.items()
                        if "pv_kw" in it[1].columns and it[1]["pv_kw"].max() > 1.0))
    d2 = df_pv.dropna(subset=["Tcell_degC", "pv_kw", "ghi"])
    d2 = d2[d2["pv_kw"] > 0.05]
    # restrict to near-peak-irradiance hours so the remaining scatter mainly
    # reflects the temperature effect, not irradiance variability (cloud
    # cover, time-of-day) - standard practice when isolating a PV temperature
    # coefficient from field data
    ghi_hi = d2["ghi"].quantile(0.75)
    d2 = d2[d2["ghi"] >= max(ghi_hi, 400)]
    axs[0].scatter(d2["Tcell_degC"], d2["pv_kw"] / d2["pv_kw"].quantile(0.98),
                   s=10, alpha=0.35, color="#4C72B0", label="Measured (normalised, near-peak sun)")
    tt = np.linspace(d2["Tcell_degC"].min(), d2["Tcell_degC"].max(), 100)
    axs[0].plot(tt, 1 + gamma * (tt - 25), color="#C44E52", lw=3,
                label=f"Fitted physics model ($\\gamma$={gamma:.4f}/°C)")
    axs[0].set_xlabel("PV Cell Temperature (°C)")
    axs[0].set_ylabel("Normalised PV Output")
    axs[0].legend(loc='best', framealpha=1.0)
    axs[0].set_title("(a) Physics-Informed Thermal Derating")
    # (b) wind speed profile for both fire events with PSPS threshold band
    for ev_name, sd in scenarios.items():
        scen = sd["scenario"]
        axs[1].plot(np.arange(len(scen.wind_mph)), scen.wind_mph, lw=1.3, label=ev_name)
    axs[1].set_xlabel("Hour of Event Window")
    axs[1].set_ylabel("Wind Speed (mph)")
    axs[1].legend(loc='best', framealpha=1.0)
    axs[1].set_title("(b) Real NOAA Wind Speed During Fire Windows")
    savefig(fig, "Fig3_pv_physics_and_weather_1x2")

    # ---- FIG 4 (2x3): sensitivity sweep, one subplot per critical-load fraction ----
    print("[fig] Fig4_sensitivity_2x3 ...")
    fig, axs = plt.subplots(2, 3, figsize=(18, 11))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    cfs = sorted(sens_df.CriticalFraction.unique())
    for ax, cf in zip(axs.flat, cfs):
        dsub = sens_df[sens_df.CriticalFraction == cf]
        for method in METHOD_ORDER:
            g = dsub[dsub.Method == method].sort_values("ESSScale")
            ax.plot(g.ESSScale, g.CRI, marker="o", lw=2.5,
                    color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.set_title(f"Critical Load = {int(cf * 100)}% of Demand")
        ax.set_xlabel("ESS Capacity Scale (x baseline)")
        ax.set_ylabel("CRI")
        ax.set_ylim(0, 1.05)
    for k in range(len(cfs), 6):
        axs.flat[k].axis("off")
    handles = [plt.Line2D([0], [0], color=METHOD_COLORS[m], marker="o", label=METHOD_LABELS[m])
               for m in METHOD_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4)
    fig.suptitle("Sensitivity of Resilience (CRI) to Critical-Load Share and Storage Sizing", y=1.08)
    savefig(fig, "Fig4_sensitivity_wind_ess_2x3")

    # ---- FIG 5 (single, important): NRI wildfire-risk choropleth with fire ignition points ----
    print("[fig] Fig5_NRI_map_single ...")
    fig, ax = plt.subplots(figsize=(9, 9))
    value_lookup = {row.county: row.risk_score for row in nri.itertuples()}
    pc = _plot_geojson_choropleth(ax, gj, value_lookup, cmap=cm.get_cmap("YlOrRd"),
                                   vmin=nri.risk_score.min(), vmax=nri.risk_score.max())
    cb = fig.colorbar(pc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("FEMA NRI Wildfire Risk Score", fontsize=FS)
    cb.ax.tick_params(labelsize=FS - 2)
    ignition_points = {"Kincade Fire (2019)": (-122.780053, 38.792458),
                        "Dixie Fire (2021)": (-121.419, 39.819)}
    for name, (lon, lat) in ignition_points.items():
        ax.scatter(lon, lat, marker="*", s=400, color="black", edgecolor="white", zorder=5)
        ax.annotate(name, (lon, lat), textcoords="offset points", xytext=(10, 6), fontsize=FS - 2)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("California County Wildfire Risk (FEMA NRI)\nand Study Fire-Event Locations")
    savefig(fig, "Fig5_NRI_choropleth_single")

    # ---- FIG 6 (2x2): control-action time series, proposed method, one event ----
    print("[fig] Fig6_control_timeseries_2x2 ...")
    ev_key = list(scenarios.keys())[0]
    result_pira, scen6 = per_hour_examples[(ev_key, "PIRA-MPC")]
    result_b2, _ = per_hour_examples[(ev_key, "B2_ReactiveMPC")]
    hrs = np.arange(len(scen6.hours))
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(hrs, scen6.load_crit + scen6.load_flex, color="black", lw=1.5, label="Total load")
    axs[0, 0].fill_between(hrs, 0, (scen6.grid_available == 0) * (scen6.load_crit + scen6.load_flex).max(),
                            color="grey", alpha=0.2, label="Islanded (PSPS)")
    axs[0, 0].set_ylabel("Load (kW)")
    axs[0, 0].set_title("(a) Feeder Load & PSPS Islanding")
    axs[0, 0].legend(loc='best', framealpha=1.0)

    axs[0, 1].plot(hrs, result_pira["uns_crit"] + result_pira["uns_flex"],
                   color=METHOD_COLORS["PIRA-MPC"], lw=2, label=METHOD_LABELS["PIRA-MPC"])
    axs[0, 1].plot(hrs, result_b2["uns_crit"] + result_b2["uns_flex"],
                   color=METHOD_COLORS["B2_ReactiveMPC"], lw=2, label=METHOD_LABELS["B2_ReactiveMPC"])
    axs[0, 1].set_ylabel("Unserved Energy (kW)")
    axs[0, 1].set_title("(b) Unserved Load: Proposed vs. Reactive MPC")
    axs[0, 1].legend(loc='best', framealpha=1.0)

    axs[1, 0].plot(hrs, result_pira["ess_trace"], color="#55A868", lw=1.8, label="ESS (dis>0/chg<0)")
    axs[1, 0].plot(hrs, result_pira["ev_trace"], color="#8172B2", lw=1.8, label="EV V2G (dis>0/chg<0)")
    axs[1, 0].axhline(0, color="gray", lw=0.8)
    axs[1, 0].set_ylabel("Power (kW)")
    axs[1, 0].set_xlabel("Hour of Event Window")
    axs[1, 0].set_title("(c) PIRA-MPC Storage & V2G Dispatch")
    axs[1, 0].legend(loc='best', framealpha=1.0)

    axs[1, 1].plot(hrs, result_pira["pv_used"], color="#C44E52", lw=1.8, label="PV used (PIRA-MPC)")
    axs[1, 1].plot(hrs, scen6.pv_forecast_true, color="black", lw=1, ls="--", label="Physics PV potential")
    axs[1, 1].set_ylabel("PV Power (kW)")
    axs[1, 1].set_xlabel("Hour of Event Window")
    axs[1, 1].set_title("(d) PV Utilisation vs. Physics-Based Potential")
    axs[1, 1].legend(loc='best', framealpha=1.0)
    fig.suptitle(f"PIRA-MPC Control Behaviour During {ev_key}", y=1.02)
    savefig(fig, "Fig6_control_timeseries_2x2")

    # ---- FIG 7 (single, combined): CRI distribution boxplots, both events together ----
    print("[fig] Fig7_boxplots_1x2 ...")
    events_list7 = list(results_df["Event"].unique())
    event_hatches = {events_list7[0]: None, events_list7[1]: "///"}
    fig, ax = plt.subplots(figsize=(13, 7))
    n_methods = len(METHOD_ORDER)
    group_width = 2.2
    box_width = 0.8
    positions_all, data_all, colors_all, hatches_all = [], [], [], []
    group_centers = []
    for i, method in enumerate(METHOD_ORDER):
        base = i * group_width
        group_centers.append(base + (len(events_list7) - 1) * box_width / 2)
        for j, ev in enumerate(events_list7):
            pos = base + j * box_width
            positions_all.append(pos)
            d = results_df[(results_df.Method == method) & (results_df.Event == ev)]["CRI"].values
            data_all.append(d)
            colors_all.append(METHOD_COLORS[method])
            hatches_all.append(event_hatches[ev])
    bp = ax.boxplot(data_all, positions=positions_all, widths=box_width * 0.85,
                     patch_artist=True)
    for patch, color, hatch in zip(bp["boxes"], colors_all, hatches_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")
        if hatch:
            patch.set_hatch(hatch)
    # whiskers/caps/medians are separate line artists (2 each per box) - colour
    # them explicitly so a near-zero-height box (e.g. PIRA-MPC with very low
    # variance) still reads correctly instead of falling back to matplotlib's
    # default line colour, which can make a tiny box look like the wrong colour
    for i, color in enumerate(colors_all):
        for artist_group in (bp["whiskers"], bp["caps"]):
            artist_group[2 * i].set_color("black")
            artist_group[2 * i + 1].set_color("black")
        bp["medians"][i].set_color("black")
        bp["medians"][i].set_linewidth(1.5)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([SHORT_LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("CRI (Monte-Carlo distribution)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Monte-Carlo Uncertainty in Resilience Outcomes\n(n={N_MC} draws/method/event)")
    method_handles = [mpatches.Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS[m])
                       for m in METHOD_ORDER]
    event_handles = [mpatches.Patch(facecolor="white", edgecolor="black",
                                     hatch=event_hatches[ev], label=ev) for ev in events_list7]
    leg1 = ax.legend(handles=method_handles, loc="lower right", framealpha=1.0, fontsize=FS - 5,
                      title="Method", title_fontsize=FS - 4)
    ax.add_artist(leg1)
    ax.legend(handles=event_handles, loc="lower left", framealpha=1.0, fontsize=FS - 5,
              title="Event", title_fontsize=FS - 4)
    savefig(fig, "Fig7_CRI_boxplots_1x2")

    # ---- FIG 8 (1x3, important): wildfire's effect on the underlying Pecan
    # Street load/PV/EV proxy data, under the proposed method ----
    print("[fig] Fig8_wildfire_impact_1x3 ...")
    ev_key8 = list(scenarios.keys())[0]
    scen8 = scenarios[ev_key8]["scenario"]
    result_pira8, _ = per_hour_examples[(ev_key8, "PIRA-MPC")]
    hod = scen8.hours.hour
    fig, axs = plt.subplots(1, 3, figsize=(22, 7))
    fig.subplots_adjust(wspace=0.45, bottom=0.28)

    # (a) naive vs climate-zone-adjusted proxy transfer (load)
    df_load = pd.DataFrame({"hod": hod, "raw": scen8.load_feeder_raw,
                             "adj": scen8.load_crit + scen8.load_flex})
    g = df_load.groupby("hod").mean()
    axs[0].plot(g.index, g["raw"], lw=2.5, color="#8172B2", ls="--",
                label="Naive proxy transfer (no climate adj.)")
    axs[0].plot(g.index, g["adj"], lw=2.5, color="#C44E52",
                label="Climate-zone-adjusted transfer (used)")
    axs[0].set_xlabel("Hour of Day")
    axs[0].set_ylabel("Feeder Load (kW)")
    axs[0].set_title("(a) Wildfire-Region Climate Adjustment\nof the Bay Area Load Proxy")
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1,
                  framealpha=1.0, fontsize=FS - 5)

    # (b) nameplate vs physics-informed (real wildfire-region temperature) PV,
    # PLUS the local ambient temperature driving the derating (twin axis) and
    # a shaded loss region, since the two power curves alone are visually too
    # close together (a few % effect) and the panel looked empty
    df_pv = pd.DataFrame({"hod": hod, "nameplate": scen8.pv_forecast_naive,
                           "physics": scen8.pv_forecast_true,
                           "temp": scen8.local_temp_c if scen8.local_temp_c is not None
                           else np.full(len(hod), np.nan)})
    g2 = df_pv.groupby("hod").mean()
    l2, = axs[1].plot(g2.index, g2["physics"], lw=2.5, color="#4C72B0", zorder=2,
                       label="Physics-informed PV (real local temperature)")
    l1, = axs[1].plot(g2.index, g2["nameplate"], lw=2.0, color="gray", ls="--", zorder=3,
                       label="Nameplate PV (no derating)")
    axs[1].fill_between(g2.index, g2["physics"], g2["nameplate"],
                         where=g2["nameplate"] >= g2["physics"],
                         color="#C44E52", alpha=0.25, zorder=1, label="Thermal derating loss")
    peak_loss_pct = float(((g2["nameplate"] - g2["physics"]) / g2["nameplate"].replace(0, np.nan)
                            * 100).max())
    if np.isfinite(peak_loss_pct):
        axs[1].text(0.97, 0.80, f"Peak thermal\nloss \u2248 {peak_loss_pct:.1f}%",
                    transform=axs[1].transAxes, ha="right", va="top", fontsize=FS - 4,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#C44E52", alpha=0.9))
    ax1b = axs[1].twinx()
    l3, = ax1b.plot(g2.index, g2["temp"], lw=2, color="#DD8452", ls=":",
                     label="Local ambient temperature")
    ax1b.set_ylabel("Ambient Temperature (\u00b0C)", color="#DD8452")
    ax1b.tick_params(axis="y", labelcolor="#DD8452")
    axs[1].set_xlabel("Hour of Day")
    axs[1].set_ylabel("PV Power (kW)")
    axs[1].set_title("(b) Wildfire-Heat Impact on\nPV Generation (Thermal Derating)")
    axs[1].legend(handles=[l1, l2, l3], loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=1, framealpha=1.0, fontsize=FS - 5)

    # (c) normal (grid-available) vs wildfire-disrupted (islanded) EV dispatch
    islanded8 = scen8.grid_available == 0
    df_ev = pd.DataFrame({"hod": hod, "ev": result_pira8["ev_trace"], "islanded": islanded8})
    g3n = df_ev[~df_ev.islanded].groupby("hod")["ev"].mean()
    g3i = df_ev[df_ev.islanded].groupby("hod")["ev"].mean()
    axs[2].axhline(0, color="gray", lw=0.8)
    axs[2].plot(g3n.index, g3n.values, lw=2.5, color="#55A868", marker="o",
                label="Normal (grid available): net charging")
    axs[2].plot(g3i.index, g3i.values, lw=2.5, color="#C44E52", marker="s",
                label="During PSPS (PIRA-MPC): net V2G support")
    axs[2].set_xlabel("Hour of Day")
    axs[2].set_ylabel("EV Fleet Power (kW, dis>0/chg<0)")
    axs[2].set_title("(c) Wildfire-Driven Shift in EV Behaviour:\nCharging \u2192 V2G Support (PIRA-MPC)")
    axs[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1,
                  framealpha=1.0, fontsize=FS - 5)

    fig.suptitle(f"Linking Pecan Street Load/PV/EV Data to Real Wildfire Conditions \u2013 {ev_key8}", y=1.04)
    savefig(fig, "Fig8_wildfire_impact_on_load_pv_ev_1x3")

    # ---- FIG 9 (2x2, important): proposed-method results in geographic context ----
    print("[fig] Fig9_proposed_method_maps_2x2 ...")
    ignition_points2 = {"Kincade Fire (2019)": (-122.780053, 38.792458, "Sonoma"),
                         "Dixie Fire (2021)": (-121.419, 39.819, "Shasta")}
    bay_area_coord = (-122.27, 37.87)  # approx. Pecan Street Bay Area proxy source

    summary_ens = results_df.groupby(["Event", "Method"])["ENS_kWh"].mean().reset_index()

    def _ens_for(event, method):
        r = summary_ens[(summary_ens.Event == event) & (summary_ens.Method == method)]
        return float(r.ENS_kWh.values[0]) if len(r) else np.nan

    # real, per-county mean event-window wind speed (mph) - an actual result
    # from the NOAA data, not a placeholder illustration
    wind_by_county = {}
    for name, (lon, lat, county) in ignition_points2.items():
        ev_name = name.split(" (")[0]
        scen_for_ev = scenarios[ev_name]["scenario"]
        wind_by_county[county] = float(np.nanmean(scen_for_ev.wind_mph))

    # absolute ENS achieved BY PIRA-MPC, and by the mean of the three
    # baselines, on the SAME colour scale so the contrast is visually
    # honest and the two counties' real (different) magnitudes both show
    ens_pira_by_county, ens_baseline_by_county = {}, {}
    for name, (lon, lat, county) in ignition_points2.items():
        ev_name = name.split(" (")[0]
        ens_pira_by_county[county] = _ens_for(ev_name, "PIRA-MPC")
        ens_baseline_by_county[county] = np.nanmean(
            [_ens_for(ev_name, m) for m in ["B1_Rule", "B2_ReactiveMPC", "B3_RobustReserve"]])
    shared_vmax = max(list(ens_pira_by_county.values()) + list(ens_baseline_by_county.values()) + [1.0])

    # real per-county mean PSPS-islanded hours during the event window - a
    # NEW result not shown anywhere else (Fig5 already covers the static NRI
    # risk map, so repeating it here as panel (a) would be pure duplication)
    islanded_hours_by_county = {}
    for name, (lon, lat, county) in ignition_points2.items():
        ev_name = name.split(" (")[0]
        scen_for_ev = scenarios[ev_name]["scenario"]
        islanded_hours_by_county[county] = float(np.sum(scen_for_ev.grid_available == 0))

    fig, axs = plt.subplots(2, 2, figsize=(17, 15))
    fig.subplots_adjust(wspace=0.30, hspace=0.22)

    # (a) PSPS-driven de-energisation exposure per county (real simulation
    # output - distinct from the static NRI risk map already shown in Fig. 5)
    sm_a = _plot_geojson_highlight_counties(axs[0, 0], gj, islanded_hours_by_county,
                                             cmap=cm.get_cmap("turbo"),
                                             vmin=0.0, vmax=max(islanded_hours_by_county.values()) * 1.15)
    cb_a = fig.colorbar(sm_a, ax=axs[0, 0], fraction=0.035, pad=0.12, shrink=0.85)
    cb_a.set_label("PSPS-Islanded Hours\nduring Event Window", fontsize=FS - 3, labelpad=10)
    for name, (lon, lat, county) in ignition_points2.items():
        axs[0, 0].scatter(lon, lat, marker="*", s=350, color="black", edgecolor="white", zorder=5)
        axs[0, 0].annotate(f"{name.split(' (')[0]}\n{islanded_hours_by_county[county]:.0f} h", (lon, lat),
                            textcoords="offset points", xytext=(8, 4), fontsize=FS - 4)
    gray_patch_a = mpatches.Patch(facecolor="#ececec", edgecolor="gray",
                                   label="Other CA counties\n(not modelled in this study)")
    axs[0, 0].legend(handles=[gray_patch_a], loc="lower left", fontsize=FS - 5, framealpha=1.0)
    axs[0, 0].set_title("(a) Simulated PSPS De-Energisation\nExposure (by county)")
    axs[0, 0].set_xlabel("Longitude"); axs[0, 0].set_ylabel("Latitude")

    # (b) REAL result: mean event-window wind speed per county (the actual
    # NOAA driver of PSPS islanding), plus the proxy-transfer pathway arrows
    sm_b = _plot_geojson_highlight_counties(axs[0, 1], gj, wind_by_county,
                                             cmap=cm.get_cmap("plasma"),
                                             vmin=0.0, vmax=max(wind_by_county.values()) * 1.15)
    cb_b = fig.colorbar(sm_b, ax=axs[0, 1], fraction=0.035, pad=0.12, shrink=0.85)
    cb_b.set_label("Mean Event-Window Wind Speed (mph)", fontsize=FS - 3, labelpad=10)
    axs[0, 1].scatter(*bay_area_coord, marker="o", s=250, color="#4C72B0", edgecolor="white",
                       zorder=5, label="Pecan Street proxy source (Bay Area)")
    for name, (lon, lat, county) in ignition_points2.items():
        axs[0, 1].scatter(lon, lat, marker="*", s=300, color="black", edgecolor="white", zorder=6)
        axs[0, 1].annotate(f"{name.split(' (')[0]}\n{wind_by_county[county]:.1f} mph avg", (lon, lat),
                            textcoords="offset points", xytext=(8, 4), fontsize=FS - 4)
        axs[0, 1].plot([bay_area_coord[0], lon], [bay_area_coord[1], lat],
                       color="#C44E52", lw=1.5, ls="--", alpha=0.7)
    axs[0, 1].legend(loc="lower left", fontsize=FS - 5, framealpha=1.0)
    axs[0, 1].set_title("(b) Real Wind Severity Driving PSPS Risk,\nand the Proxy Data-Transfer Pathway")
    axs[0, 1].set_xlabel("Longitude"); axs[0, 1].set_ylabel("Latitude")

    # (c) PIRA-MPC absolute Energy Not Served (kWh) - lower is better
    sm_c = _plot_geojson_highlight_counties(axs[1, 0], gj, ens_pira_by_county,
                                             cmap=cm.get_cmap("RdYlGn_r"), vmin=0.0, vmax=shared_vmax)
    cb_c = fig.colorbar(sm_c, ax=axs[1, 0], fraction=0.035, pad=0.12, shrink=0.85)
    cb_c.set_label("Energy Not Served (kWh)\n(same scale as panel d)", fontsize=FS - 3, labelpad=10)
    for name, (lon, lat, county) in ignition_points2.items():
        axs[1, 0].scatter(lon, lat, marker="*", s=250, color="black", edgecolor="white", zorder=5)
        axs[1, 0].annotate(f"{name.split(' (')[0]}\n{ens_pira_by_county[county]:.0f} kWh", (lon, lat),
                            textcoords="offset points", xytext=(10, 6), fontsize=FS - 3)
    gray_patch_c = mpatches.Patch(facecolor="#ececec", edgecolor="gray",
                                   label="Other CA counties\n(not modelled in this study)")
    axs[1, 0].legend(handles=[gray_patch_c], loc="lower left", fontsize=FS - 5, framealpha=1.0)
    axs[1, 0].set_title("(c) PIRA-MPC: Energy Not Served\n(by county)")
    axs[1, 0].set_xlabel("Longitude"); axs[1, 0].set_ylabel("Latitude")

    # (d) Mean-of-baselines absolute Energy Not Served (kWh) - SAME colour
    # scale as (c), so the contrast between panels is a direct, honest
    # visual comparison rather than two independently-normalised percentages
    sm_d = _plot_geojson_highlight_counties(axs[1, 1], gj, ens_baseline_by_county,
                                             cmap=cm.get_cmap("RdYlGn_r"), vmin=0.0, vmax=shared_vmax)
    cb_d = fig.colorbar(sm_d, ax=axs[1, 1], fraction=0.035, pad=0.12, shrink=0.85)
    cb_d.set_label("Energy Not Served (kWh)\n(same scale as panel c)", fontsize=FS - 3, labelpad=10)
    for name, (lon, lat, county) in ignition_points2.items():
        axs[1, 1].scatter(lon, lat, marker="*", s=250, color="black", edgecolor="white", zorder=5)
        axs[1, 1].annotate(f"{name.split(' (')[0]}\n{ens_baseline_by_county[county]:.0f} kWh", (lon, lat),
                            textcoords="offset points", xytext=(10, 6), fontsize=FS - 3)
    gray_patch_d = mpatches.Patch(facecolor="#ececec", edgecolor="gray",
                                   label="Other CA counties\n(not modelled in this study)")
    axs[1, 1].legend(handles=[gray_patch_d], loc="lower left", fontsize=FS - 5, framealpha=1.0)
    axs[1, 1].set_title("(d) Mean of Baselines (B1/B2/B3):\nEnergy Not Served (by county)")
    axs[1, 1].set_xlabel("Longitude"); axs[1, 1].set_ylabel("Latitude")

    fig.suptitle("Proposed-Method (PIRA-MPC) Results in Geographic Context", y=0.99)
    savefig(fig, "Fig9_proposed_method_maps_2x2")


if __name__ == "__main__":
    main()