import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.spatial import cKDTree
from scipy.optimize import milp, LinearConstraint, Bounds
import statsmodels.api as sm

OUT_DIR = r"D:\ouda\future research\green threshod\Round 1\revised manuscript\04_Reproducible_GIS_Audit_and_Figures"
os.makedirs(OUT_DIR, exist_ok=True)

GDB = r"D:\ouda\future research\green threshod\New Folder\MyProject1\MyProject1.gdb"
PARKS_SHP = r"D:\ouda\future research\green threshod\tables\parks2.shp"
STREAMS_SHP = r"D:\ouda\future research\green threshod\New Folder\MyProject1\shapefiles\stream.shp"

print("Loading GIS layers...")
rd = gpd.read_file(GDB, layer="road1")
bldg = gpd.read_file(GDB, layer="Extract_Buidlin1")
parks_pt = gpd.read_file(PARKS_SHP)
parks_poly = gpd.read_file(GDB, layer="Park_Ex1_Merge_ExportFeatures")
cand_poly = gpd.read_file(GDB, layer="Candidate_UPGS")
cand_pt = gpd.read_file(GDB, layer="Candidate_UPGS_Centroid")
districts = gpd.read_file(GDB, layer="Fig3_District_Boundaries_Final")
central_b = gpd.read_file(GDB, layer="Fig3_Central_Urban_Boundary")
fac_arcgis = gpd.read_file(GDB, layer="Facilities1oy0xuo")
od_arcgis = gpd.read_file(GDB, layer="ODLineshzsu8o")
la_arcgis = gpd.read_file(GDB, layer="LALines5lju8w")

# --- 1. CLEAN & RECONCILE PARK AREAS ---
# Spatial join missing Area_m2 in parks_pt from parks_poly
parks_pt = parks_pt.copy()
parks_pt["park_id"] = np.arange(len(parks_pt))
# Use polygon area if Area_m2 is null or 0
poly_valid = parks_poly[parks_poly["Shape_Area"] > 10].copy()
joined = gpd.sjoin_nearest(parks_pt[["park_id", "Area_m2", "geometry"]], poly_valid[["Shape_Area", "geometry"]], how="left", max_distance=100)
joined = joined.drop_duplicates(subset="park_id")
parks_pt["Area_m2_clean"] = parks_pt["Area_m2"].fillna(joined["Shape_Area"]).fillna(1000.0)
parks_pt.loc[parks_pt["Area_m2_clean"] <= 0, "Area_m2_clean"] = 1000.0

print(f"Existing Parks (n={len(parks_pt)}): total Area_m2_clean = {parks_pt['Area_m2_clean'].sum()/1e4:.2f} ha ({parks_pt['Area_m2_clean'].sum()/1e6:.4f} km2)")
print(f"  min={parks_pt['Area_m2_clean'].min()/1e4:.4f} ha, med={parks_pt['Area_m2_clean'].median()/1e4:.4f} ha, mean={parks_pt['Area_m2_clean'].mean()/1e4:.4f} ha, max={parks_pt['Area_m2_clean'].max()/1e4:.4f} ha")

# Candidate parcels
cand = cand_pt.copy()
cand["Area_m2"] = cand_poly["Shape_Area"].values
cand["Area_ha"] = cand["Area_m2"] / 10000.0
cand["cand_id"] = np.arange(len(cand))
cand_large = cand[cand["Type"] == "Large-scale UPGS"].copy().reset_index(drop=True)
cand_small = cand[cand["Type"] == "Small-scale UPGS"].copy().reset_index(drop=True)
print(f"Candidate Parcels (n={len(cand)}): {len(cand_large)} Large ({cand_large['Area_ha'].sum():.2f} ha), {len(cand_small)} Small ({cand_small['Area_ha'].sum():.2f} ha)")

# Demand points
bldg = bldg.copy()
bldg["bldg_id"] = np.arange(len(bldg))
bldg["pop"] = bldg["RASTERVALU"].astype(float)
total_pop = bldg["pop"].sum()
print(f"Demand Points: n={len(bldg)}, total_pop={total_pop:,.2f}, min={bldg['pop'].min():.2f}, med={bldg['pop'].median():.2f}, mean={bldg['pop'].mean():.2f}, max={bldg['pop'].max():.2f}")

# --- 2. BUILD TOPOLOGICAL PEDESTRIAN NETWORK ---
# Exclude motorway and construction (unwalkable/under construction); keep rest as street graph
excl_hw = {"motorway", "motorway_link", "construction"}
rd_walk = rd[~rd["highway"].isin(excl_hw)].copy()

G = nx.Graph()
for geom in rd_walk.geometry:
    lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
    for ln in lines:
        coords = list(ln.coords)
        for i in range(len(coords) - 1):
            u = (round(coords[i][0], 1), round(coords[i][1], 1))
            v = (round(coords[i+1][0], 1), round(coords[i+1][1], 1))
            w = float(np.hypot(u[0] - v[0], u[1] - v[1]))
            if w > 0:
                if G.has_edge(u, v):
                    if w < G[u][v]["weight"]:
                        G[u][v]["weight"] = w
                else:
                    G.add_edge(u, v, weight=w)

comps = sorted(nx.connected_components(G), key=len, reverse=True)
print(f"Network Graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, components={len(comps)}, largest={len(comps[0])}")

all_nodes = np.array(list(G.nodes()))
node_list = list(G.nodes())
node_tree = cKDTree(all_nodes)

def snap_points(gdf, max_snap=500.0):
    xy = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    dists, idxs = node_tree.query(xy)
    snapped_nodes = [node_list[i] if d <= max_snap else None for i, d in zip(idxs, dists)]
    return snapped_nodes, dists

bldg["node"], bldg["snap_dist"] = snap_points(bldg, max_snap=500.0)
parks_pt["node"], parks_pt["snap_dist"] = snap_points(parks_pt, max_snap=500.0)
cand["node"], cand["snap_dist"] = snap_points(cand, max_snap=500.0)
cand_large["node"] = cand.loc[cand["Type"] == "Large-scale UPGS", "node"].values
cand_small["node"] = cand.loc[cand["Type"] == "Small-scale UPGS", "node"].values

print(f"Snapped within 500m: bldg={(bldg['node'].notna()).sum()}/{len(bldg)}, parks={(parks_pt['node'].notna()).sum()}/{len(parks_pt)}, cand={(cand['node'].notna()).sum()}/{len(cand)}")

# --- 3. COMPUTE SHORTEST-PATH NETWORK DISTANCES UP TO 3000m ---
# Multi-source Dijkstra from each facility node up to cutoff=3000m (plus snap distances!)
# To be fast, we map each building node -> list of bldg_ids
bldg_node_to_ids = {}
for idx, nd in enumerate(bldg["node"]):
    if nd is not None:
        bldg_node_to_ids.setdefault(nd, []).append(idx)

def compute_od_matrix(fac_df, cutoff=3000.0):
    # Returns DataFrame of (bldg_idx, fac_idx, net_dist, euc_dist)
    fac_xy = np.column_stack([fac_df.geometry.x, fac_df.geometry.y])
    b_xy = np.column_stack([bldg.geometry.x, bldg.geometry.y])
    records = []
    for f_idx, row in fac_df.iterrows():
        f_node = row["node"]
        if f_node is None:
            continue
        f_snap = row["snap_dist"]
        # Dijkstra up to cutoff
        lengths = nx.single_source_dijkstra_path_length(G, f_node, cutoff=cutoff, weight="weight")
        for nd, path_len in lengths.items():
            if nd in bldg_node_to_ids:
                for b_idx in bldg_node_to_ids[nd]:
                    tot_net = path_len + f_snap + bldg.at[b_idx, "snap_dist"]
                    if tot_net <= cutoff:
                        euc = float(np.hypot(fac_xy[f_idx, 0] - b_xy[b_idx, 0], fac_xy[f_idx, 1] - b_xy[b_idx, 1]))
                        records.append((b_idx, f_idx, tot_net, euc))
    return pd.DataFrame(records, columns=["bldg_idx", "fac_idx", "net_dist", "euc_dist"])

print("Computing OD matrices (cutoff=3000m)...")
od_existing = compute_od_matrix(parks_pt.reset_index(drop=True), cutoff=3000.0)
od_cand_all = compute_od_matrix(cand.reset_index(drop=True), cutoff=3000.0)
print(f"OD pairs <= 3000m: existing={len(od_existing)}, candidate={len(od_cand_all)}")

# --- 4. CIRCUITY DIAGNOSTICS ---
# Compare with ArcGIS OD_Lines_Export.csv and our NetworkX OD
od_valid = od_existing[od_existing["euc_dist"] > 1.0].copy()
od_valid["circuity"] = od_valid["net_dist"] / od_valid["euc_dist"]
print(f"NetworkX OD Circuity (n={len(od_valid)} pairs <=3000m): median={od_valid['circuity'].median():.3f}, mean={od_valid['circuity'].mean():.3f}, IQR=({od_valid['circuity'].quantile(0.25):.3f}, {od_valid['circuity'].quantile(0.75):.3f}), max={od_valid['circuity'].max():.2f}")

# Also check ArcGIS ODLineshzsu8o circuity
od_arc = od_arcgis[od_arcgis["Shape_Length"] > 1.0].copy()
od_arc["circuity"] = od_arc["Total_Length"] / od_arc["Shape_Length"]
print(f"ArcGIS ODLines (n={len(od_arc)}): median={od_arc['circuity'].median():.3f}, mean={od_arc['circuity'].mean():.3f}, IQR=({od_arc['circuity'].quantile(0.25):.3f}, {od_arc['circuity'].quantile(0.75):.3f}), max={od_arc['circuity'].max():.2f}")

# --- 5. G2SFCA & INEQUALITY METRICS FUNCTION ---
def gaussian_decay(d, d0):
    return np.where(d <= d0, (np.exp(-0.5 * (d / d0) ** 2) - np.exp(-0.5)) / (1.0 - np.exp(-0.5)), 0.0)

def weighted_gini(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum_w = np.cumsum(w)
    cum_vw = np.cumsum(v * w)
    if cum_vw[-1] == 0:
        return 0.0
    # Lorenz curve area
    w_norm = cum_w / cum_w[-1]
    vw_norm = cum_vw / cum_vw[-1]
    w_prev = np.concatenate(([0.0], w_norm[:-1]))
    vw_prev = np.concatenate(([0.0], vw_norm[:-1]))
    area = np.sum((w_norm - w_prev) * (vw_norm + vw_prev) / 2.0)
    return float(1.0 - 2.0 * area)

def weighted_theil(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mu = np.average(v, weights=w)
    if mu <= 0:
        return 0.0
    pos = v > 0
    r = v[pos] / mu
    p = w[pos] / w.sum()
    return float(np.sum(p * r * np.log(r)))

def run_g2sfca(od_df, fac_areas, d0, decay="gaussian", dist_col="net_dist"):
    sub = od_df[od_df[dist_col] <= d0].copy()
    if len(sub) == 0:
        return np.zeros(len(bldg))
    if decay == "gaussian":
        sub["w_ij"] = gaussian_decay(sub[dist_col].values, d0)
    else:
        sub["w_ij"] = 1.0
    sub["pop_i"] = bldg["pop"].values[sub["bldg_idx"].values]
    # Step 1: R_j = S_j / sum_i (P_i * w_ij)
    denom = sub.groupby("fac_idx").apply(lambda g: np.sum(g["pop_i"] * g["w_ij"]), include_groups=False)
    R_j = np.zeros(len(fac_areas))
    for f_idx, d_val in denom.items():
        if d_val > 0:
            R_j[int(f_idx)] = fac_areas[int(f_idx)] / d_val
    sub["R_j"] = R_j[sub["fac_idx"].values]
    sub["contrib"] = sub["R_j"] * sub["w_ij"]
    Ai = np.zeros(len(bldg))
    agg = sub.groupby("bldg_idx")["contrib"].sum()
    for b_idx, val in agg.items():
        Ai[int(b_idx)] = val
    return Ai

# --- 6. MULTI-THRESHOLD BASELINE G2SFCA ---
thresholds = [300, 400, 800, 1500, 3000]
baseline_rows = []
fac_areas_ex = parks_pt["Area_m2_clean"].values

for d0 in thresholds:
    Ai = run_g2sfca(od_existing, fac_areas_ex, d0, decay="gaussian", dist_col="net_dist")
    served_mask = Ai > 0
    served_bldg = int(served_mask.sum())
    served_pop = float(bldg.loc[served_mask, "pop"].sum())
    pct_pop = served_pop / total_pop * 100.0
    gini = weighted_gini(Ai, bldg["pop"].values)
    theil = weighted_theil(Ai, bldg["pop"].values)
    mean_ai = float(np.average(Ai, weights=bldg["pop"].values))
    mean_ai_served = float(np.average(Ai[served_mask], weights=bldg.loc[served_mask, "pop"].values)) if served_bldg > 0 else 0.0
    baseline_rows.append({
        "Threshold_m": d0,
        "Walk_Time_min_1.2ms": round(d0 / (1.2 * 60), 1),
        "Served_Centroids": served_bldg,
        "Pct_Centroids": round(served_bldg / len(bldg) * 100, 2),
        "Served_Pop": round(served_pop, 0),
        "Pct_Pop_Served": round(pct_pop, 2),
        "Unserved_Pop": round(total_pop - served_pop, 0),
        "Pct_Pop_Unserved": round(100 - pct_pop, 2),
        "PopWeighted_Mean_Ai": round(mean_ai, 4),
        "Served_Mean_Ai": round(mean_ai_served, 4),
        "Gini": round(gini, 4),
        "Theil": round(theil, 4),
    })

df_baseline = pd.DataFrame(baseline_rows)
df_baseline.to_csv(os.path.join(OUT_DIR, "baseline_multithreshold_g2sfca.csv"), index=False)
print("\n=== BASELINE MULTI-THRESHOLD G2SFCA ===")
print(df_baseline.to_string(index=False))

# --- 7. ABLATION ANALYSIS (1500m & 800m) ---
# (a) Euclidean G2SFCA vs Network G2SFCA
# Compute Euclidean OD <= 1500m across ALL buildings x parks
b_xy = np.column_stack([bldg.geometry.x, bldg.geometry.y])
p_xy = np.column_stack([parks_pt.geometry.x, parks_pt.geometry.y])
p_tree = cKDTree(p_xy)
euc_pairs = []
for b_idx in range(len(bldg)):
    idxs = p_tree.query_ball_point(b_xy[b_idx], r=1500.0)
    for f_idx in idxs:
        d = float(np.hypot(b_xy[b_idx, 0] - p_xy[f_idx, 0], b_xy[b_idx, 1] - p_xy[f_idx, 1]))
        euc_pairs.append((b_idx, f_idx, d))
od_euc_1500 = pd.DataFrame(euc_pairs, columns=["bldg_idx", "fac_idx", "euc_dist"])

ablation_rows = []
for d0 in [800, 1500]:
    Ai_net_g = run_g2sfca(od_existing, fac_areas_ex, d0, decay="gaussian", dist_col="net_dist")
    Ai_net_b = run_g2sfca(od_existing, fac_areas_ex, d0, decay="binary", dist_col="net_dist")
    Ai_euc_g = run_g2sfca(od_euc_1500, fac_areas_ex, d0, decay="gaussian", dist_col="euc_dist")
    for name, Ai in [("Network G2SFCA (Gaussian)", Ai_net_g), ("Network 2SFCA (Binary)", Ai_net_b), ("Euclidean G2SFCA (Gaussian)", Ai_euc_g)]:
        m = Ai > 0
        sp = float(bldg.loc[m, "pop"].sum())
        ablation_rows.append({
            "Threshold_m": d0,
            "Model": name,
            "Served_Centroids": int(m.sum()),
            "Served_Pop": round(sp, 0),
            "Pct_Pop_Served": round(sp / total_pop * 100, 2),
            "PopWeighted_Mean_Ai": round(float(np.average(Ai, weights=bldg["pop"].values)), 4),
            "Gini": round(weighted_gini(Ai, bldg["pop"].values), 4),
        })
df_ablation = pd.DataFrame(ablation_rows)
df_ablation.to_csv(os.path.join(OUT_DIR, "ablation_comparison.csv"), index=False)
print("\n=== ABLATION COMPARISON ===")
print(df_ablation.to_string(index=False))

# --- 8. LOCATION-ALLOCATION: LSCP (56 LARGE) & P-MEDIAN (198 SMALL) ---
# Build coverage matrix at 1500m (and 3000m for LSCP)
# First: which buildings are already covered by existing 156 parks within 1500m?
ex_cov_1500 = set(od_existing.loc[od_existing["net_dist"] <= 1500.0, "bldg_idx"].unique())
ex_cov_3000 = set(od_existing.loc[od_existing["net_dist"] <= 3000.0, "bldg_idx"].unique())
print(f"\nExisting parks cover {len(ex_cov_1500)} bldgs at 1500m, {len(ex_cov_3000)} bldgs at 3000m")

# LSCP on 56 Large Candidate Parcels (covering all reachable unserved demand within 3000m and 1500m)
large_ids = set(cand[cand["Type"] == "Large-scale UPGS"]["cand_id"].values)
small_ids = set(cand[cand["Type"] == "Small-scale UPGS"]["cand_id"].values)

def solve_lscp(cand_subset_ids, cutoff_m, ex_covered):
    sub_od = od_cand_all[(od_cand_all["fac_idx"].isin(cand_subset_ids)) & (od_cand_all["net_dist"] <= cutoff_m)]
    # Demand points that are NOT already covered by existing parks AND are reachable by at least one candidate in subset
    reachable_b = sorted(set(sub_od["bldg_idx"].unique()) - ex_covered)
    cand_list = sorted(list(cand_subset_ids))
    c_to_col = {c: j for j, c in enumerate(cand_list)}
    b_to_row = {b: i for i, b in enumerate(reachable_b)}
    A = np.zeros((len(reachable_b), len(cand_list)), dtype=float)
    for _, r in sub_od.iterrows():
        bi = int(r["bldg_idx"])
        fi = int(r["fac_idx"])
        if bi in b_to_row and fi in c_to_col:
            A[b_to_row[bi], c_to_col[fi]] = 1.0
    c_obj = np.ones(len(cand_list))
    constraints = LinearConstraint(A, lb=np.ones(len(reachable_b)), ub=np.full(len(reachable_b), np.inf))
    res = milp(c=c_obj, integrality=np.ones(len(cand_list)), bounds=Bounds(0, 1), constraints=constraints)
    chosen = [cand_list[j] for j, val in enumerate(res.x) if val > 0.5]
    cov_b = ex_covered | set(reachable_b)
    cov_pop = float(bldg.loc[list(cov_b), "pop"].sum())
    return chosen, len(reachable_b), cov_pop, cov_pop / total_pop * 100.0

lscp_3000, new_b_3000, pop_3000, pct_3000 = solve_lscp(large_ids, 3000.0, ex_cov_3000)
lscp_1500, new_b_1500, pop_1500, pct_1500 = solve_lscp(large_ids, 1500.0, ex_cov_1500)
print(f"LSCP (56 Large) @ 3000m: {len(lscp_3000)} facilities needed -> covers +{new_b_3000} bldgs, total {pct_3000:.2f}% pop ({pop_3000:,.0f})")
print(f"LSCP (56 Large) @ 1500m: {len(lscp_1500)} facilities needed -> covers +{new_b_1500} bldgs, total {pct_1500:.2f}% pop ({pop_1500:,.0f})")

# P-Median / Maximal Covering sweep on 198 Small Candidate Parcels (p = 5..50) @ 1500m
# Using fast greedy + 1-swap local search to minimize weighted distance (with penalty M=3000 for uncovered)
# Precompute for each building: best existing distance (capped at 1500m, else penalty=3000m)
b_best_ex = np.full(len(bldg), 3000.0)
for _, r in od_existing[od_existing["net_dist"] <= 1500.0].iterrows():
    bi = int(r["bldg_idx"])
    if r["net_dist"] < b_best_ex[bi]:
        b_best_ex[bi] = r["net_dist"]

# Dict of candidate -> (bldg_idx array, net_dist array) for <= 1500m
cand_reach_1500 = {}
sub_small_1500 = od_cand_all[(od_cand_all["fac_idx"].isin(small_ids)) & (od_cand_all["net_dist"] <= 1500.0)]
for fi, grp in sub_small_1500.groupby("fac_idx"):
    cand_reach_1500[int(fi)] = (grp["bldg_idx"].values.astype(int), grp["net_dist"].values.astype(float))

pop_arr = bldg["pop"].values
p_sweep_rows = []
selected_small_greedy = []
cur_dist = b_best_ex.copy()
remaining = set(cand_reach_1500.keys())

for step in range(1, 51):
    best_fi = None
    best_imp = -1.0
    best_new_dist = None
    for fi in remaining:
        b_idxs, d_vals = cand_reach_1500[fi]
        # reduction in weighted impedance
        old_d = cur_dist[b_idxs]
        improved = np.maximum(0.0, old_d - d_vals)
        imp = float(np.sum(pop_arr[b_idxs] * improved))
        if imp > best_imp:
            best_imp = imp
            best_fi = fi
    if best_fi is None:
        break
    selected_small_greedy.append(best_fi)
    remaining.remove(best_fi)
    b_idxs, d_vals = cand_reach_1500[best_fi]
    cur_dist[b_idxs] = np.minimum(cur_dist[b_idxs], d_vals)
    if step in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        cov_mask = cur_dist <= 1500.0
        cov_pop = float(pop_arr[cov_mask].sum())
        tot_imp = float(np.sum(pop_arr * cur_dist))
        p_sweep_rows.append({
            "p": step,
            "Covered_Centroids": int(cov_mask.sum()),
            "Covered_Pop": round(cov_pop, 0),
            "Pct_Pop_Covered_1500m": round(cov_pop / total_pop * 100, 2),
            "Total_Weighted_Impedance_10e6_m": round(tot_imp / 1e6, 2),
        })

df_psweep = pd.DataFrame(p_sweep_rows)
df_psweep.to_csv(os.path.join(OUT_DIR, "pmedian_small_sweep.csv"), index=False)
print("\n=== P-MEDIAN SWEEP (198 SMALL PARCELS @ 1500m) ===")
print(df_psweep.to_string(index=False))

# --- 9. POST-INTERVENTION G2SFCA & INEQUALITY (PRE vs POST) ---
# Scenario A: Existing (156) + p=30 Small Parcels
# Scenario B: Existing (156) + p=30 Small + LSCP Large (at 1500m)
sel_30_small = selected_small_greedy[:30]
sel_combined = list(set(sel_30_small) | set(lscp_1500))

def eval_post_intervention(sel_cand_ids, label):
    # Combine existing parks + selected candidate parcels
    od_sel = od_cand_all[od_cand_all["fac_idx"].isin(sel_cand_ids)].copy()
    od_sel["fac_idx"] = od_sel["fac_idx"] + len(parks_pt)  # offset fac_idx
    od_merged = pd.concat([od_existing, od_sel], ignore_index=True)
    # Build merged areas array
    merged_areas = np.zeros(len(parks_pt) + len(cand))
    merged_areas[:len(parks_pt)] = fac_areas_ex
    merged_areas[len(parks_pt):] = cand["Area_m2"].values
    rows = []
    for d0 in thresholds:
        Ai = run_g2sfca(od_merged, merged_areas, d0, decay="gaussian", dist_col="net_dist")
        m = Ai > 0
        sp = float(bldg.loc[m, "pop"].sum())
        rows.append({
            "Scenario": label,
            "Threshold_m": d0,
            "Served_Centroids": int(m.sum()),
            "Pct_Centroids": round(m.sum() / len(bldg) * 100, 2),
            "Served_Pop": round(sp, 0),
            "Pct_Pop_Served": round(sp / total_pop * 100, 2),
            "PopWeighted_Mean_Ai": round(float(np.average(Ai, weights=pop_arr)), 4),
            "Gini": round(weighted_gini(Ai, pop_arr), 4),
            "Theil": round(weighted_theil(Ai, pop_arr), 4),
        })
    return pd.DataFrame(rows), run_g2sfca(od_merged, merged_areas, 1500, decay="gaussian", dist_col="net_dist")

df_post_30, Ai_post_30_1500 = eval_post_intervention(sel_30_small, "Post_p30_Small")
df_post_comb, Ai_post_comb_1500 = eval_post_intervention(sel_combined, f"Post_Combined_p30_plus_{len(lscp_1500)}Large")
df_post_all = pd.concat([df_post_30, df_post_comb], ignore_index=True)
df_post_all.to_csv(os.path.join(OUT_DIR, "post_intervention_g2sfca.csv"), index=False)
print("\n=== POST-INTERVENTION G2SFCA ===")
print(df_post_all.to_string(index=False))

# --- 10. ZONAL STATISTICS & REGRESSION TREND DIAGNOSTICS ---
Ai_base_1500 = run_g2sfca(od_existing, fac_areas_ex, 1500, decay="gaussian", dist_col="net_dist")
bldg["Ai_base_1500"] = Ai_base_1500
bldg["Ai_post30_1500"] = Ai_post_30_1500
bldg["Ai_postComb_1500"] = Ai_post_comb_1500

# Spatial join to 5 districts
b_dist = gpd.sjoin(bldg, districts[["Name_EN", "Name_AR", "geometry"]], how="left", predicate="within")
zonal_rows = []
for dname, grp in b_dist.groupby("Name_EN", dropna=False):
    w = grp["pop"].values
    zonal_rows.append({
        "District": str(dname),
        "Centroids": len(grp),
        "Pop": round(w.sum(), 0),
        "Base_Served_Pct": round(grp.loc[grp["Ai_base_1500"] > 0, "pop"].sum() / w.sum() * 100, 2),
        "Post30_Served_Pct": round(grp.loc[grp["Ai_post30_1500"] > 0, "pop"].sum() / w.sum() * 100, 2),
        "PostComb_Served_Pct": round(grp.loc[grp["Ai_postComb_1500"] > 0, "pop"].sum() / w.sum() * 100, 2),
        "Base_Mean_Ai": round(np.average(grp["Ai_base_1500"], weights=w), 4),
        "Post30_Mean_Ai": round(np.average(grp["Ai_post30_1500"], weights=w), 4),
        "PostComb_Mean_Ai": round(np.average(grp["Ai_postComb_1500"], weights=w), 4),
    })
df_zonal = pd.DataFrame(zonal_rows)
df_zonal.to_csv(os.path.join(OUT_DIR, "zonal_district_stats.csv"), index=False)
print("\n=== ZONAL DISTRICT STATS (1500m) ===")
print(df_zonal.to_string(index=False))

# Polynomial regression statistics (XZ and YZ) on Ai_base_1500
x_km = (bldg.geometry.x - bldg.geometry.x.mean()) / 1000.0
y_km = (bldg.geometry.y - bldg.geometry.y.mean()) / 1000.0
for axis_name, coord in [("XZ (East-West)", x_km), ("YZ (South-North)", y_km)]:
    X = np.column_stack([coord, coord**2])
    X = sm.add_constant(X)
    model = sm.OLS(Ai_base_1500, X).fit()
    print(f"\nOLS Quadratic Trend {axis_name}: R2={model.rsquared:.4f}, F_pval={model.f_pvalue:.4e}")
    print(f"  params: b0={model.params[0]:.4f} (p={model.pvalues[0]:.4e}), b1={model.params[1]:.4f} (p={model.pvalues[1]:.4e}), b2={model.params[2]:.4f} (p={model.pvalues[2]:.4e})")

# Save building-level results for plotting
bldg[["bldg_id", "pop", "Ai_base_1500", "Ai_post30_1500", "Ai_postComb_1500"]].to_csv(
    os.path.join(OUT_DIR, "buildings_audit_results.csv"), index=False
)
print("\nSaved all audit CSVs to", OUT_DIR)
