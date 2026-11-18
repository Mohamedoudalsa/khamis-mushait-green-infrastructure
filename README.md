# Replication Archive: Green Infrastructure Optimization

This repository provides the replication datasets, topological rules, optimization solver settings, and Python code for:

> **Article**: Integrating Advanced Spatial Analytics and Predictive Location-Allocation Modeling for Green Infrastructure Optimization in Rapidly Growing Cities  
> **Journal**: *Applied Spatial Analysis and Policy* (Springer Nature)  
> **Submission ID**: 946c157a-c687-4efd-9acf-5ddb3a0d9b44  

This repository fulfills the open science requirements of **Reviewer 1 (Major Concern #17)** by depositing the six core analytical components:

`
├── centroids/         # 7,430 residential demand centroids with population weights (.geojson, .csv)
├── gates/             # Geocoded park perimeter pedestrian ingress gates (.geojson, .csv)
├── parcels/           # 254 candidate municipal parcels: 56 Large, 198 Small (.geojson, .csv)
├── network_rules/     # Walking speed (1.2 m/s), street filtering, and wadi barrier rules (.json)
├── solver_settings/   # Multi-threshold G2SFCA, LSCP (MILP), and P-Median solver parameters (.json)
└── code/              # Complete reproducible Python spatial audit & optimization pipeline (.py)
`

---

## 1. Centroids (centroids/)
- Contains the 7,430 disaggregated residential centroids evaluating 365,467 residents across the contiguous urban core of Khamis Mushait.
- Attributes: centroid_id, population (dasymetric WorldPop / building census weight), x_utm38n, y_utm38n, geometry.

## 2. Gates (gates/)
- Contains geocoded perimeter pedestrian ingress gates across existing urban parks.
- Attributes: gate_id, park_area_m2, x_utm38n, y_utm38n, geometry.

## 3. Parcels (parcels/)
- Contains the 254 vacant municipal plots stratified by size and functional planning role:
  - **Large-scale parcels** (56 plots, $\ge 1.50$ ha, total 2.24 km²) allocated via LSCP.
  - **Small-scale parcels** (198 plots, $< 1.50$ ha, total 1.04 km²) allocated via P-Median.
- Attributes: parcel_id, 	ype, rea_m2, rea_ha, geometry.

## 4. Network Rules (
etwork_rules/)
- Standard pedestrian velocity: .2$ m/s (.32$ km/h; WHO, 2016).
- Included corridors: residential, living-street, pedestrian, footway, service, secondary, tertiary, paved sidewalks.
- Excluded links: motorways, motorway links, trunk links, construction.
- Natural wadi barriers: 176 incised drainage channels modeled as impassable unless intersecting designated bridge overpasses.
- Route circuity: median .44$ (IQR .30 - 1.72$), mean .69$ across 58,382 OD pairs $\le 3$ km.

## 5. Solver Settings (solver_settings/)
- **G2SFCA**: Continuous Gaussian impedance decay with $\beta = 1,500$ m across $ m, $ m, $ m, ,500$ m, and ,000$ m catchments.
- **LSCP**: Mixed-Integer Linear Programming (MILP) solved via HiGHS branch-and-cut minimizing facility count subject to full demand coverage.
- **P-Median**: Impedance minimization sweep ( = 5$ to $ in increments of 5), identifying the knee inflection point at  = 30$ (.11\%$ coverage).
- **Combined Portfolio**: 30 small + 41 large parcels achieving .78\%$ coverage, mean supply of .26$ m²/person, Gini of .7780$, and Theil of .4913$.

## 6. Code (code/)
- 
un_full_spatial_audit.py: End-to-end Python pipeline executing NetworkX network shortest paths, G2SFCA accessibility scores, LSCP MILP optimization, P-Median greedy/exact selection, and population-weighted Gini & Theil inequality metrics.

---

## 📜 License
Released under the **MIT License** for academic research and replication.
