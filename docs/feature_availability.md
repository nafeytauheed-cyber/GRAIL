# Feature availability at initial reporting

Primary task: estimate the disruption grade of an already reported incident. The dataset does not contain timestamped first-report snapshots, so actual first-report availability is not proven. This is a conservative retrospective proxy, not validated pre-incident forecasting.

Every original column is accounted for below. Unknown columns are excluded automatically by an explicit allowlist. No global imputation, target encoding, feature selection or resampling precedes splitting.

| Column(s) | Primary handling | Reason / caveat |
|---|---|---|
| ID | Metadata only | Incident identity, duplicate checks, split manifest; not a feature. |
| Source | Audit only; explicit shortcut experiment | Provider may encode reporting policies, label regimes and geography. |
| Severity | Target, four ordered classes | Traffic impact, not injuries. Never an input. |
| Start_Time | Cyclic hour, weekday, month | Plausibly known when reporting begins; local-time convention. Year only in shortcut experiment. |
| End_Time | Excluded | Outcome/end information; used only for audit, never row selection by duration. |
| Start_Lat, Start_Lng | Included; removable together with State | Initial location plausible; coordinates may memorize geography. No outcome-based spatial smoothing. |
| End_Lat, End_Lng | Excluded | May reflect eventual affected segment. |
| Distance(mi) | Excluded | Affected extent may be updated and proxies disruption. |
| Description | Excluded | Text may contain outcomes, closures, final duration or severity cues; no version history. |
| Street, City, County, Zipcode | Excluded | High-cardinality memorization; uncertainty about updates. State/coordinates suffice for initial geography ablation. |
| State | Included; also group metadata | Administrative context, not a causal domain. Holdout states absent from fitted encoder vocabulary. |
| Country | Excluded | Constant/redundant. |
| Timezone | Audit metadata; not a model feature | Cross-state comparisons use local calendar boundaries, not synchronized UTC. |
| Airport_Code | Excluded | Weather-station identity can proxy provider/location. |
| Weather_Timestamp | Availability mask only | Retain weather only if observation is at/before Start_Time and no more than 120 minutes old. This cannot verify transmission delay. Missing timestamps mask all weather. Never forward-fill across incidents. |
| Temperature(F), Wind_Chill(F), Humidity(%), Pressure(in), Visibility(mi), Wind_Speed(mph), Precipitation(in) | Included only under weather mask; train-only median plus missing indicators | Observation values plausibly available before reporting. Missing precipitation is not assumed to mean dry weather. Outlier validity needs full-data audit. |
| Wind_Direction, Weather_Condition | Included under same weather mask | Train-only categorical vocabulary, missing token, unseen categories ignored. |
| Amenity, Bump, Crossing, Give_Way, Junction, No_Exit, Railway, Roundabout, Station, Stop, Traffic_Calming, Traffic_Signal, Turning_Loop | Included as binary/unknown | Map context plausibly available, but historical map snapshots not supplied. Turning_Loop may be constant; no claim that presence causes disruption. |
| Sunrise_Sunset, Civil_Twilight, Nautical_Twilight, Astronomical_Twilight | Included | Astronomical context calculable from location/time; correlated encodings. |
| Duration / target-derived flags / arbitrary new columns | Excluded | Explicitly not present in allowlist. |

Weather perturbation diagnostics change temperature by at most 2°F, humidity by 3 percentage points, visibility by 0.5 miles and wind by 1 mph. They preserve missingness, avoid moving already implausible values, and respect bounds. These are local sensitivity scenarios, not a measured sensor-noise model. Correlated weather consistency is not guaranteed, which limits interpretation.
