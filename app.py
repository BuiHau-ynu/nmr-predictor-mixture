import io
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors, rdMolDescriptors
from itertools import product

st.set_page_config(page_title="¹H NMR Structure Predictor", layout="wide")

# ==========================================
# 1. Load Model & QM Database (Cached)
# ==========================================
@st.cache_resource
def load_assets():
    model_bundle = joblib.load('nmr_rf_model.pkl')
    qm_database = joblib.load('qm_database.pkl')
    return (
        model_bundle['model'],
        model_bundle['module_names'],
        model_bundle.get('cv_accuracy', None),
        qm_database
    )

try:
    rf_model, module_names, cv_accuracy, qm_database = load_assets()
except Exception as e:
    st.error(f"Could not load model files (`nmr_rf_model.pkl` / `qm_database.pkl`). Error: {e}")
    st.stop()

j_cols = ['J1_Hz', 'J2_Hz', 'J3_Hz', 'J4_Hz']

# ==========================================
# 2. Core Pipeline Functions
# ==========================================
def _estimate_unit_1h(integrals):
    """Estimate the 1H base unit (approximate GCD) from a compound's anchor integrals."""
    vals = sorted([float(x) for x in integrals if pd.notna(x) and float(x) > 0])
    if not vals:
        return 1.0
    min_val = vals[0]
    # Check if the smallest anchor corresponds to 1H, 2H, or 3H to best fit all anchors
    best_u, best_err = min_val, float("inf")
    for div in [1, 2, 3]:
        u = min_val / div
        ratios = [v / u for v in vals]
        err = sum(abs(r - round(r)) / max(round(r), 1) for r in ratios)
        if err < best_err and all(round(r) >= 1 for r in ratios):
            best_err, best_u = err, u
    return best_u


def _best_int_error(val, u, max_n=6):
    """Return the minimum relative error between val/u and a positive integer."""
    if u <= 0:
        return 1.0, 0
    n = int(round(val / u))
    if n < 1 or n > max_n:
        return 1.0, 0
    return abs(val / u - n) / n, n


def deconvolve_excel_mixture(df, tol=0.15):
    """
    Deconvolve a 1H NMR Excel peak table into {compound_id: normalized_df}.
    - Empty Compound_ID column -> Auto-Anchor Mode (grid search for major/minor ratio).
    - Partially filled Compound_ID (>= 2 unique IDs) -> Guided-Anchor Mode.
    - Fully filled Compound_ID -> Direct grouping and 1H normalization.
    """
    work_df = df.copy()
    if "Compound_ID" not in work_df.columns:
        work_df["Compound_ID"] = ""

    # Normalize Compound_ID column (convert NaN/whitespace to empty string)
    work_df["Compound_ID"] = (
        work_df["Compound_ID"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": "", "Unknown_Sample": "", "Unknown": ""})
    )
    work_df["Integral"] = pd.to_numeric(work_df["Integral"], errors="coerce")

    labeled = work_df[work_df["Compound_ID"] != ""]
    unlabeled = work_df[work_df["Compound_ID"] == ""]
    unique_ids = labeled["Compound_ID"].unique().tolist()

    # CASE 1: Single compound (all rows share 1 ID)
    if len(unique_ids) == 1 and len(unlabeled) == 0:
        return {unique_ids[0]: work_df}

    # CASE 2: Pre-separated mixture (all rows labeled with >= 2 IDs)
    if len(unique_ids) >= 2 and len(unlabeled) == 0:
        result = {}
        for cid in unique_ids:
            sub = work_df[work_df["Compound_ID"] == cid].copy()
            u = _estimate_unit_1h(sub["Integral"].dropna())
            sub["Integral"] = sub["Integral"] / u
            result[cid] = sub
        return result

    # CASE 3: Guided-Anchor Mode (partially labeled with >= 2 IDs)
    if len(unique_ids) >= 2 and len(unlabeled) > 0:
        units = {
            cid: _estimate_unit_1h(labeled[labeled["Compound_ID"] == cid]["Integral"].dropna())
            for cid in unique_ids
        }
        buckets = {cid: [labeled[labeled["Compound_ID"] == cid].copy()] for cid in unique_ids}

        for _, row in unlabeled.iterrows():
            val = row["Integral"]
            if pd.isna(val) or val <= 0:
                continue

            # Compare relative integer-multiple error against each compound
            scores = {cid: _best_int_error(val, u) for cid, u in units.items()}
            best_cid = min(scores, key=lambda k: scores[k][0])
            best_err, _ = scores[best_cid]

            if best_err <= tol:
                r = row.copy()
                r["Compound_ID"] = best_cid
                buckets[best_cid].append(pd.DataFrame([r]))
            else:
                # Check for peak overlap between the first two compounds (n1*u1 + n2*u2)
                c1, c2 = unique_ids[0], unique_ids[1]
                u1, u2 = units[c1], units[c2]
                matched_overlap = False
                for n1 in range(1, 5):
                    for n2 in range(1, 5):
                        est = n1 * u1 + n2 * u2
                        if abs(val - est) / est <= tol:
                            r1, r2 = row.copy(), row.copy()
                            r1["Compound_ID"], r1["Integral"], r1["Mixture_Overlap"] = c1, n1 * u1, 1
                            r2["Compound_ID"], r2["Integral"], r2["Mixture_Overlap"] = c2, n2 * u2, 1
                            buckets[c1].append(pd.DataFrame([r1]))
                            buckets[c2].append(pd.DataFrame([r2]))
                            matched_overlap = True
                            break
                    if matched_overlap:
                        break

        # Concatenate and normalize each compound to 1H = 1.0
        result = {}
        for cid in unique_ids:
            sub = pd.concat(buckets[cid], ignore_index=True)
            sub["Integral"] = sub["Integral"] / units[cid]
            result[cid] = sub
        return result

    # CASE 4: Auto-Anchor Mode (entire Compound_ID column is empty)
    # Filter clean peaks (non-solvent-overlapped, positive integral)
    clean_mask = (
        work_df["Integral"].notna()
        & (work_df["Integral"] > 0)
        & (pd.to_numeric(work_df.get("Solvent_Overlap", 0), errors="coerce").fillna(0) == 0)
    )
    clean_vals = work_df.loc[clean_mask, "Integral"].values
    if len(clean_vals) < 4:
        return {"Unknown_Sample": work_df}

    # Estimate major 1H base unit u1 and test if a single compound explains >= 85% of clean peaks
    u1 = float(np.median(clean_vals[clean_vals >= np.median(clean_vals)]))
    single_fit = sum(1 for v in clean_vals if _best_int_error(v, u1)[0] <= tol)
    if single_fit / len(clean_vals) >= 0.85:
        work_df["Integral"] = work_df["Integral"] / u1
        return {"Auto_Single": work_df}

    # Grid search for minor molar ratio r = u2 / u1 (0.15 to 0.85, avoiding ambiguous 1:2 ratio)
    best_u2, best_count = None, 0
    for r in np.arange(0.15, 0.86, 0.02):
        if abs(r - 0.5) < 0.06:
            continue
        u2_cand = u1 * r
        fit_count = sum(
            1
            for v in clean_vals
            if min(_best_int_error(v, u1)[0], _best_int_error(v, u2_cand)[0]) <= tol
        )
        if fit_count > best_count:
            best_count, best_u2 = fit_count, u2_cand

    if best_u2 is None or best_count <= single_fit + 1:
        work_df["Integral"] = work_df["Integral"] / u1
        return {"Auto_Single": work_df}

    # Split into Major / Minor components and normalize integrals to 1H = 1.0
    rows_major, rows_minor = [], []
    for _, row in work_df.iterrows():
        val = row["Integral"]
        if pd.isna(val) or val <= 0:
            continue
        e1, _ = _best_int_error(val, u1)
        e2, _ = _best_int_error(val, best_u2)
        if e1 <= e2:
            r = row.copy()
            r["Integral"] = val / u1
            rows_major.append(r)
        else:
            r = row.copy()
            r["Integral"] = val / best_u2
            rows_minor.append(r)

    return {
        "Auto_Major": pd.DataFrame(rows_major),
        "Auto_Minor": pd.DataFrame(rows_minor),
    }
def extract_spectroscopic_features(compound_df):
    """Extract 1H NMR spin-system features for trans-alkene, AA'BB' (coumaric), ABX (caffeic),
    and 4-state midfield (4.3-5.5 ppm) classification.
    Window 5.3-9.2 ppm matches the augmented training pipeline in main.py."""
    trans_peak_count = 0.0      # Rule 1: Peaks with J > 13 Hz (trans-alkene)
    trans_int_sum = 0.0         # Rule 1: Total integral of trans-alkene peaks

    aabb_2h_doublets = 0.0      # Rule 2: 2H doublets with J = 8-9 Hz (7.5-9.8 Hz)

    abx_ortho_only_1h = 0.0     # Rule 3a: 1H aromatic peak with J = 8-9 Hz only
    abx_meta_only_1h = 0.0      # Rule 3b: 1H aromatic peak with J = 1-3 Hz only
    abx_ortho_meta_1h = 0.0     # Rule 3c: 1H aromatic peak with BOTH J = 8-9 Hz and J = 1-3 Hz

    arom_4h_singlet_int = 0.0   # Symmetric hydroquinone 4H singlet/narrow multiplet
    total_ortho_doublets = 0.0  # All aromatic peaks with J = 8-9 Hz

    # 4-state midfield (4.3-5.5 ppm) trackers
    anomeric_peaks = 0
    anomeric_h = 0.0
    c6_ch2_peaks = []
    methoxy_peaks = []
    solvent_obscured_peaks = []
    bypass_sugar_filter = False

    j_cols = [
        c for c in compound_df.columns
        if str(c).lower().startswith('j') and any(ch.isdigit() for ch in str(c))
    ]

    for _, row in compound_df.iterrows():
        ppm = pd.to_numeric(row.get('Shift_ppm'), errors='coerce')
        if pd.isna(ppm):
            continue

        raw_js = []
        for c in j_cols:
            val = pd.to_numeric(row.get(c), errors='coerce')
            if pd.notna(val) and val >= 0.5:
                raw_js.append(float(val))
        row_js = sorted(raw_js, reverse=True)
        n_j = len(row_js)
        max_j = row_js[0] if row_js else 0.0

        is_solvent_ov = (pd.to_numeric(row.get('Solvent_Overlap', 0), errors='coerce') == 1)
        is_mix = (pd.to_numeric(row.get('Mixture_Overlap', 0), errors='coerce') == 1)
        raw_int = pd.to_numeric(row.get('Integral', 0), errors='coerce')
        eff_int = 1.0 if is_mix else (float(raw_int) if pd.notna(raw_int) else 0.0)

        # Midfield 4-state classification (4.3-5.5 ppm)
        if 4.3 <= ppm <= 5.5:
            # State 4: Solvent-obscured peak (Solvent_Overlap == 1 and N_J == 0)
            if is_solvent_ov and n_j == 0:
                solvent_obscured_peaks.append((float(ppm), eff_int))
                bypass_sugar_filter = True
                continue
            # State 1: Confirmed Anomeric Proton (N_J == 1 and J1 < 11.0 Hz)
            elif (not is_solvent_ov) and n_j == 1 and max_j < 11.0:
                anomeric_peaks += 1
                anomeric_h += eff_int
            # State 2: Confirmed Acylated C6-CH2 (N_J >= 2 or J1 >= 11.0 Hz)
            elif (not is_solvent_ov) and (n_j >= 2 or max_j >= 11.0):
                c6_ch2_peaks.append((float(ppm), eff_int, row_js))
            # State 3: Methoxy group -OCH3 (N_J == 0 and Integral ≈ 3H)
            elif (not is_solvent_ov) and n_j == 0 and (2.5 <= eff_int <= 3.5):
                methoxy_peaks.append((float(ppm), eff_int))

        # Skip solvent-overlapped peaks for aromatic/olefinic spin-system rules
        if is_solvent_ov:
            continue

        # Rule 1: Trans-olefinic double bond (J > 13 Hz in 5.3-9.2 ppm)
        if 5.3 <= ppm <= 9.2 and max_j > 13.0:
            trans_peak_count += 1.0
            trans_int_sum += eff_int

        # Rules 2 & 3: Aromatic spin-system patterns (5.3-9.2 ppm)
        if 5.3 <= ppm <= 9.2:
            has_ortho = any(7.5 <= j <= 9.8 for j in row_js)
            has_meta = any(1.0 <= j <= 3.5 for j in row_js)
            no_larger_j = all(j <= 10.5 for j in row_js)

            if has_ortho and no_larger_j:
                total_ortho_doublets += 1.0

            # Rule 2: AA'BB' system in coumaric -> pairs of 2H doublets (J = 8-9 Hz)
            if has_ortho and (not has_meta) and no_larger_j and (1.6 <= eff_int <= 2.5):
                aabb_2h_doublets += 1.0

            # Rule 3: ABX triad in caffeic -> three 1H signals
            if no_larger_j and (0.7 <= eff_int <= 1.35):
                if has_ortho and has_meta:
                    abx_ortho_meta_1h += 1.0
                elif has_ortho and not has_meta:
                    abx_ortho_only_1h += 1.0
                elif has_meta and not has_ortho:
                    abx_meta_only_1h += 1.0

            # Symmetric hydroquinone (~4H singlet/multiplet)
            if 5.7 <= ppm <= 7.9 and max_j <= 2.0 and (3.5 <= eff_int <= 4.5):
                arom_4h_singlet_int += eff_int

    has_trans_pair = 1.0 if (trans_peak_count >= 2 or trans_int_sum >= 1.8) else 0.0
    aabb_pair_count = float(int(aabb_2h_doublets) // 2)

    if trans_peak_count >= 1 and aabb_2h_doublets >= 2:
        coumaric_motif_score = 1.0
    elif trans_peak_count >= 1 and aabb_2h_doublets == 1:
        coumaric_motif_score = 0.5
    else:
        coumaric_motif_score = 0.0

    abx_triad_completeness = (
        min(abx_ortho_only_1h, 1.0)
        + min(abx_meta_only_1h, 1.0)
        + min(abx_ortho_meta_1h, 1.0)
    )
    has_full_abx = 1.0 if (abx_ortho_only_1h >= 1 and abx_meta_only_1h >= 1 and abx_ortho_meta_1h >= 1) else 0.0
    if trans_peak_count >= 1 and has_full_abx == 1.0:
        caffeic_motif_score = 1.0
    elif trans_peak_count >= 1 and abx_triad_completeness >= 2.0:
        caffeic_motif_score = 0.6
    else:
        caffeic_motif_score = 0.0

    # Keep exact 14 features to match the trained 858-feature Random Forest model
    domain_vec = np.array([
        trans_peak_count,
        trans_int_sum,
        has_trans_pair,
        aabb_2h_doublets,
        aabb_pair_count,
        coumaric_motif_score,
        abx_ortho_only_1h,
        abx_meta_only_1h,
        abx_ortho_meta_1h,
        abx_triad_completeness,
        has_full_abx,
        caffeic_motif_score,
        arom_4h_singlet_int,
        total_ortho_doublets
    ])

    domain_info = {
        'trans_peaks': int(trans_peak_count),
        'trans_int': trans_int_sum,
        'aabb_2h_doublets': int(aabb_2h_doublets),
        'aabb_pairs': int(aabb_pair_count),
        'abx_ortho_1h': int(abx_ortho_only_1h),
        'abx_meta_1h': int(abx_meta_only_1h),
        'abx_dd_1h': int(abx_ortho_meta_1h),
        'abx_score': int(abx_triad_completeness),
        'has_full_abx': bool(has_full_abx == 1.0),
        # 4-state midfield outputs for Stage 1 & Stage 2
        'anomeric_peaks': anomeric_peaks,
        'anomeric_h': anomeric_h,
        'c6_ch2_peaks': c6_ch2_peaks,
        'methoxy_peaks': methoxy_peaks,
        'solvent_obscured_peaks': solvent_obscured_peaks,
        'bypass_sugar_filter': bypass_sugar_filter,
    }
    return domain_vec, domain_info

def create_nmr_vector(compound_df, min_ppm=0.0, max_ppm=12.0, bin_size=0.1):
    bins = np.arange(min_ppm, max_ppm + bin_size, bin_size)
    j_count_vector = np.zeros(len(bins) - 1)
    j1_vector = np.zeros(len(bins) - 1)
    j2_vector = np.zeros(len(bins) - 1)
    j3_vector = np.zeros(len(bins) - 1)
    j4_vector = np.zeros(len(bins) - 1)
    integral_vector = np.zeros(len(bins) - 1)
    width_vector = np.zeros(len(bins) - 1)

    aromatic_int, anomeric_int, sugar_body_int, peak_count = 0.0, 0.0, 0.0, 0.0

    for _, row in compound_df.iterrows():
        if row.get('Solvent_Overlap', 0) == 1:
            continue
        ppm = row['Shift_ppm']
        if pd.notna(ppm) and isinstance(ppm, (int, float)):
            peak_count += 1.0
            effective_int = 1.0 if row.get('Mixture_Overlap', 0) == 1 else float(row['Integral'])

            row_js = [float(row[c]) for c in j_cols if c in row and pd.notna(row[c]) and float(row[c]) > 0]

            if 6.2 <= ppm <= 8.5:
                aromatic_int += effective_int
            elif 4.3 <= ppm <= 5.5:
                # Anomeric proton rule: strictly 1 coupling constant (doublet) and J < 11.5 Hz
                if len(row_js) == 1 and row_js[0] < 11.5:
                    anomeric_int += effective_int
                else:
                    # Downfield-shifted CH2 (e.g., acylated C6-CH2 with geminal J >= 11.5 Hz or dd)
                    sugar_body_int += effective_int
            elif 3.0 <= ppm < 4.3:
                sugar_body_int += effective_int

            bin_idx = np.digitize(ppm, bins) - 1
        if 0 <= bin_idx < len(bins) - 1:
            valid_js = []
            for c in j_cols:
                if c in row and pd.notna(row[c]):
                    try:
                        val = float(row[c])
                        if val >= 0.5:
                            valid_js.append(val)
                    except (ValueError, TypeError):
                        pass
            valid_js.sort(reverse=True)

            if len(valid_js) > 0:
                if len(valid_js) > j_count_vector[bin_idx]:
                    j_count_vector[bin_idx] = len(valid_js)
                if len(valid_js) >= 1 and valid_js[0] > j1_vector[bin_idx]:
                    j1_vector[bin_idx] = valid_js[0]
                if len(valid_js) >= 2 and valid_js[1] > j2_vector[bin_idx]:
                    j2_vector[bin_idx] = valid_js[1]
                if len(valid_js) >= 3 and valid_js[2] > j3_vector[bin_idx]:
                    j3_vector[bin_idx] = valid_js[2]
                if len(valid_js) >= 4 and valid_js[3] > j4_vector[bin_idx]:
                    j4_vector[bin_idx] = valid_js[3]
            elif row.get('Is_Multiplet', False) and j_count_vector[bin_idx] == 0.0:
                j_count_vector[bin_idx] = -1.0

            integral_vector[bin_idx] += effective_int
            width_val = float(row['Multiplet_Width_Hz']) if ('Multiplet_Width_Hz' in row and pd.notna(row['Multiplet_Width_Hz'])) else 0.0
            if width_val > width_vector[bin_idx]:
                width_vector[bin_idx] = width_val

    regional_features = np.array([aromatic_int, anomeric_int, sugar_body_int, peak_count])
    domain_vec, domain_info = extract_spectroscopic_features(compound_df)
    return np.concatenate([j_count_vector, j1_vector, j2_vector, j3_vector, j4_vector, integral_vector, width_vector, regional_features, domain_vec]), regional_features, domain_info

def get_sorted_j_values(row):
    return sorted([float(row[c]) for c in j_cols if c in row and pd.notna(row[c]) and float(row[c]) > 0], reverse=True)

def calculate_j_penalty(actual_js, ref_js, actual_is_m=False, ref_is_m=False):
    if actual_is_m and ref_is_m:
        return 0.0
    if (actual_is_m and len(ref_js) > 0) or (ref_is_m and len(actual_js) > 0):
        return 0.3
    if actual_is_m or ref_is_m:
        return 2.0
    if len(actual_js) == 0 and len(ref_js) == 0:
        return 0.0
    if len(actual_js) == 0 or len(ref_js) == 0:
        return 2.0

    min_len, max_len = min(len(actual_js), len(ref_js)), max(len(actual_js), len(ref_js))
    avg_diff = sum(abs(actual_js[i] - ref_js[i]) for i in range(min_len)) / min_len
    count_diff_penalty = (max_len - min_len) * 0.4
    return (5.0 + avg_diff * 0.5 + count_diff_penalty) if avg_diff > 1.5 else (avg_diff * 0.3 + count_diff_penalty)

def auto_match_and_analyze(complex_df, active_modules, qm_db, min_delta=0.3, max_delta=1.5):
    module_list = list(set([m.strip().lower() for m in active_modules]))
    relevant_qm = qm_db[qm_db['Proton_ID'].apply(
        lambda pid: ('overlap' not in str(pid).lower()) and any(mod in str(pid).lower() for mod in module_list)
    )].copy()

    if relevant_qm.empty:
        return []

    found_linkages = []
    for _, row in complex_df.iterrows():
        if row['Solvent_Overlap'] == 1:
            continue
        actual_shift = row['Shift_ppm']
        actual_int = row['Integral']
        actual_width = row['Multiplet_Width_Hz']
        actual_js = get_sorted_j_values(row)
        actual_is_m = bool(row.get('Is_Multiplet', False))
        is_mix_overlap = (row['Mixture_Overlap'] == 1)

        best_match, lowest_cost = None, float('inf')
        for _, ref_row in relevant_qm.iterrows():
            delta_shift = actual_shift - ref_row['Shift_ppm']
            abs_delta = abs(delta_shift)
            if abs_delta > max_delta:
                continue

            ref_js = get_sorted_j_values(ref_row)
            ref_is_m = bool(ref_row.get('Is_Multiplet', False))
            shift_cost = abs_delta * 2.0
            j_cost = calculate_j_penalty(actual_js, ref_js, actual_is_m, ref_is_m)

            if is_mix_overlap:
                int_cost = 0.0
            else:
                rel_int_err = abs(actual_int - ref_row['Integral']) / max(ref_row['Integral'], 1.0)
                int_cost = rel_int_err * 0.5 if rel_int_err <= 0.30 else 3.0

            width_cost = abs(actual_width - ref_row['Multiplet_Width_Hz']) * 0.02 if (actual_width > 0 and ref_row['Multiplet_Width_Hz'] > 0) else 0.0
            total_cost = shift_cost + j_cost + int_cost + width_cost

            if total_cost < lowest_cost:
                lowest_cost = total_cost
                j_str = "m" if actual_is_m else (", ".join(f"{j:.1f}" for j in actual_js) if actual_js else "s")
                ref_j_str = ", ".join(f"{j:.1f}" for j in ref_js) if ref_js else "s"
                best_match = {
                    'Exp. Shift (ppm)': round(actual_shift, 2),
                    'Exp. J (Hz)': j_str,
                    'Integral': round(actual_int, 2),
                    'Matched Ref': ref_row['Proton_ID'],
                    'Ref Shift (ppm)': round(ref_row['Shift_ppm'], 2),
                    'Ref J (Hz)': ref_j_str,
                    'Delta (ppm)': f"{delta_shift:+.2f}",
                    'Effect': "Deshielding" if delta_shift > 0 else "Shielding",
                    'Note': "Mix-Overlap (Int. Ignored)" if is_mix_overlap else "",
                    '_abs_delta': abs_delta,
                    '_cost': total_cost
                }

        if best_match and best_match['_abs_delta'] >= min_delta and best_match['_cost'] < 6.0:
            found_linkages.append({k: v for k, v in best_match.items() if k != '_abs_delta'})
    # Rule: Downfield-shifted CH2 in 4.3–5.5 ppm (non-anomeric: >= 2 J-values or J >= 11.5 Hz) -> Acylation at Sugar-C6
    SUGAR_KEYWORDS = ('glc', 'gal', 'man', 'ara', 'xyl', 'rha', 'fru', 'sugar')
    active_sugars = [m for m in module_list if any(k in m for k in SUGAR_KEYWORDS)]

    if active_sugars:
        c6_ch2_peaks = []
        c6_ch2_js = []
        c6_ch2_int = 0.0
        for _, row in complex_df.iterrows():
            if row.get('Solvent_Overlap', 0) == 1:
                continue
            ppm = row['Shift_ppm']
            if pd.notna(ppm) and isinstance(ppm, (int, float)) and (4.3 <= ppm <= 5.5):
                row_js = [float(row[c]) for c in j_cols if c in row and pd.notna(row[c]) and float(row[c]) > 0]
                is_anomeric = (len(row_js) == 1 and row_js[0] < 11.5)
                if not is_anomeric:
                    eff_int = 1.0 if row.get('Mixture_Overlap', 0) == 1 else float(row['Integral'])
                    c6_ch2_peaks.append(float(ppm))
                    c6_ch2_js.extend(row_js)
                    c6_ch2_int += eff_int

        if c6_ch2_int >= 1.5:
            avg_ppm = float(np.mean(c6_ch2_peaks))
            ref_c6 = 3.75  # Typical unacylated Sugar-H6 shift
            delta_c6 = avg_ppm - ref_c6
            j_disp = ", ".join(f"{j:.1f}" for j in sorted(set(c6_ch2_js), reverse=True)[:2]) if c6_ch2_js else "m"

            for sugar_mod in active_sugars:
                already_has_c6 = any(
                    sugar_mod in str(lk.get('Matched Ref', '')).lower() and '6' in str(lk.get('Matched Ref', ''))
                    for lk in found_linkages
                )
                if not already_has_c6:
                    found_linkages.append({
                        'Exp. Shift (ppm)': round(avg_ppm, 2),
                        'Exp. J (Hz)': j_disp,
                        'Integral': round(c6_ch2_int, 2),
                        'Matched Ref': f'{sugar_mod}_H6',
                        'Ref Shift (ppm)': ref_c6,
                        'Ref J (Hz)': '12.0, 5.0',
                        'Delta (ppm)': f"{delta_c6:+.2f}",
                        'Effect': 'Deshielding',
                        'Note': 'Downfield C6-CH2 (non-anomeric)',
                        '_cost': 0.5
                    })
    return found_linkages

def select_modules_by_threshold(rf_model, module_names, proba_list, threshold=70.0):
    """Select modules whose total presence probability P(count >= 1) exceeds threshold,
    while retaining full probability breakdowns for all modules."""
    module_counts = {}
    module_confs = {}
    stage1_summary = []

    for i, m_name in enumerate(module_names):
        classes = rf_model.estimators_[i].classes_
        probs = proba_list[i][0]

        # Total probability that this module is present (count >= 1)
        presence_prob = float(sum(p for c, p in zip(classes, probs) if c > 0) * 100.0)

        # Find most probable non-zero count
        nz_pairs = [(int(c), float(p * 100.0)) for c, p in zip(classes, probs) if c > 0]
        if nz_pairs:
            best_nz_count, _ = max(nz_pairs, key=lambda x: x[1])
        else:
            best_nz_count = 0

        # Format vote breakdown string (e.g., "1x (75%) | 0x (25%)")
        sorted_votes = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        breakdown_str = " | ".join(f"{int(c)}x ({p*100:.0f}%)" for c, p in sorted_votes if p >= 0.01)

        if presence_prob > threshold and best_nz_count > 0:
            selected_cnt = best_nz_count
            status = f"✅ Selected (>{threshold:.0f}%)"
        elif presence_prob >= 20.0:
            selected_cnt = 0
            status = "⚠️ Candidate (Consider)"
        else:
            selected_cnt = 0
            status = "➖ Low (<20%)"

        module_counts[m_name] = selected_cnt
        module_confs[m_name] = round(presence_prob, 1)
        stage1_summary.append({
            "Module": m_name,
            "Presence Prob.": f"{presence_prob:.1f}%",
            "Vote Breakdown": breakdown_str,
            "Assembled": f"{selected_cnt}x" if selected_cnt > 0 else "—",
            "Status": status
        })

    return module_counts, module_confs, stage1_summary

AROMATIC_H_MAP = {'hydroquinone': 4.0, 'caffeic': 5.0, 'coumaric': 6.0}

def apply_stoichiometric_filter(module_counts, module_confidences, reg_feats, bypass_sugar_filter=False):
    aromatic_int, anomeric_int, sugar_body_int, _ = reg_feats
    filtered = dict(module_counts)
    adjustments = []

    if bypass_sugar_filter:
        adjustments.append("Bypassed sugar trimming filter (solvent-obscured peak in 4.3–5.5 ppm with N_J=0)")
    elif filtered.get('bglc', 0) > 0:
        max_glc = int(round(anomeric_int)) if anomeric_int >= 0.5 else int(round(sugar_body_int / 6.0))
        if filtered['bglc'] > max_glc:
            adjustments.append(f"bglc: {filtered['bglc']}x → {max_glc}x (Anomeric: {anomeric_int:.1f}H, Sugar Body: {sugar_body_int:.1f}H)")
            filtered['bglc'] = max_glc

    arom_candidates = sorted(
        [(m, filtered[m], module_confidences.get(m, 0.0)) for m in AROMATIC_H_MAP if filtered.get(m, 0) > 0],
        key=lambda x: x[2], reverse=True
    )
    remaining_h, tolerance = aromatic_int, 1.2
    for m_name, orig_cnt, _ in arom_candidates:
        h_needed = AROMATIC_H_MAP[m_name]
        allowed = 0
        while allowed < orig_cnt and (remaining_h + tolerance) >= h_needed:
            allowed += 1
            remaining_h -= h_needed
        if allowed != orig_cnt:
            adjustments.append(f"{m_name}: {orig_cnt}x → {allowed}x (Aromatic budget: {aromatic_int:.1f}H)")
            filtered[m_name] = allowed
    return filtered, adjustments
from itertools import combinations_with_replacement


def resolve_solvent_obscured_hypotheses(
    base_counts,
    module_confs,
    module_names,
    domain_info,
    reg_feats,
    calc_mw_fn,
    active_mw=None,
    mw_tol=2.0,
):
    """
    Resolve solvent-obscured peaks (Solvent_Overlap == 1, N_J == 0 in 4.3-5.5 ppm)
    using a clean baseline M_clean locked to confirmed clean anomeric H (N_J == 1, J < 11.0 Hz).

    Preserves all anomeric rules:
      - Lower bound ALWAYS enforced: total sugar units >= true_anomeric_h.
      - Avoids double-counting sugars already predicted by RF when upper bound was bypassed.
      - Evaluates all 3 physical assignments (Acylated C6-CH2, Anomeric, Methoxy) per obscured peak.
      - Without MW: keeps RF prediction (base_counts) and exposes all hypotheses.
      - With MW: filters hypotheses by exact MW tolerance and returns the best match.
    """
    obscured_peaks = domain_info.get("solvent_obscured_peaks", [])
    if not obscured_peaks:
        return base_counts, []

    SUGAR_KEYWORDS = ("glc", "gal", "man", "ara", "xyl", "rha", "fru", "sugar")
    METHOXY_KEYWORDS = ("ome", "methoxy", "och3", "methyl")

    all_sugar_mods = [m for m in module_names if any(k in m.lower() for k in SUGAR_KEYWORDS)]
    best_sugar = max(all_sugar_mods, key=lambda x: module_confs.get(x, 0.0)) if all_sugar_mods else None

    methoxy_mods = [m for m in module_names if any(k in m.lower() for k in METHOXY_KEYWORDS)]
    best_methoxy = max(methoxy_mods, key=lambda x: module_confs.get(x, 0.0)) if methoxy_mods else None

    # 1. Build clean baseline M_clean locked strictly to confirmed clean anomeric H
    true_anomeric_h = int(round(domain_info.get("anomeric_h", reg_feats[1])))
    clean_counts = dict(base_counts)
    current_sugars = sum(clean_counts.get(m, 0) for m in all_sugar_mods)
    trimmed_sugars = []  # Preserve RF's original sugar identities if trimmed down to M_clean

    if current_sugars < true_anomeric_h and best_sugar is not None:
        # Enforce Lower Bound on M_clean
        clean_counts[best_sugar] = clean_counts.get(best_sugar, 0) + (true_anomeric_h - current_sugars)
    elif current_sugars > true_anomeric_h:
        # Temporarily trim excess sugars (lowest RF confidence first) to prevent double-counting
        excess = current_sugars - true_anomeric_h
        active_sugars = [m for m in all_sugar_mods if clean_counts.get(m, 0) > 0]
        for m in sorted(active_sugars, key=lambda x: module_confs.get(x, 0.0)):
            while clean_counts.get(m, 0) > 0 and excess > 0:
                clean_counts[m] -= 1
                trimmed_sugars.append(m)
                excess -= 1
        # Reverse so highest-confidence trimmed sugars are restored first in Anomeric hypotheses
        trimmed_sugars.reverse()

    # 2. Generate all unique combinations of the 3 states across obscured peaks (cap at 3 peaks)
    n_obscured = min(len(obscured_peaks), 3)
    states = ("C6-CH2", "Anomeric", "Methoxy")
    hypotheses = []

    for combo in combinations_with_replacement(states, n_obscured):
        n_c6 = combo.count("C6-CH2")
        n_anom = combo.count("Anomeric")
        n_ome = combo.count("Methoxy")

        # Skip Anomeric hypotheses if the model has no sugar modules defined
        if n_anom > 0 and best_sugar is None:
            continue

        h_counts = dict(clean_counts)

        # Add Anomeric sugar units (restore RF-trimmed sugars first, then best_sugar)
        for i in range(n_anom):
            sugar_to_add = trimmed_sugars[i] if i < len(trimmed_sugars) else best_sugar
            h_counts[sugar_to_add] = h_counts.get(sugar_to_add, 0) + 1

        # Add Methoxy units (either as explicit module or +14.0157 Da O-methylation delta)
        extra_ome_mass = 0.0
        if n_ome > 0:
            if best_methoxy is not None:
                h_counts[best_methoxy] = h_counts.get(best_methoxy, 0) + n_ome
            else:
                extra_ome_mass = 14.0157 * n_ome

        est_mw = float(calc_mw_fn(h_counts)) + extra_ome_mass

        parts = []
        if n_c6:
            parts.append(f"{n_c6}x Acylated C6-CH2")
        if n_anom:
            parts.append(f"{n_anom}x Anomeric (+{n_anom} sugar)")
        if n_ome:
            parts.append(f"{n_ome}x Methoxy (-OCH3)")
        label = " + ".join(parts)

        total_h_sugars = sum(h_counts.get(m, 0) for m in all_sugar_mods)
        hypotheses.append({
            "label": label,
            "counts": h_counts,
            "est_mw": est_mw,
            "n_c6": n_c6,
            "n_anom": n_anom,
            "n_ome": n_ome,
            "note": (
                f"Total sugars = {total_h_sugars} "
                f"(clean anomeric: {true_anomeric_h} + obscured anomeric: {n_anom})."
            ),
        })

    # 3. Arbitration by Molecular Weight
    # Case A: No MW provided -> keep RF prediction (base_counts, already lower-bound protected)
    if active_mw is None or active_mw <= 0:
        return base_counts, hypotheses

    # Case B: MW provided -> filter hypotheses within tolerance and pick closest match
    matched = [h for h in hypotheses if abs(h["est_mw"] - active_mw) <= mw_tol]
    matched.sort(key=lambda h: abs(h["est_mw"] - active_mw))

    if matched:
        return matched[0]["counts"], matched
    return base_counts, []
def compute_overall_confidence(module_counts, module_confs, reg_feats, linkages):
    """Calculate a composite 0-100% confidence score for the predicted structure."""
    active = [m for m, c in module_counts.items() if c > 0]
    if not active:
        return 0.0, "Low"

    # 1. RF Model Confidence (50% weight)
    rf_score = float(np.mean([module_confs[m] for m in active]))

    # 2. Stoichiometric Proton Consistency (30% weight)
    expected_arom = sum(AROMATIC_H_MAP.get(m, 0.0) * module_counts[m] for m in active)
    expected_glc_anom = float(module_counts.get('bglc', 0))
    arom_err = abs(reg_feats[0] - expected_arom)
    anom_err = abs(reg_feats[1] - expected_glc_anom)
    stoich_score = max(0.0, 100.0 - (arom_err * 12.0 + anom_err * 20.0))

    # 3. QM Linkage Match Quality (20% weight)
    if linkages:
        avg_cost = float(np.mean([lk['_cost'] for lk in linkages]))
        qm_score = max(0.0, min(100.0, 100.0 - avg_cost * 12.0))
    else:
        qm_score = 75.0  # Neutral baseline when no shifted linkage peaks are present

    overall = round(0.5 * rf_score + 0.3 * stoich_score + 0.2 * qm_score, 1)
    tier = "High" if overall >= 80 else ("Moderate" if overall >= 60 else "Low")
    return overall, tier
# ==========================================
# Combinatorial 2D Structure Assembly & Scoring
# ==========================================
MODULE_TEMPLATES = {
    'hydroquinone': {
        'smiles': 'Oc1ccc(O)cc1',
        'donors': [],
        'acceptors': [
            (0, 1, 'Hydroquinone-OH'),
            (5, 4, 'Hydroquinone-OH')
        ]
    },
    'bglc': {
        'smiles': 'OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O',
        'donors': [
            ('glycosyl', 5, None, 'Glc-C1')
        ],
        'acceptors': [
            (0, 1, 'Glc-C6'),
            (7, 6, 'Glc-C2'),
            (9, 8, 'Glc-C3'),
            (11, 10, 'Glc-C4')
        ]
    },
    'caffeic': {
        'smiles': 'Oc1ccc(/C=C/C(=O)O)cc1O',
        'donors': [
            ('acyl', 7, 9, 'Caffeic-C=O')
        ],
        'acceptors': [
            (0, 1, 'Caffeic-C4-OH'),
            (12, 11, 'Caffeic-C3-OH')
        ]
    },
    'coumaric': {
        'smiles': 'Oc1ccc(/C=C/C(=O)O)cc1',
        'donors': [
            ('acyl', 7, 9, 'Coumaric-C=O')
        ],
        'acceptors': [
            (0, 1, 'Coumaric-C4-OH')
        ]
    }
}

SITE_PRIOR = {
    'Hydroquinone-OH': 1.00,
    'Glc-C6': 1.00,
    'Glc-C2': 0.93,
    'Glc-C4': 0.86,
    'Glc-C3': 0.83,
    'Caffeic-C4-OH': 1.00,
    'Caffeic-C3-OH': 0.90,
    'Coumaric-C4-OH': 1.00
}

def _match_linkage_to_nmr_evidence(formed_sites, linkages):
    """Score how well a candidate's specific linkage sites align with Stage 2 delta-shift evidence."""
    if not formed_sites:
        return 85.0, "Single monomer (no inter-module linkage)"

    # Extract shifted site hints from Stage 2 Matched Ref labels
    detected_hints = set()
    for lk in linkages:
        ref = str(lk.get('Matched Ref', '')).lower()
        if 'h6' in ref and 'glc' in ref:
            detected_hints.add('Glc-C6')
        if 'h2' in ref and 'glc' in ref:
            detected_hints.add('Glc-C2')
        if 'h3' in ref and 'glc' in ref:
            detected_hints.add('Glc-C3')
        if 'h4' in ref and 'glc' in ref:
            detected_hints.add('Glc-C4')
        if 'hydroquinone' in ref:
            detected_hints.add('Hydroquinone-OH')
        if 'caffeic' in ref:
            detected_hints.add('Caffeic-C4-OH')
            detected_hints.add('Caffeic-C3-OH')
        if 'coumaric' in ref:
            detected_hints.add('Coumaric-C4-OH')

    prior_scores = [SITE_PRIOR.get(s, 0.85) * 100.0 for s in formed_sites]
    base_regio = float(np.mean(prior_scores))

    if not detected_hints:
        return base_regio, "Regioselectivity prior (no specific CH Δδ tag)"

    matched = [s for s in formed_sites if s in detected_hints]
    match_ratio = len(matched) / len(formed_sites)
    regio_score = min(100.0, 0.55 * base_regio + 45.0 * match_ratio)

    if matched:
        evidence = f"Matches Δδ at {', '.join(sorted(set(matched)))}"
    else:
        evidence = "No direct Δδ match at linked positions"
    return regio_score, evidence

def enumerate_all_assemblies(module_counts, linkages, base_confidence, max_states=1500):
    """Generate all chemically valid regioisomers from predicted modules and rank by confidence."""
    mods = []
    for m_name, cnt in module_counts.items():
        if m_name in MODULE_TEMPLATES and cnt > 0:
            mods.extend([m_name] * int(cnt))

    n = len(mods)
    if n == 0:
        return []

    if n == 1:
        mol = Chem.MolFromSmiles(MODULE_TEMPLATES[mods[0]]['smiles'])
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        return [{
            'mol': mol,
            'smiles': smi,
            'confidence': round(base_confidence, 1),
            'pathway': mods[0],
            'evidence': "Single monomer structure"
        }]

    # Precompute all possible directed (donor -> acceptor) bonds between distinct module instances
    all_possible_bonds = []
    for d_mod in range(n):
        for d_idx, d_info in enumerate(MODULE_TEMPLATES[mods[d_mod]]['donors']):
            for a_mod in range(n):
                if d_mod == a_mod:
                    continue
                for a_idx, a_info in enumerate(MODULE_TEMPLATES[mods[a_mod]]['acceptors']):
                    all_possible_bonds.append((d_mod, d_idx, a_mod, a_idx))

    candidates_by_smiles = {}
    states_explored = 0

    def find_root(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def dfs(start_bond_idx, chosen_edges, used_donors, used_acceptors, parent):
        nonlocal states_explored
        if states_explored >= max_states:
            return

        if len(chosen_edges) == n - 1:
            states_explored += 1
            try:
                rw = Chem.RWMol()
                offsets = []
                curr_offset = 0
                for m_name in mods:
                    m_obj = Chem.MolFromSmiles(MODULE_TEMPLATES[m_name]['smiles'])
                    offsets.append(curr_offset)
                    rw = Chem.RWMol(Chem.CombineMols(rw, m_obj))
                    curr_offset += m_obj.GetNumAtoms()

                bonds_to_add = []
                atoms_to_remove = []
                pathway_parts = []
                formed_acceptor_sites = []

                for d_mod, d_idx, a_mod, a_idx in chosen_edges:
                    d_type, bond_atom, drop_atom, d_label = MODULE_TEMPLATES[mods[d_mod]]['donors'][d_idx]
                    acc_o, acc_c, a_label = MODULE_TEMPLATES[mods[a_mod]]['acceptors'][a_idx]

                    if d_type == 'glycosyl':
                        bonds_to_add.append((offsets[d_mod] + bond_atom, offsets[a_mod] + acc_c))
                        atoms_to_remove.append(offsets[a_mod] + acc_o)
                    elif d_type == 'acyl':
                        bonds_to_add.append((offsets[d_mod] + bond_atom, offsets[a_mod] + acc_o))
                        atoms_to_remove.append(offsets[d_mod] + drop_atom)

                    pathway_parts.append(f"{d_label} → {a_label}")
                    formed_acceptor_sites.append(a_label)

                for u, v in bonds_to_add:
                    rw.AddBond(u, v, Chem.BondType.SINGLE)
                for idx in sorted(set(atoms_to_remove), reverse=True):
                    rw.RemoveAtom(idx)

                final_mol = rw.GetMol()
                Chem.SanitizeMol(final_mol)
                can_smi = Chem.MolToSmiles(final_mol, isomericSmiles=True)

                regio_score, evidence = _match_linkage_to_nmr_evidence(formed_acceptor_sites, linkages)
                cand_conf = round(0.75 * base_confidence + 0.25 * regio_score, 1)
                pathway_str = " + ".join(sorted(pathway_parts))

                if can_smi not in candidates_by_smiles or cand_conf > candidates_by_smiles[can_smi]['confidence']:
                    candidates_by_smiles[can_smi] = {
                        'mol': final_mol,
                        'smiles': can_smi,
                        'confidence': cand_conf,
                        'pathway': pathway_str,
                        'evidence': evidence
                    }
            except Exception:
                pass
            return

        remaining_needed = (n - 1) - len(chosen_edges)
        for i in range(start_bond_idx, len(all_possible_bonds) - remaining_needed + 1):
            d_mod, d_idx, a_mod, a_idx = all_possible_bonds[i]
            if (d_mod, d_idx) in used_donors or (a_mod, a_idx) in used_acceptors:
                continue

            r1, r2 = find_root(parent, d_mod), find_root(parent, a_mod)
            if r1 == r2:
                continue  # Prevent cyclic assemblies

            new_parent = list(parent)
            new_parent[r1] = r2
            used_donors.add((d_mod, d_idx))
            used_acceptors.add((a_mod, a_idx))
            chosen_edges.append((d_mod, d_idx, a_mod, a_idx))

            dfs(i + 1, chosen_edges, used_donors, used_acceptors, new_parent)

            chosen_edges.pop()
            used_donors.remove((d_mod, d_idx))
            used_acceptors.remove((a_mod, a_idx))

    dfs(0, [], set(), set(), list(range(n)))
    ranked = sorted(candidates_by_smiles.values(), key=lambda x: x['confidence'], reverse=True)
    return ranked
# Monoisotopic (Exact) Mass of each monomer (Da)
MODULE_EXACT_MASS = {
    'hydroquinone': 110.0368,  # C6H6O2
    'bglc': 180.0634,          # C6H12O6
    'caffeic': 180.0423,       # C9H8O4
    'coumaric': 164.0473       # C9H8O3
}
H2O_EXACT_MASS = 18.010565

def calc_theoretical_mw(counts_dict):
    """Calculate neutral exact mass of an acyclic assembly of modules."""
    n_mods = sum(counts_dict.values())
    if n_mods == 0:
        return 0.0
    total_mass = sum(MODULE_EXACT_MASS.get(m, 0.0) * cnt for m, cnt in counts_dict.items())
    return total_mass - (n_mods - 1) * H2O_EXACT_MASS

def apply_mw_filter(module_counts, rf_model, module_names, proba_list, target_mw, tol=1.0):
    """Narrow down module combinations to those matching target_mw (+/- tol),
    ranked by Random Forest joint probability."""
    if not target_mw or target_mw <= 0:
        return module_counts, None, []

    # Build probability lookup: prob_map[module][count] = probability
    prob_map = {}
    for i, m in enumerate(module_names):
        classes = rf_model.estimators_[i].classes_
        probs = proba_list[i][0]
        prob_map[m] = {int(c): float(p) for c, p in zip(classes, probs)}

    # Search all plausible combinations (0..3 units per module) matching target_mw
    matching_combos = []
    for combo in product(range(4), repeat=len(module_names)):
        if sum(combo) == 0:
            continue
        c_dict = dict(zip(module_names, combo))
        theo_mw = calc_theoretical_mw(c_dict)
        diff = abs(theo_mw - target_mw)

        if diff <= tol:
            # Score by average RF probability across modules (small floor 0.01 for unseen counts)
            avg_prob = float(np.mean([prob_map[m].get(c_dict[m], 0.01) for m in module_names]) * 100.0)
            matching_combos.append({
                'counts': c_dict,
                'theo_mw': round(theo_mw, 4),
                'diff': round(diff, 4),
                'rf_score': round(avg_prob, 1)
            })

    if not matching_combos:
        return module_counts, f"No module combination matched MW = {target_mw:.2f} ± {tol:.2f} Da.", []

    # Rank matching combinations by RF probability (highest first), then smallest mass error
    matching_combos.sort(key=lambda x: (-x['rf_score'], x['diff']))
    best = matching_combos[0]
    note = (
        f"MW Filter ({target_mw:.2f} ± {tol:.2f} Da) narrowed to {len(matching_combos)} valid "
        f"combination(s). Selected top match: Exact Mass = {best['theo_mw']:.4f} Da (Δ = {best['diff']:.4f} Da)."
    )
    return best['counts'], note, matching_combos

# ==========================================
# 3. Streamlit Web Interface
# ==========================================
st.title("🧪 Automated ¹H NMR Structural Predictor")
st.markdown("Upload your experimental ¹H NMR peak list (`.xlsx`) to predict molecular modules, linkage sites, and 2D structures.")

# Sidebar: Template Download & Instructions
with st.sidebar:
    st.header("⚙️ Prediction Settings")
    conf_threshold = st.slider(
        "Module Selection Threshold (%)",
        min_value=50,
        max_value=95,
        value=70,
        step=5,
        help="Modules with presence probability above this threshold are assembled into the 2D structure. Modules below this threshold remain visible as candidates."
    )
    target_mw_input = st.number_input(
        "Target Molecular Weight — Neutral (Da)",
        min_value=0.0,
        max_value=2000.0,
        value=0.0,
        step=1.0,
        format="%.2f",
        help="Leave as 0.00 to disable MW filtering. Enter neutral exact mass / MW (e.g. 434.12) to narrow down valid module combinations."
    )
    mw_tolerance = st.number_input(
        "MW Tolerance (± Da)",
        min_value=0.01,
        max_value=5.0,
        value=1.0,
        step=0.1,
        format="%.2f",
        help="Use 0.05–0.1 Da for HRMS, or 1.0 Da for low-resolution MS."
    )
    st.divider()
    st.header("📥 Input Template")
    template_df = pd.DataFrame(columns=[
        'Compound_ID', 'Shift_ppm', 'J1_Hz', 'J2_Hz', 'J3_Hz', 'J4_Hz',
        'Integral', 'Multiplet_Width_Hz', 'Solvent_Overlap', 'Mixture_Overlap'
    ])
    buf = io.BytesIO()
    template_df.to_excel(buf, index=False)
    st.download_button(
        label="Download Excel Template",
        data=buf.getvalue(),
        file_name="unknown_compound_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.markdown("""
    **Quick Input Rules:**
    - **Pure compound:** Leave `Compound_ID` blank.
    - **Mixture:** Label `Unknown_1`, `Unknown_2`, etc., and set `Mixture_Overlap = 1` for overlapped peaks.
    - **Singlet (`s`):** Leave `J1_Hz` blank or `0`.
    - **Multiplet (`m`):** Type `m` in `J1_Hz`.
    - **Width:** Leave blank to auto-calculate from `J`.
    """)

uploaded_file = st.file_uploader("Upload `unknown_compound.xlsx`", type=["xlsx"])

if uploaded_file is not None:
    df_new = pd.read_excel(uploaded_file)

    # Clean & preprocess
    if 'Compound_ID' not in df_new.columns:
        df_new['Compound_ID'] = 'Unknown_Sample'
    else:
        df_new['Compound_ID'] = df_new['Compound_ID'].fillna('Unknown_Sample')

    df_new['Is_Multiplet'] = (df_new['J1_Hz'].astype(str).str.strip().str.lower() == 'm') if 'J1_Hz' in df_new.columns else False
    for col in j_cols:
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0.0) if col in df_new.columns else 0.0

    df_new['Integral'] = pd.to_numeric(df_new['Integral'], errors='coerce').fillna(0.0)
    j_sum = df_new[j_cols].sum(axis=1)
    if 'Multiplet_Width_Hz' not in df_new.columns:
        df_new['Multiplet_Width_Hz'] = j_sum
    else:
        df_new['Multiplet_Width_Hz'] = pd.to_numeric(df_new['Multiplet_Width_Hz'], errors='coerce').fillna(0.0)
        df_new['Multiplet_Width_Hz'] = np.where(df_new['Multiplet_Width_Hz'] == 0, j_sum, df_new['Multiplet_Width_Hz'])

    df_new['Solvent_Overlap'] = df_new['Solvent_Overlap'].fillna(0) if 'Solvent_Overlap' in df_new.columns else 0
    df_new['Mixture_Overlap'] = df_new['Mixture_Overlap'].fillna(0) if 'Mixture_Overlap' in df_new.columns else 0

    compounds_dict = deconvolve_excel_mixture(df_new, tol=0.15)
    if len(compounds_dict) > 1:
        st.info(f"Mixture detected: separated into {len(compounds_dict)} components ({', '.join(compounds_dict.keys())}).")

    for cid, compound_df in compounds_dict.items():
        st.divider()
        st.subheader(f"🔬 Prediction Report: `{cid}`")

        # Stage 1
        vector, reg_feats, domain_info = create_nmr_vector(compound_df)
        vec_2d = vector.reshape(1, -1)
        proba_list = rf_model.predict_proba(vec_2d)

        module_counts, module_confs, stage1_summary = select_modules_by_threshold(
            rf_model, module_names, proba_list, threshold=conf_threshold
        )
        module_counts, adjustments = apply_stoichiometric_filter(
            module_counts,
            module_confs,
            reg_feats,
            bypass_sugar_filter=domain_info.get('bypass_sugar_filter', False)
        )

        # Spectroscopic Motif Safeguard (Rules 1, 2, 3)
        expected_arom = sum(AROMATIC_H_MAP.get(m, 0.0) * cnt for m, cnt in module_counts.items())
        unaccounted_arom = reg_feats[0] - expected_arom
        has_cinnamoyl = (module_counts.get('caffeic', 0) + module_counts.get('coumaric', 0)) > 0

        if domain_info['trans_peaks'] >= 1 and not has_cinnamoyl and unaccounted_arom >= 4.0:
            if domain_info['aabb_2h_doublets'] >= 2 and module_confs.get('coumaric', 0.0) >= 20.0:
                module_counts['coumaric'] = 1
                adjustments.append(
                    f"Auto-included coumaric (1x, {module_confs['coumaric']:.1f}%): trans-alkene J > 13 Hz ({domain_info['trans_peaks']} peaks) + AA'BB' 2H doublets ({domain_info['aabb_2h_doublets']} peaks, J = 8–9 Hz)"
                )
            elif domain_info['abx_score'] >= 2 and module_confs.get('caffeic', 0.0) >= 20.0:
                module_counts['caffeic'] = 1
                adjustments.append(
                    f"Auto-included caffeic (1x, {module_confs['caffeic']:.1f}%): trans-alkene J > 13 Hz ({domain_info['trans_peaks']} peaks) + ABX aromatic triad ({domain_info['abx_score']}/3 signals)"
                )
        # Anomeric Proton Safeguard: total sugar units cannot exceed true anomeric H (1 J-value, J < 11.5 Hz)
        SUGAR_KEYWORDS = ('glc', 'gal', 'man', 'ara', 'xyl', 'rha', 'fru', 'sugar')
        sugar_mods = [m for m in module_counts if any(k in m.lower() for k in SUGAR_KEYWORDS) and module_counts[m] > 0]
        total_sugar_units = sum(module_counts[m] for m in sugar_mods)
        true_anomeric_h = int(round(reg_feats[1]))

        if true_anomeric_h >= 1 and total_sugar_units > true_anomeric_h:
            excess = total_sugar_units - true_anomeric_h
            # Trim excess units starting from the sugar module with the lowest RF confidence
            for m in sorted(sugar_mods, key=lambda x: module_confs.get(x, 0.0)):
                if excess <= 0:
                    break
                reduce_by = min(module_counts[m], excess)
                old_cnt = module_counts[m]
                module_counts[m] -= reduce_by
                excess -= reduce_by
                adjustments.append(
                    f"Adjusted {m} ({old_cnt}x → {module_counts[m]}x): only {reg_feats[1]:.1f} H in 4.3–5.5 ppm matched anomeric rule (1 J-value, J < 11.5 Hz); remaining signals assigned to CH2."
                )
        # Molecular Weight Filter (reads optional 'MW' column in Excel, or falls back to Sidebar input)
        active_mw = target_mw_input
        if 'MW' in compound_df.columns and pd.notna(compound_df['MW'].iloc[0]):
            active_mw = float(compound_df['MW'].iloc[0])

        # 1. Resolve solvent-obscured peaks (4.3–5.5 ppm) first
            mw_matches = []
            resolved_by_obscured = False
            if domain_info.get("solvent_obscured_peaks"):
                module_counts, mw_hypotheses = resolve_solvent_obscured_hypotheses(
                    base_counts=module_counts,
                    module_confs=module_confs,
                    module_names=module_names,
                    domain_info=domain_info,
                    reg_feats=reg_feats,
                    calc_mw_fn=calc_theoretical_mw,
                    active_mw=active_mw,
                    mw_tol=mw_tolerance,
                )

                if active_mw > 0 and mw_hypotheses:
                    best_h = mw_hypotheses[0]
                    resolved_by_obscured = True
                    adjustments.append(
                        f"MW Filter ({active_mw:.1f} Da) resolved solvent-obscured peak → "
                        f"{best_h['label']} (Calc MW: {best_h['est_mw']:.1f} Da)"
                    )
                    st.success(
                        f"MW Filter ({active_mw:.1f} Da) resolved solvent-obscured peak → "
                        f"**{best_h['label']}** (Calc MW: {best_h['est_mw']:.1f} Da)"
                    )
                elif active_mw <= 0:
                    with st.expander("Solvent-obscured peak (4.3–5.5 ppm): open hypotheses (enter MW to filter)", expanded=True):
                        for h in mw_hypotheses:
                            st.write(f"- **{h['label']}** (Est. MW: `{h['est_mw']:.1f} Da`): {h['note']}")

            # 2. Fallback to standard MW filter if not already resolved above
            if active_mw > 0 and not resolved_by_obscured:
                module_counts, mw_note, mw_matches = apply_mw_filter(
                    module_counts, rf_model, module_names, proba_list, active_mw, tol=mw_tolerance
                )
                if mw_note:
                    adjustments.append(mw_note)

        # Sync final post-filter counts back to Stage 1 display table
        for row in stage1_summary:
            final_c = module_counts[row["Module"]]
            row["Assembled"] = f"{final_c}x" if final_c > 0 else "—"
            if final_c > 0 and "Candidate" in row["Status"]:
                row["Status"] = "✅ Selected (NMR Motif)"

        active_modules = [m for m, c in module_counts.items() if c > 0]
        linkages = auto_match_and_analyze(compound_df, active_modules, qm_database)
        overall_conf, conf_tier = compute_overall_confidence(module_counts, module_confs, reg_feats, linkages)
        display_linkages = [{k: v for k, v in lk.items() if not k.startswith('_')} for lk in linkages]

        c0, c1, c2, c3, c4 = st.columns(5)
        c0.metric("Prediction Confidence", f"{overall_conf:.1f}%", delta=f"{conf_tier} Reliability")
        c1.metric("Aromatic (6.2–8.5 ppm)", f"{reg_feats[0]:.1f} H")
        c2.metric("Anomeric (4.3–5.5 ppm)", f"{reg_feats[1]:.1f} H")
        c3.metric("Sugar Body (3.0–4.3 ppm)", f"{reg_feats[2]:.1f} H")
        c4.metric("Total Peaks", f"{int(reg_feats[3])}")

        st.progress(min(int(overall_conf), 100) / 100.0)
        if cv_accuracy is not None:
            st.caption(f"Model Cross-Validation Baseline Accuracy: **{cv_accuracy:.1f}%**")
        st.caption(
            f"Detected Spin Systems — "
            f"**Rule 1 (Trans $J > 13$ Hz):** {domain_info['trans_peaks']} peaks ({domain_info['trans_int']:.1f} H) | "
            f"**Rule 2 (AA'BB' 2H, $J = 8\\text{{–}}9$ Hz):** {domain_info['aabb_2h_doublets']} peaks ({domain_info['aabb_pairs']} pairs) | "
            f"**Rule 3 (ABX 1H Triad):** {domain_info['abx_score']}/3 signals "
            f"[ortho: {domain_info['abx_ortho_1h']}, meta: {domain_info['abx_meta_1h']}, dd: {domain_info['abx_dd_1h']}]"
        )
        col_left, col_right = st.columns([1, 1.3])
        with col_left:
            st.markdown(f"**Stage 1 — Module Probabilities (Threshold > {conf_threshold}%)**")
            st.dataframe(pd.DataFrame(stage1_summary), use_container_width=True, hide_index=True)
            if not any(c > 0 for c in module_counts.values()):
                st.warning(f"No module exceeded the {conf_threshold}% threshold. Review the Candidate modules above or lower the threshold in the sidebar.")
            if adjustments:
                st.info("**Stoichiometric Filter Applied:**\n" + "\n".join(f"- {a}" for a in adjustments))

        with col_right:
            st.markdown("**Stage 3 — All Possible Structural Assemblies**")
            candidates = enumerate_all_assemblies(module_counts, linkages, overall_conf)
            if candidates:
                summary_rows = [
                    {
                        "Rank": f"#{idx + 1}",
                        "Confidence": f"{cand['confidence']:.1f}%",
                        "Linkage Architecture": cand['pathway'],
                        "NMR Shift Evidence": cand['evidence']
                    }
                    for idx, cand in enumerate(candidates)
                ]
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            else:
                st.warning("Could not assemble a valid 2D structure from the selected modules.")

        # Render 2D Structures Grid for All Assembled Candidates
        if candidates:
            st.markdown(f"**Assembled 2D Isomer Candidates ({len(candidates)} unique structure{'s' if len(candidates) > 1 else ''})**")
            cols_per_row = 3
            for start_idx in range(0, len(candidates), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for col_offset, cand in enumerate(candidates[start_idx:start_idx + cols_per_row]):
                    rank_num = start_idx + col_offset + 1
                    with row_cols[col_offset]:
                        badge = "🏆 " if rank_num == 1 else ""
                        st.markdown(f"**{badge}Candidate #{rank_num} — Confidence: `{cand['confidence']:.1f}%`**")
                        formula = rdMolDescriptors.CalcMolFormula(cand['mol'])
                        exact_mw = Descriptors.ExactMolWt(cand['mol'])
                        st.caption(f"Formula: `{formula}` | Exact Mass: `{exact_mw:.4f} Da`  \nLinkage: {cand['pathway']}  \n*{cand['evidence']}*")
                        img = Draw.MolToImage(cand['mol'], size=(600, 320))
                        st.image(img, use_container_width=True)
                        st.code(cand['smiles'], language="text")
                        img_buf = io.BytesIO()
                        img.save(img_buf, format="PNG")
                        st.download_button(
                            label=f"Download #{rank_num} (.png)",
                            data=img_buf.getvalue(),
                            file_name=f"predicted_{cid}_candidate_{rank_num}.png",
                            mime="image/png",
                            key=f"dl_{cid}_{rank_num}"
                        )

       # Stage 2
        st.markdown("**Stage 2 — Potential Linkage Sites ($0.3 \\le |\\Delta\\delta| \\le 1.5$ ppm)**")
        if display_linkages:
            st.dataframe(pd.DataFrame(display_linkages), use_container_width=True)
        else:
            st.caption("No significant chemical shift deviations detected.")