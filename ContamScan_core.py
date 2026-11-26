# -*- coding: utf-8 -*-

import math
import os
import re
import copy
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from matchms import Spectrum
from matchms.filtering import add_precursor_mz
from matchms.similarity import ModifiedCosine

warnings.filterwarnings("ignore")


CLASS_MAPPING = {
    1: "Alcoholamines",
    2: "Parbens",
    3: "Phthalates",
    4: "Organophosphates",
    5: "Citric acid esters",
    6: "Perfluorinated compounds",
}


def _log(log_callback, message):
    if log_callback:
        log_callback(message)


def _progress(progress_callback, value):
    if progress_callback:
        progress_callback(int(max(0, min(100, value))))


def map_class_id_to_name(class_id):
    if isinstance(class_id, str):
        if class_id.isdigit():
            class_id = int(class_id)
        else:
            return class_id

    if isinstance(class_id, (int, float)) and not pd.isna(class_id):
        class_id_int = int(class_id)
        return CLASS_MAPPING.get(class_id_int, "Unknown")

    return "Unknown"


# =========================
# Contam-screen core
# =========================

def get_list_max_num_index(num_list, topk=30):
    tmp_list = copy.deepcopy(list(num_list))
    tmp_list.sort()
    return [list(num_list).index(one) for one in tmp_list[::-1][:topk]]


def remove_neu(spectral_neu, spectral_neu_inten, ms2_error, start_neu, end_neu):
    spectral_neu = list(spectral_neu)
    spectral_neu_inten = list(spectral_neu_inten)

    for l in range(len(spectral_neu)):
        for m in range(l, len(spectral_neu)):
            if l != m and abs(spectral_neu[l] - spectral_neu[m]) < ms2_error:
                if spectral_neu_inten[l] > spectral_neu_inten[m]:
                    spectral_neu_inten[m] = 0
                else:
                    spectral_neu_inten[l] = 0

    max_num_index = get_list_max_num_index(spectral_neu_inten, topk=30)
    spectral_neu = np.array(spectral_neu)[max_num_index]
    spectral_neu_inten = np.array(spectral_neu_inten)[max_num_index]

    keep = np.where(spectral_neu > start_neu)
    spectral_neu = spectral_neu[keep]
    spectral_neu_inten = spectral_neu_inten[keep]

    keep = np.where(spectral_neu < end_neu)
    spectral_neu = spectral_neu[keep]
    spectral_neu_inten = spectral_neu_inten[keep]

    return spectral_neu, spectral_neu_inten


def screening_vector_binning(
    spectral,
    spectral_inten,
    end_mass,
    start_mass,
    start_neu,
    end_neu,
    ms2_error,
    precursor_mz,
):
    spectral = np.array(spectral, dtype=float)
    spectral_inten = np.array(spectral_inten, dtype=float)

    if len(spectral) == 0 or len(spectral_inten) == 0:
        return [], [], [], [], [], []

    max_intensity = np.max(spectral_inten)
    if max_intensity <= 0:
        return [], [], [], [], [], []

    spectral_inten_norm = spectral_inten / max_intensity
    spectral_inten_sqrt = np.sqrt(spectral_inten_norm)

    keep = np.where(spectral_inten_norm > 0.01)
    spectral = spectral[keep]
    spectral_inten_norm = spectral_inten_norm[keep]
    spectral_inten_sqrt = spectral_inten_sqrt[keep]

    keep = np.where(spectral - precursor_mz < 0.01)
    spectral = spectral[keep]
    spectral_inten_norm = spectral_inten_norm[keep]
    spectral_inten_sqrt = spectral_inten_sqrt[keep]

    keep = np.where(spectral < end_mass)
    spectral = spectral[keep]
    spectral_inten_norm = spectral_inten_norm[keep]
    spectral_inten_sqrt = spectral_inten_sqrt[keep]

    keep = np.where(spectral > start_mass)
    spectral = spectral[keep]
    spectral_inten_norm = spectral_inten_norm[keep]
    spectral_inten_sqrt = spectral_inten_sqrt[keep]

    spectral_neu = []
    spectral_neu_inten_sqrt = []

    for n in range(len(spectral)):
        spectral_neu.append(abs(precursor_mz - spectral[n]))
        spectral_neu_inten_sqrt.append(spectral_inten_sqrt[n])

    spectral_neu, spectral_neu_inten_sqrt = remove_neu(
        spectral_neu,
        spectral_neu_inten_sqrt,
        ms2_error,
        start_neu,
        end_neu,
    )

    spectral_neu_inten_norm = []
    for neu_mz in spectral_neu:
        idx = np.where(
            np.abs(np.array([abs(precursor_mz - mz) for mz in spectral]) - neu_mz) < 0.001
        )[0]
        if len(idx) > 0:
            spectral_neu_inten_norm.append(spectral_inten_norm[idx[0]])
        else:
            spectral_neu_inten_norm.append(0)

    return (
        spectral,
        spectral_inten_norm,
        spectral_inten_sqrt,
        spectral_neu,
        spectral_neu_inten_norm,
        spectral_neu_inten_sqrt,
    )


def parse_msms_spectrum(msms_spectrum_str):
    spectral = []
    spectral_inten = []

    if pd.isna(msms_spectrum_str):
        return spectral, spectral_inten

    text = str(msms_spectrum_str).strip()
    if not text:
        return spectral, spectral_inten

    # Original format: "mz:intensity mz:intensity"
    tokens = re.split(r"[\s;]+", text)

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        if ":" in token:
            parts = token.split(":", 1)
        else:
            continue

        try:
            mz = float(parts[0])
            intensity = float(parts[1])
            spectral.append(mz)
            spectral_inten.append(intensity)
        except Exception:
            continue

    return spectral, spectral_inten


def filter_by_adduct_type(df, ion_mode, log_callback=None):
    if "Adduct type" not in df.columns:
        _log(log_callback, "Warning: column 'Adduct type' not found. Skip adduct filtering.")
        return df

    original_count = len(df)

    if ion_mode == "P":
        allowed_adducts = ["[M+H]+", "[M+NH4]+"]
    else:
        allowed_adducts = ["[M-H]-"]

    df_filtered = df[df["Adduct type"].astype(str).str.strip().isin(allowed_adducts)]
    removed_count = original_count - len(df_filtered)

    _log(
        log_callback,
        f"Adduct filter: kept {len(df_filtered)} of {original_count}, removed {removed_count}.",
    )

    return df_filtered


def filter_screen_result_by_ion_mode(df, ion_mode, log_callback=None):
    if "predicted_classID" not in df.columns:
        return df

    if ion_mode == "P":
        valid_classes = [1, 3, 4, 5]
    else:
        valid_classes = [2, 6]

    original_count = len(df)
    df_filtered = df[df["predicted_classID"].isin(valid_classes)]
    _log(
        log_callback,
        f"Ion mode class filter: kept {len(df_filtered)} of {original_count}, removed {original_count - len(df_filtered)}.",
    )
    return df_filtered


def remove_empty_msms(df, log_callback=None):
    if "MS/MS spectrum" not in df.columns:
        _log(log_callback, "Warning: column 'MS/MS spectrum' not found. Skip empty MS/MS filtering.")
        return df

    original_count = len(df)

    df_filtered = df.dropna(subset=["MS/MS spectrum"])
    df_filtered = df_filtered[df_filtered["MS/MS spectrum"].astype(str).str.strip() != ""]
    df_filtered = df_filtered[df_filtered["MS/MS spectrum"].astype(str).str.strip() != "nan"]
    df_filtered = df_filtered[df_filtered["MS/MS spectrum"].astype(str).str.strip() != "None"]

    _log(
        log_callback,
        f"MS/MS filter: kept {len(df_filtered)} of {original_count}, removed {original_count - len(df_filtered)}.",
    )

    return df_filtered


def run_contam_screen(
    peak_list_path,
    model_path,
    output_path,
    ion_mode="P",
    progress_callback=None,
    log_callback=None,
):
    start_mass = 50
    end_mass = 800
    bin_width = 0.01
    start_neu = 12
    end_neu = 200
    ms2_error = 0.01

    required_columns = [
        "Spectrum reference file name",
        "Average Mz",
        "Alignment ID",
        "Average Rt(min)",
        "MS/MS spectrum",
    ]

    peak_list_path = str(peak_list_path)
    model_path = str(model_path)
    output_path = str(output_path)

    if not os.path.exists(peak_list_path):
        raise FileNotFoundError(f"Peak list not found: {peak_list_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _progress(progress_callback, 1)
    _log(log_callback, "Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)

    _progress(progress_callback, 5)
    _log(log_callback, "Reading peak list...")
    excel_data = pd.read_excel(peak_list_path)

    missing_columns = [c for c in required_columns if c not in excel_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in peak list: {missing_columns}")

    _log(log_callback, f"Original rows: {len(excel_data)}")

    excel_data = filter_by_adduct_type(excel_data, ion_mode, log_callback)
    excel_data = remove_empty_msms(excel_data, log_callback)
    excel_data = excel_data.reset_index(drop=True)

    if len(excel_data) == 0:
        raise ValueError("No valid data after filtering.")

    shape_num = len(excel_data)
    bin_num = round((end_mass - start_mass) / bin_width)
    bin_neu_num = round((end_neu - start_neu) / bin_width)

    result_rows = []

    _log(log_callback, "Start prediction...")

    for i in range(shape_num):
        row = excel_data.loc[i]

        file_ori = row["Spectrum reference file name"]
        precursor_mz = round(float(row["Average Mz"]), 5)
        item_id = row["Alignment ID"]
        rt = round(float(row["Average Rt(min)"]), 2)
        msms_spectrum_str = str(row["MS/MS spectrum"])

        spectral, spectral_inten = parse_msms_spectrum(msms_spectrum_str)

        result = {
            "ID": int(item_id),
            "FILES": file_ori,
            "PEPMASS": precursor_mz,
            "RT": rt,
            "MSMS": "",
            "pred_probability": np.nan,
            "predicted_classID": np.nan,
            "predicted_class": "Unknown",
        }

        if len(spectral) > 0:
            (
                spectral,
                spectral_inten_norm,
                spectral_inten_sqrt,
                spectral_neu,
                spectral_neu_inten_norm,
                spectral_neu_inten_sqrt,
            ) = screening_vector_binning(
                spectral,
                spectral_inten,
                end_mass,
                start_mass,
                start_neu,
                end_neu,
                ms2_error,
                precursor_mz,
            )

            ms2_parts = []
            for m in range(len(spectral)):
                ms2_parts.append(f"{round(float(spectral[m]), 5)} {round(float(spectral_inten_norm[m]), 3)}")
            result["MSMS"] = ";".join(ms2_parts)

            if len(spectral) >= 3:
                spectral_box = [0] * bin_num
                neu_box = [0] * bin_neu_num

                for l in range(len(spectral)):
                    bin_position = math.floor((float(spectral[l]) - start_mass) / bin_width)
                    if 0 <= bin_position < bin_num:
                        spectral_box[bin_position] = max(
                            round(float(spectral_inten_sqrt[l]), 3),
                            spectral_box[bin_position],
                        )

                for l in range(len(spectral_neu)):
                    bin_neu_position = math.floor((float(spectral_neu[l]) - start_neu) / bin_width)
                    if 0 <= bin_neu_position < bin_neu_num:
                        neu_box[bin_neu_position] = max(
                            round(float(spectral_neu_inten_sqrt[l]), 3),
                            neu_box[bin_neu_position],
                        )

                screen_input = np.array(spectral_box + neu_box).reshape(1, int(bin_num + bin_neu_num))
                pred_result = model.predict(screen_input, verbose=0)

                pred_probability = float(max(pred_result[0]))
                pred_class_idx = int(np.argmax(pred_result, axis=-1)[0])
                class_id = pred_class_idx + 1

                result["pred_probability"] = pred_probability
                result["predicted_classID"] = class_id
                result["predicted_class"] = map_class_id_to_name(class_id)

        result_rows.append(result)

        if i % 10 == 0 or i == shape_num - 1:
            _progress(progress_callback, 5 + (i + 1) / shape_num * 85)

    result_df = pd.DataFrame(result_rows)
    result_df = filter_screen_result_by_ion_mode(result_df, ion_mode, log_callback)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_excel(output_path, index=False, engine='openpyxl')

    _progress(progress_callback, 100)
    _log(log_callback, f"Saved: {output_path}")
    _log(log_callback, f"Final rows: {len(result_df)}")

    return str(output_path)


# =========================
# Contam-confirm core
# =========================

def get_compound_name(row):
    possible_columns = ["Name", "name", "NAME", "Compound_Name", "Compound name", "Library Name"]

    for col in possible_columns:
        if col in row.index:
            value = row[col]
            if pd.notna(value) and str(value).strip() != "" and str(value) != "nan":
                return str(value).strip()

    return "Unknown"


def parse_spectrum_string(spectrum_str):
    if pd.isna(spectrum_str) or str(spectrum_str).strip() == "":
        return np.array([]), np.array([])

    peaks = str(spectrum_str).split(";")
    mz_list = []
    intensity_list = []

    for peak in peaks:
        if peak.strip():
            parts = re.split(r"[\s,]+", peak.strip())
            if len(parts) >= 2:
                try:
                    mz_list.append(float(parts[0]))
                    intensity_list.append(float(parts[1]))
                except ValueError:
                    continue

    return np.array(mz_list), np.array(intensity_list)


def create_matchms_spectrum(mz_array, intensity_array, precursor_mz, spectrum_id=""):
    if len(mz_array) == 0 or len(intensity_array) == 0:
        return None

    max_intensity = np.max(intensity_array)
    if max_intensity > 0:
        intensity_array = intensity_array / max_intensity

    spectrum = Spectrum(
        mz=mz_array,
        intensities=intensity_array,
        metadata={
            "spectrum_id": spectrum_id,
            "precursor_mz": float(precursor_mz) if precursor_mz else 0.0,
        },
    )

    spectrum = add_precursor_mz(spectrum)
    return spectrum


def calculate_similarity(msms_str1, msms_str2, precursor1, precursor2):
    if (
        pd.isna(msms_str1)
        or pd.isna(msms_str2)
        or str(msms_str1).strip() == ""
        or str(msms_str2).strip() == ""
        or pd.isna(precursor1)
        or pd.isna(precursor2)
    ):
        return 0.0, 0

    try:
        precursor1 = float(precursor1)
        precursor2 = float(precursor2)
    except Exception:
        return 0.0, 0

    mz1, intensity1 = parse_spectrum_string(msms_str1)
    mz2, intensity2 = parse_spectrum_string(msms_str2)

    if len(mz1) == 0 or len(mz2) == 0:
        return 0.0, 0

    spec1 = create_matchms_spectrum(mz1, intensity1, precursor1, "spectrum1")
    spec2 = create_matchms_spectrum(mz2, intensity2, precursor2, "spectrum2")

    if spec1 is None or spec2 is None:
        return 0.0, 0

    try:
        modified_cosine = ModifiedCosine(tolerance=0.01, mz_power=0.0, intensity_power=1.0)
        result = modified_cosine.pair(spec1, spec2)

        if result is None:
            return 0.0, 0

        if hasattr(result, "score"):
            score = float(result.score)
            matches = int(getattr(result, "matches", 0))
        elif isinstance(result, np.ndarray):
            if result.dtype.names is not None:
                score = float(result["score"]) if "score" in result.dtype.names else float(result[0])
                matches = int(result["matches"]) if "matches" in result.dtype.names else 0
            else:
                score = float(result[0]) if len(result) >= 1 else 0.0
                matches = int(result[1]) if len(result) >= 2 else 0
        elif isinstance(result, tuple):
            score = float(result[0]) if len(result) >= 1 else 0.0
            matches = int(result[1]) if len(result) >= 2 else 0
        else:
            score = float(result)
            matches = 0

        if np.isnan(score):
            return 0.0, 0

        return score, matches

    except Exception:
        return 0.0, 0


def calculate_mass_difference(pepmass1, pepmass2):
    try:
        return float(pepmass2) - float(pepmass1)
    except Exception:
        return None


def find_formula_difference(mass_diff, formula_db, delta_threshold=0.01):
    if mass_diff is None:
        return None

    if "Mass" not in formula_db.columns:
        raise ValueError("PMD database must contain column: Mass")

    mass_diff_abs = abs(float(mass_diff))

    for _, row in formula_db.iterrows():
        try:
            db_mass = float(row["Mass"])
            delta = abs(mass_diff_abs - db_mass)
            if delta <= delta_threshold:
                return row.get("Formula_Difference", None)
        except Exception:
            continue

    return None


def check_edge_priority(edge):
    source1 = edge.get("Source_1", "")
    source2 = edge.get("Source_2", "")

    if source1 == "seed" and source2 == "seed":
        return False

    name1 = str(edge.get("NAME_1", "")).strip()
    name2 = str(edge.get("NAME_2", "")).strip()

    name1_not_empty = name1 not in ["", "Unknown", "nan", "None"]
    name2_not_empty = name2 not in ["", "Unknown", "nan", "None"]

    return (name1_not_empty and not name2_not_empty) or (not name1_not_empty and name2_not_empty)


def remove_self_connections(edges_list):
    return [edge for edge in edges_list if edge.get("ID_1") != edge.get("ID_2")]


def filter_edges_with_priority(edges_list, top_k=10):
    if len(edges_list) <= top_k:
        return edges_list

    node_to_priority_edges = defaultdict(list)
    node_to_normal_edges = defaultdict(list)

    for edge in edges_list:
        node1 = edge["Index_1"]
        node2 = edge["Index_2"]
        score = edge.get("Similarity_score", 0)
        edge_info = (score, edge)

        if check_edge_priority(edge):
            node_to_priority_edges[node1].append(edge_info)
            node_to_priority_edges[node2].append(edge_info)
        else:
            node_to_normal_edges[node1].append(edge_info)
            node_to_normal_edges[node2].append(edge_info)

    selected_edge_ids = set()
    all_nodes = set(list(node_to_priority_edges.keys()) + list(node_to_normal_edges.keys()))

    for node in all_nodes:
        priority_edges = node_to_priority_edges.get(node, [])
        normal_edges = node_to_normal_edges.get(node, [])

        priority_edges.sort(key=lambda x: x[0], reverse=True)
        normal_edges.sort(key=lambda x: x[0], reverse=True)

        selected_count = 0

        for _, edge in priority_edges:
            if selected_count >= top_k:
                break
            selected_edge_ids.add((edge["Index_1"], edge["Index_2"]))
            selected_count += 1

        for _, edge in normal_edges:
            if selected_count >= top_k:
                break
            edge_id = (edge["Index_1"], edge["Index_2"])
            if edge_id not in selected_edge_ids:
                selected_edge_ids.add(edge_id)
                selected_count += 1

    return [edge for edge in edges_list if (edge["Index_1"], edge["Index_2"]) in selected_edge_ids]


def optimize_formula_network(edges_list, formula_db, threshold=3, pmd_tolerance=0.01):
    if len(edges_list) == 0:
        return edges_list

    edges = [edge.copy() for edge in edges_list]

    for edge in edges:
        edge["is_priority"] = check_edge_priority(edge)
        mass_diff = edge.get("Mass")
        if mass_diff is not None:
            formula_diff = find_formula_difference(mass_diff, formula_db, pmd_tolerance)
            edge["Standard_Mass"] = abs(float(mass_diff)) if formula_diff is not None else None
        else:
            edge["Standard_Mass"] = None

    node_mass_edges = defaultdict(lambda: defaultdict(list))
    edge_to_info = {}

    for i, edge in enumerate(edges):
        node1, node2 = edge["Index_1"], edge["Index_2"]
        std_mass = edge["Standard_Mass"]

        if std_mass is None:
            continue

        edge_info = {
            "edge": edge,
            "edge_id": i,
            "is_priority": edge["is_priority"],
        }

        node_mass_edges[node1][std_mass].append(edge_info)
        node_mass_edges[node2][std_mass].append(edge_info)
        edge_to_info[i] = edge_info

    node_selected_edges = defaultdict(set)

    for node in node_mass_edges.keys():
        selected_count = 0
        for std_mass in sorted(node_mass_edges[node].keys()):
            if selected_count >= threshold:
                break

            mass_edges = node_mass_edges[node][std_mass]
            mass_edges.sort(
                key=lambda x: (
                    -int(x["is_priority"]),
                    -float(x["edge"].get("Similarity_score", 0)),
                )
            )

            for edge_info in mass_edges:
                if selected_count >= threshold:
                    break

                if edge_info["edge_id"] not in node_selected_edges[node]:
                    node_selected_edges[node].add(edge_info["edge_id"])
                    selected_count += 1

    kept_edges = set()
    for edge_ids in node_selected_edges.values():
        kept_edges.update(edge_ids)

    final_edges = {}
    for edge_id in kept_edges:
        edge = edge_to_info[edge_id]["edge"].copy()
        edge.pop("Standard_Mass", None)
        edge.pop("is_priority", None)

        id1 = edge["ID_1"]
        id2 = edge["ID_2"]
        edge_type = edge.get("Edge_type", "Mass_difference")

        key = (str(min(str(id1), str(id2))), str(max(str(id1), str(id2))), edge_type)

        if key not in final_edges or edge.get("Similarity_score", 0) > final_edges[key].get("Similarity_score", 0):
            final_edges[key] = edge

    return list(final_edges.values())


def safe_sheet_name(name):
    name = str(name)
    name = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)
    return name[:31] if len(name) > 31 else name


def run_contam_confirm(
    node_list_path,
    seed_list_path,
    formula_db_path,
    output_path,
    similarity_threshold=0.7,
    similarity_top_n=10,
    pmd_tolerance=0.01,
    pmd_top_n=3,
    progress_callback=None,
    log_callback=None,
):
    node_list_path = str(node_list_path)
    seed_list_path = str(seed_list_path)
    formula_db_path = str(formula_db_path)
    output_path = str(output_path)

    for path in [node_list_path, seed_list_path, formula_db_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    _progress(progress_callback, 1)
    _log(log_callback, "Reading node list, seed list and PMD database...")

    df1 = pd.read_excel(node_list_path)
    df2 = pd.read_excel(seed_list_path)
    formula_db = pd.read_excel(formula_db_path)

    required_node_columns = ["PEPMASS", "MSMS"]
    required_seed_columns = ["PEPMASS", "MSMS"]

    missing_node = [c for c in required_node_columns if c not in df1.columns]
    missing_seed = [c for c in required_seed_columns if c not in df2.columns]

    if missing_node:
        raise ValueError(f"Missing required columns in node list: {missing_node}")
    if missing_seed:
        raise ValueError(f"Missing required columns in seed list: {missing_seed}")
    if "Mass" not in formula_db.columns:
        raise ValueError("PMD database must contain column: Mass")

    df1["Source"] = "sample"
    df2["Source"] = "seed"

    if "predicted_classID" in df1.columns:
        df1["Group_Class"] = df1["predicted_classID"].apply(map_class_id_to_name)
    elif "predicted_class" in df1.columns:
        df1["Group_Class"] = df1["predicted_class"].apply(map_class_id_to_name)
    else:
        df1["Group_Class"] = "Unknown_sample"

    if "Class" in df2.columns:
        df2["Group_Class"] = df2["Class"].apply(map_class_id_to_name)
    else:
        df2["Group_Class"] = "Unknown_seed"

    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df = combined_df.dropna(subset=["MSMS"])
    combined_df = combined_df[combined_df["MSMS"].astype(str).str.strip() != ""]

    class_groups = combined_df.groupby("Group_Class").groups

    total_combinations = 0
    for _, indices in class_groups.items():
        if len(indices) >= 2:
            total_combinations += len(list(combinations(indices, 2)))

    if total_combinations == 0:
        raise ValueError("No valid pair combinations found.")

    similarity_results_by_class = defaultdict(list)
    formula_results_by_class = defaultdict(list)

    done = 0
    _progress(progress_callback, 5)
    _log(log_callback, f"Total valid rows: {len(combined_df)}")
    _log(log_callback, f"Total combinations: {total_combinations}")

    for class_name, indices in class_groups.items():
        if len(indices) < 2:
            continue

        class_combinations = list(combinations(indices, 2))
        _log(log_callback, f"Processing class: {class_name}, pairs: {len(class_combinations)}")

        for idx1, idx2 in class_combinations:
            row1 = combined_df.loc[idx1]
            row2 = combined_df.loc[idx2]

            similarity_score, matched_peaks = calculate_similarity(
                row1["MSMS"],
                row2["MSMS"],
                row1["PEPMASS"],
                row2["PEPMASS"],
            )

            mass_diff = calculate_mass_difference(row1["PEPMASS"], row2["PEPMASS"])
            formula_diff = find_formula_difference(mass_diff, formula_db, pmd_tolerance)

            result_row = {
                "Index_1": idx1,
                "Source_1": row1["Source"],
                "NAME_1": get_compound_name(row1),
                "ID_1": row1.get("ID", "Unknown"),
                "PEPMASS_1": row1["PEPMASS"],
                "MSMS_1": row1["MSMS"],
                "Class_1": class_name,
                "Index_2": idx2,
                "Source_2": row2["Source"],
                "NAME_2": get_compound_name(row2),
                "ID_2": row2.get("ID", "Unknown"),
                "PEPMASS_2": row2["PEPMASS"],
                "MSMS_2": row2["MSMS"],
                "Class_2": class_name,
                "Similarity_score": similarity_score,
                "Matched_Peaks": matched_peaks,
                "Mass": mass_diff,
                "Formula_Difference": formula_diff,
            }

            if similarity_score >= similarity_threshold:
                similarity_results_by_class[class_name].append(result_row.copy())

            if formula_diff is not None:
                formula_results_by_class[class_name].append(result_row.copy())

            done += 1
            if done % 100 == 0 or done == total_combinations:
                _progress(progress_callback, 5 + done / total_combinations * 85)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_classes = set(similarity_results_by_class.keys()) | set(formula_results_by_class.keys())

    _log(log_callback, "Merging and saving results...")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for class_name in sorted(all_classes):
            merged_edges = []

            for edge in similarity_results_by_class.get(class_name, []):
                edge_copy = edge.copy()
                edge_copy["Edge_type"] = "Similarity"
                merged_edges.append(edge_copy)

            for edge in formula_results_by_class.get(class_name, []):
                edge_copy = edge.copy()
                edge_copy["Edge_type"] = "Mass_difference"
                merged_edges.append(edge_copy)

            merged_edges = remove_self_connections(merged_edges)

            deduplicated = {}
            for edge in merged_edges:
                id1 = edge["ID_1"]
                id2 = edge["ID_2"]
                edge_type = edge["Edge_type"]

                if str(id1) < str(id2):
                    key = (str(id1), str(id2), edge_type)
                else:
                    key = (str(id2), str(id1), edge_type)

                score = edge.get("Similarity_score", 0)

                if key not in deduplicated or score > deduplicated[key].get("Similarity_score", 0):
                    deduplicated[key] = edge

            deduplicated_list = list(deduplicated.values())

            sim_edges = [e for e in deduplicated_list if e["Edge_type"] == "Similarity"]
            pmd_edges = [e for e in deduplicated_list if e["Edge_type"] == "Mass_difference"]

            sim_edges = filter_edges_with_priority(sim_edges, similarity_top_n)
            pmd_edges = optimize_formula_network(
                pmd_edges,
                formula_db,
                threshold=pmd_top_n,
                pmd_tolerance=pmd_tolerance,
            )

            final_edges = sim_edges + pmd_edges

            if not final_edges:
                continue

            df_out = pd.DataFrame(final_edges)
            cols = ["Edge_type"] + [c for c in df_out.columns if c != "Edge_type"]
            df_out = df_out[cols]

            sheet = safe_sheet_name(class_name)
            df_out.to_excel(writer, sheet_name=sheet, index=False)

            summary_rows.append(
                {
                    "Class": class_name,
                    "Total_edges": len(df_out),
                    "Similarity_edges": len(df_out[df_out["Edge_type"] == "Similarity"]),
                    "Mass_difference_edges": len(df_out[df_out["Edge_type"] == "Mass_difference"]),
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        if len(summary_df) == 0:
            summary_df = pd.DataFrame([{"Message": "No edges found with current parameters."}])
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    _progress(progress_callback, 100)
    _log(log_callback, f"Saved: {output_path}")

    return str(output_path)