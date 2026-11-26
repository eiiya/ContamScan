# ContamScan

ContamScan is a software toolkit for the suspect and non-targeted screening of chemical risk substances in food contact materials (FCMs) using high-resolution mass spectrometry (HRMS) data. It consists of two modules: **ContamScreen** (machine learning–based screening) and **ContamConfirm** (two-layer molecular network–based structural annotation).

A ready-to-run Windows package is available on the Releases page for users without a Python environment.

---

## 1. Overview

Food packaging systems contain numerous additives and non-intentionally added substances (NIAS), which make traditional target analysis insufficient for comprehensive safety assessment. ContamScan addresses this by combining:

- **ContamScreen**: A machine learning classifier that assigns unknown MS/MS spectra to one of six chemical classes based on binned fragment-ion and neutral-loss features.
- **ContamConfirm**: A two-layer molecular network (spectral similarity network + mass difference network) for rapid structural annotation of screened substances.

### Covered chemical classes

| Class ID | Class name |
|----------|------------|
| 1 | Alcoholamines |
| 2 | Parabens |
| 3 | Phthalates |
| 4 | Organophosphates |
| 5 | Citric acid esters |
| 6 | Perfluorinated compounds |

---

## 2. Requirements

- Python 3.8+
- NumPy
- pandas
- TensorFlow (2.x)
- matchms
- openpyxl
- spectral_entropy (for seed annotation)
- tools (bundled with this repository)

These dependencies are required only when running ContamScan from source. Users of the Windows package (Section 5.4) do not need to install Python or these packages.

Install all dependencies with:

    pip install -r requirements.txt

The pinned versions are listed in `requirements.txt`.

---

## 3. Model file and files

ContamScreen requires a pretrained model file (`.h5`). Because of its size (~275 MB), the model is not stored in this repository. Download it from the Releases page of this repository and place it in the `model/` directory.

A ready-to-run Windows package is also available on the Releases page. It is distributed as a zip archive containing the executable and all required dependencies; no Python installation is needed.

Expected project layout:

    ContamScan/
    ├── ContamScan_core.py
    ├── seed_annotation.py
    ├── tools.py
    ├── requirements.txt
    ├── README.md
    ├── LICENSE
    ├── .gitignore
    ├── model/
    │   └── screen_model.h5        <-- download from Releases
    ├── library/
    │   └── risk_substance_library.csv
    └── examples/
        ├── peak_list_example.xlsx
        ├── node_list_example.xlsx
        ├── seed_list_example.xlsx
        └── pmd_database_example.xlsx

- `seed_annotation.py`: a seed annotation tool used after ContamScreen. It matches screened spectra against a reference library by precursor m/z, retention time, and spectral entropy similarity.
- `tools.py`: helper functions used by `seed_annotation.py` for spectrum cleaning.
- `library/risk_substance_library.csv`: the reference library used by `seed_annotation.py` for seed identification.

If you place the model elsewhere, pass its path via `model_path`.

---

## 4. Input data

### 4.1 ContamScreen input (peak list)

An Excel file (`.xlsx`) containing at least the following columns:

| Column | Description |
|--------|-------------|
| `Spectrum reference file name` | Source file name of the spectrum |
| `Average Mz` | Precursor m/z |
| `Alignment ID` | Unique spectrum identifier |
| `Average Rt(min)` | Retention time (min) |
| `MS/MS spectrum` | MS/MS peaks in `mz:intensity mz:intensity` format |
| `Adduct type` | Adduct type (e.g. `[M+H]+`, `[M-H]-`) |

Adduct filtering is applied automatically according to the ionization mode:

- Positive mode (`P`): `[M+H]+`, `[M+NH4]+`
- Negative mode (`N`): `[M-H]-`

### 4.2 ContamConfirm input

1. **Node list** (`.xlsx`): sample spectra, with at least `PEPMASS` and `MSMS` columns. A `Name` / `name` / `Compound_Name` column is optional.
2. **Seed list** (`.xlsx`): reference/seed spectra, with at least `PEPMASS` and `MSMS` columns. A `Class` column is optional.
3. **PMD database** (`.xlsx`): formula-difference database, must contain a `Mass` column and a `Formula_Difference` column.

The `MSMS` column should contain peaks separated by `;`, each in `mz intensity` format (e.g. `101.0234 0.85; 145.0678 0.42`).

Example input files are provided in the `examples/` directory.

### 4.3 Seed annotation input

`seed_annotation.py` takes two inputs:

1. **Library file** (`.csv`): the risk substance library, containing at least the columns `Name`, `m/z`, `rt`, and `MS2`. The `MS2` column stores peaks in `mz intensity; mz intensity; ...` format.
2. **Screen result** (`.xlsx`): the output of ContamScreen, containing at least `PEPMASS`, `RT`, and `MSMS` columns.

The script matches each screened spectrum against the library by precursor m/z (5 ppm), retention time (±0.2 min, skipped if the library RT is empty), and spectral entropy similarity. Matches with similarity ≥ 0.7 are assigned confidence level 1 (RT matched) or level 2 (library RT empty); matches below 0.7 are kept with confidence level 0.

---

## 5. Usage

### 5.1 ContamScreen

    from ContamScan_core import run_contam_screen

    run_contam_screen(
        peak_list_path="examples/peak_list_example.xlsx",
        model_path="model/screen_model.h5",
        output_path="screen_result.xlsx",
        ion_mode="P",              # "P" for positive, "N" for negative
        progress_callback=print,   # optional
        log_callback=print,        # optional
    )

**Output**: an Excel file containing `ID`, `FILES`, `PEPMASS`, `RT`, `MSMS`, `pred_probability`, `predicted_classID`, and `predicted_class`.

### 5.2 ContamConfirm

    from ContamScan_core import run_contam_confirm

    run_contam_confirm(
        node_list_path="examples/node_list_example.xlsx",
        seed_list_path="examples/seed_list_example.xlsx",
        formula_db_path="examples/pmd_database_example.xlsx",
        output_path="confirm_result.xlsx",
        similarity_threshold=0.7,
        similarity_top_n=10,
        pmd_tolerance=0.01,
        pmd_top_n=3,
        progress_callback=print,   # optional
        log_callback=print,        # optional
    )

**Output**: an Excel file with one sheet per chemical class (columns include `Edge_type`, `Index_1/2`, `Source_1/2`, `NAME_1/2`, `ID_1/2`, `PEPMASS_1/2`, `Similarity_score`, `Matched_Peaks`, `Mass`, `Formula_Difference`) plus a `Summary` sheet.

### 5.3 Seed annotation

Before running, set the input paths in `seed_annotation.py` (`library.csv` path and screen result path), then run:

    python seed_annotation.py

The script writes an annotated Excel file containing `Matched_Name`, `CL`, and `Similarity_Score` columns.

### 5.4 Windows executable package (optional)

For users who prefer not to configure a Python environment, a ready-to-run Windows package is provided on the Releases page as a zip archive (e.g. `ContamScan.zip`). The archive contains the executable and all required dependencies. To use it:

1. Download the zip archive from the Releases page.
2. Extract the archive to a local folder.
3. Place the pretrained model file (Section 3) in the expected location, if it is not already bundled.
4. Double-click `ContamScan.exe` inside the extracted folder to run the software.

No Python installation or `pip install` is required for this package.

---

## 6. Key parameters

### 6.1 ContamScreen

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_mass` | 50 Da | Lower m/z bound for fragment binning |
| `end_mass` | 800 Da | Upper m/z bound for fragment binning |
| `bin_width` | 0.01 Da | Bin width for fragment and neutral-loss features |
| `start_neu` | 12 Da | Lower neutral-loss bound |
| `end_neu` | 200 Da | Upper neutral-loss bound |
| `ms2_error` | 0.01 Da | Tolerance for neutral-loss deduplication |

### 6.2 ContamConfirm

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.7 | Minimum modified cosine score to keep a similarity edge |
| `similarity_top_n` | 10 | Max similarity edges retained per node |
| `pmd_tolerance` | 0.01 Da | Mass tolerance for formula-difference matching |
| `pmd_top_n` | 3 | Max mass-difference edges retained per node |

### 6.3 Seed annotation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ppm_tolerance` | 5e-6 | Precursor m/z tolerance (5 ppm) |
| `rt_tolerance` | 0.2 min | Retention time tolerance |
| `spectral_similarity_threshold` | 0.7 | Minimum spectral entropy similarity for CL ≥ 1 |
| `ms2_da` | 0.01 Da | Fragment matching tolerance for spectral entropy |

---

## 7. Method summary

1. **Spectrum preprocessing**: peaks are normalized to the base peak, filtered (intensity > 1%, m/z below precursor), and converted to neutral-loss features.
2. **Feature encoding**: fragment ions and neutral losses are binned (0.01 Da) into a fixed-length vector, with the maximum binned intensity retained per bin.
3. **Classification**: a trained neural network predicts the chemical class and confidence for each spectrum; classes incompatible with the ionization mode are removed.
4. **Network construction**: within each predicted class, spectra are paired and scored by modified cosine similarity and by mass differences matched against a formula-difference database.
5. **Network optimization**: edges are deduplicated, prioritized (seed–sample edges first), and pruned to a per-node top-k to yield the final two-layer network.
6. **Seed annotation**: screened spectra are matched against a reference library by precursor m/z, retention time, and spectral entropy similarity, yielding confidence levels 0–2.

---

## 8. Notes and limitations

- ContamScreen requires the pretrained model file described in Section 3. Without it, ContamScreen cannot run.
- The Windows package on the Releases page is self-contained; users of this package do not need to install Python or the dependencies listed in Section 2.
- The classifier assigns spectra to one of six predefined classes; spectra not belonging to these classes may still receive a label and should be treated with caution.
- Similarity and mass-difference edges are **annotation hypotheses**, not confirmed structures. Final identification requires reference standards or orthogonal evidence.
- Seed annotation results are library-based matches; confidence level 0 indicates a precursor m/z and RT match with similarity below 0.7 and should be interpreted cautiously.
- Results are intended for research use in FCM safety assessment and do not by themselves constitute a food safety risk assessment.

---

## 9. Citation

If you use ContamScan in your work, please cite:

> *(Citation information to be added upon publication.)*

---

## 10. License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 11. Contact

For questions or bug reports, please open an issue on GitHub or contact the corresponding author.
