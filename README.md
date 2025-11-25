# ROI Detection and ROI-Aware Compression for Liver Ultrasound DICOM (B-mode-and-CEUS-Liver)

This project implements a classical digital image processing pipeline for liver ultrasound:
-  [Data is available here](https://drive.google.com/drive/folders/1dcFbxS8MK-GCy3hG1u4kaQjIGN83CaLx?usp=sharing)
- Read ultrasound DICOM studies.
- Extract B-mode frames as PNG (splitting dual B-mode/CEUS panels when needed).
- Automatically detect electronic calipers in the B-mode image.
- Use caliper locations to drive seeded region growing and generate lesion masks.
- Measure lesion area in cm² using the physical caliper length from the machine.
- Optionally detect and remove cine-loop DICOMs.

The code is written in Python with OpenCV and standard scientific libraries.

## Repository Structure

```
├─ src/
│  ├─ inspect_dicom_tags.py       # Scan DICOM tags and write them to a CSV
│  ├─ filter_single_frame_dicoms.py         # Classify DICOMs as single-frame vs cine loop and (optionally) delete cine loops
│  ├─ extract_frames_color.py      # DICOM → PNG frames, with dual-panel (B-mode | CEUS) splitting
│  ├─ detect_calipers.py           # Detect yellow '+' electronic calipers in B-mode PNG
│  └─ auto_caliper_seg.py          # Caliper-seeded lesion segmentation + area measurement
│
└─ data/
   ├─ raw/                         # Original DICOMs (organized by patient/study)
   ├─ frames_full/                 # PNG frames extracted from DICOMs
   ├─ Bmode_frames/                # B-mode overlays from auto_caliper_seg.py
   ├─ Masks/                       # Binary lesion masks from auto_caliper_seg.py
```

**Important:** The repository include real patient liver data from the TCIA.

## Installation & Dependencies

### 1. Python environment

You’ll need:

- Python 3.9+
- Recommended: use a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install required packages

Install the core dependencies:

```bash
pip install numpy opencv-python pydicom imageio
```

Optional but often useful for further analysis/plotting (not required to run the core scripts):

```bash
pip install pandas matplotlib seaborn
```

## How to use the code?

**Note:** Make sure your virtual environment is activated before executing the scripts.

### Step 0 – Put your DICOM data in data/raw/

Organize DICOMs like:

```
data/raw/
  LiverUS-01/
    1-1.dcm
    1-2.dcm
    ...
  LiverUS-02/
    1-1.dcm
    ...
```

The exact naming doesn’t matter; the scripts recursively scan `data/raw/`.

### Step 1 – (Optional: only if you are interested in specifics) Inspect DICOM tags

**Script:** `src/inspect_dicom_tags.py`

**What it does:**

- Walks through all DICOM files under `data/raw/`.
- Reads each dataset with pydicom.
- Dumps all DICOM tags (tag ID, keyword, VR, value, etc.) into `data/inspect_dicom.csv`.

**Run:**

```bash
python src/inspect_dicom_tags.py
```

### Step 2 – Classify / remove cine-loop DICOMs

**Script name:** `src/filter_single_frame_dicoms.py`

**What it does:**

- Scans `data/raw/` for `.dcm` files.
- For each file:
  - Reads pixel data and `NumberOfFrames` if present.
  - Uses shape + tag heuristics to classify as:
    - single-frame (e.g. (H, W) or (H, W, C)) vs
    - cine loop (e.g. (T, H, W[, C]) with T > 1).
- Prints a summary of how many files are cine vs single.
- Optionally deletes cine-loop files from disk (depending on a `DRY_RUN` flag in the script).

**Run:**

- Make sure `DRY_RUN = True` in the script first. (To avoid accidentally deleting files)

```bash
python src/filter_single_frame_dicoms.py
```

- Check the printed classification.
- If you’re satisfied with the output, set `DRY_RUN = False` and re-run to actually delete cine loops.

**Warning:** Always keep a backup of `data/raw/` before running with deletion enabled.

### Step 3 – Extract PNG frames from DICOMs

**Script:** `src/extract_frames_color.py`

**What it does:**

- Reads all remaining DICOMs in `data/raw/`.
- For each DICOM:
  - Converts pixel data to 8-bit [0, 255] using per-image normalization.
  - If the frame looks like a dual-panel screenshot (very wide image with a dark vertical band in the middle), it is split into:
    - `<base>_fXXX_bmode.png`
    - `<base>_fXXX_ceus.png`
  - Otherwise, it saves `<base>_fXXX.png`.

All PNGs go into `data/frames_full/`.

**Run:**

```bash
python src/extract_frames_color.py
```

### Step 4 – Detect electronic calipers on a B-mode frame

**Script:** `src/detect_calipers.py`

**What it does:**

- Loads a single PNG frame (you set `IMG_PATH` at the bottom of the script).
- Converts to HSV and thresholds yellow pixels.
- Intersects yellow mask with the ultrasound wedge mask to suppress UI/text regions.
- Runs connected components to find small, roughly square “+” blobs.
- Returns the caliper centers (x, y) and saves debug images:
  - `_wedge_mask.png`
  - `_yellow_mask.png`
  - `_yellow_in_wedge.png`
  - `_calipers_overlay.png` (original image with red crosses on detected calipers)

**Run:**

- Open `src/detect_calipers.py`.
- At the bottom, change:

```python
IMG_PATH = Path(r"X:\DIP-Project\data\frames_full\LiverUS-01_1-5_f000_bmode.png") 
```

to point to one of the B-mode PNGs (in `data/frames_full/`).

```bash
python src/detect_calipers.py
```

Output goes to `data/detect_calipers_output/` (created automatically).

### Step 5 – Auto lesion segmentation & measurement (caliper-seeded)

**Script:** `src/auto_caliper_seg.py`

**What it does:**

- Loads a B-mode PNG frame.
- Calls `detect_caliper_seeds()` to get caliper centers and the wedge mask.
- Picks a caliper pair and uses their midpoint as a region-growing seed.
- Converts the image to grayscale, applies local smoothing, and restricts region growing to:
  - within the ultrasound wedge,
  - inside a disc around the seed whose radius depends on caliper distance,
  - outside the bright yellow overlay.
- Cleans the result with morphological operations and keeps the largest connected component → binary lesion mask.
- Uses the physical caliper length L from the machine (e.g. “L = 1.56 cm”) to estimate:
  - pixel → cm scale,
  - lesion area in cm².
- Saves:
  - a binary mask in `data/Masks/`
  - an overlay (original frame + lesion contour + calipers + text) in `data/bmode_frames/`
  - one row per image into `data/lesion_measurements.csv` (image name, seed, pixel distance, cm per pixel, area, etc.)

**Run:**

```python
img_path = Path(r"data/frames_full/LiverUS-01_1-3_f000_bmode.png")
measured_length_cm = 1.56  # from on-screen "L 1.56 cm"
```

```bash
python src/auto_caliper_seg.py
```

Check the overlay PNG in `data/bmode_frames/` to visually confirm that the red contour matches the lesion between the calipers.

## Notes

- **Data privacy:** This repo is intended to be shared without real patient data. Do not commit actual DICOMs with PHI; keep them locally under `data/raw/` and add `data/` to `.gitignore` if needed.

- The project is aimed to connect directly to topics from **Gonzalez & Woods - Digital Image Processing Book**:
  - Ch. 3 – Intensity transformations & spatial filtering
  - Ch. 5 – Image restoration and noise models
  - Ch. 9 – Morphological image processing
  - Ch. 10 – Image segmentation