# Dataset Download Instructions

## 🎯 YOU NEED TO DOWNLOAD THESE DATASETS

I'll build all the code to process them, but you need to manually download the data files.

---

## Dataset 1: NASA Milling Dataset (PRIMARY - REQUIRED)

### Step 1: Go to NASA Repository
**URL**: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

### Step 2: Find the Milling Dataset
- Look for **"Milling Data Set"** or **"Mill Tool Wear"**
- Usually listed under "Prognostics" or "Tool Wear" category

### Step 3: Download Files
Download all available files, typically:
- `mill_data_case_*.csv` or similar files
- Look for files containing:
  - Force measurements (Fx, Fy, Fz)
  - Acoustic emission (AE)
  - Vibration data
  - Tool wear measurements

### Step 4: Place Files Here
```
c:\imsa\SPINN_ASME\data\raw\nasa\
```

**Create this folder structure:**
```
data/
└── raw/
    └── nasa/
        ├── mill_data_case_1.csv
        ├── mill_data_case_2.csv
        └── [all other files]
```

---

## Dataset 2: PHM Society 2010 (SECONDARY - VALIDATION)

### Step 1: Search for PHM 2010 Challenge Data
**Search Terms**: "PHM Society 2010 Data Challenge milling"

**Possible Sources:**
- https://www.phmsociety.org/competition/phm/10
- Or search: "PHM 2010 prognostics challenge dataset"

### Step 2: Download Files
- Download the **training data** and **test data**
- Look for milling/cutting tool wear data
- Should contain force, vibration, and wear measurements

### Step 3: Place Files Here
```
c:\imsa\SPINN_ASME\data\raw\phm\
```

**Create this folder structure:**
```
data/
└── raw/
    └── phm/
        ├── train/
        │   └── [training files]
        └── test/
            └── [test files]
```

---

## 📁 Final Directory Structure Should Look Like:

```
c:\imsa\SPINN_ASME\
├── data/
│   ├── raw/
│   │   ├── nasa/          ← PUT NASA FILES HERE
│   │   │   ├── *.csv
│   │   │   └── [data files]
│   │   └── phm/           ← PUT PHM FILES HERE
│   │       ├── train/
│   │       └── test/
│   ├── processed/         ← I'll create this automatically
│   └── download_data.py   ← I'll create this
├── [rest of project files]
```

---

## ⚠️ If You Can't Find The Datasets

**Option 1**: 
- Email me the links you find and I'll help identify the right files
- Sometimes these datasets move or get renamed

**Option 2**: 
- Look for alternative manufacturing datasets on:
  - Kaggle (search "tool wear", "CNC milling")
  - IEEE DataPort
  - UC Irvine Machine Learning Repository

**Option 3**:
- If absolutely stuck, let me know and I'll search for mirror/alternative sources

---

## ✅ What To Do After Downloading

1. Download both datasets to the folders above
2. Run: `python data/download_data.py --check` (I'll create this script)
3. I'll automatically:
   - Verify the data
   - Extract relevant features
   - Preprocess and normalize
   - Create train/val/test splits
   - Generate data summary report

---

## 📊 What I Expect From The Data

### NASA Dataset Should Have:
- **Time series data** from multiple cutting experiments
- **Force measurements**: Fx, Fy, Fz (Newtons)
- **Tool wear**: Flank wear in micrometers (μm)
- **Process parameters**: Speed, feed, depth of cut
- Multiple tool life cycles (hopefully 10+ experiments)

### PHM Dataset Should Have:
- Similar force and wear measurements
- Multiple tool conditions
- Can be used for validation and testing

---

## ⏱️ TIME ESTIMATE

**Download time**: 10-30 minutes (depends on file sizes, usually 100MB-1GB)
**My processing time**: Automated once you have the files

---

## 🚨 DO THIS FIRST

**IMMEDIATE ACTION:**
1. Go to NASA website now: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
2. Download the milling dataset
3. Put files in `c:\imsa\SPINN_ASME\data\raw\nasa\`
4. Let me know when done - I'll build the rest!

**While you download, I'm building all the code...**
