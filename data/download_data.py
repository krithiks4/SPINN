"""
NASA Milling Dataset Downloader and Validator
Handles automatic download, verification, and initial exploration of NASA milling data
"""

import os
import sys
import requests
import zipfile
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

class NASAMillingDownloader:
    """Download and verify NASA milling dataset"""
    
    def __init__(self, base_dir: str = "data/raw/nasa"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Known NASA milling dataset files (update these URLs if they change)
        self.dataset_urls = {
            # These are placeholder URLs - need to be updated with actual NASA repository links
            "case_1": "https://ti.arc.nasa.gov/c/3/",  # Placeholder
            "case_2": "https://ti.arc.nasa.gov/c/4/",  # Placeholder
        }
        
    def check_existing_files(self) -> Dict[str, bool]:
        """Check what files already exist"""
        print("\n🔍 Checking for existing data files...\n")
        
        results = {
            "csv_files": [],
            "txt_files": [],
            "other_files": [],
            "total_size_mb": 0
        }
        
        if not self.base_dir.exists():
            print(f"❌ Directory not found: {self.base_dir}")
            print(f"📁 Please create: {self.base_dir.absolute()}")
            return results
        
        for file in self.base_dir.rglob("*"):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                results["total_size_mb"] += size_mb
                
                if file.suffix.lower() == ".csv":
                    results["csv_files"].append(file.name)
                elif file.suffix.lower() == ".txt":
                    results["txt_files"].append(file.name)
                else:
                    results["other_files"].append(file.name)
        
        # Print results
        if results["csv_files"]:
            print(f"✅ Found {len(results['csv_files'])} CSV files:")
            for f in results["csv_files"][:10]:  # Show first 10
                print(f"   - {f}")
            if len(results["csv_files"]) > 10:
                print(f"   ... and {len(results['csv_files']) - 10} more")
        else:
            print("❌ No CSV files found")
        
        if results["txt_files"]:
            print(f"\n✅ Found {len(results['txt_files'])} TXT files")
        
        print(f"\n📊 Total data size: {results['total_size_mb']:.2f} MB")
        
        return results
    
    def validate_data_structure(self) -> bool:
        """Validate that downloaded data has expected structure"""
        print("\n🔬 Validating data structure...\n")
        
        csv_files = list(self.base_dir.glob("*.csv"))
        
        if not csv_files:
            print("❌ No CSV files found to validate")
            print("\n📋 Expected columns:")
            print("   - Time/timestamp")
            print("   - Force measurements (X, Y, Z)")
            print("   - Tool wear (VB - flank wear)")
            print("   - Process parameters (speed, feed, depth)")
            return False
        
        print(f"Found {len(csv_files)} CSV files. Checking first file...\n")
        
        try:
            # Check first CSV file
            df = pd.read_csv(csv_files[0], nrows=5)
            print(f"✅ Successfully loaded: {csv_files[0].name}")
            print(f"\n📊 Data shape: {df.shape} (first 5 rows shown)")
            print(f"\n📋 Columns found ({len(df.columns)}):")
            for col in df.columns:
                print(f"   - {col}")
            
            print(f"\n🔍 Sample data (first 2 rows):")
            print(df.head(2).to_string())
            
            # Check for expected patterns
            expected_patterns = ["force", "wear", "time", "vb", "fx", "fy", "fz"]
            found_patterns = []
            
            for pattern in expected_patterns:
                if any(pattern.lower() in col.lower() for col in df.columns):
                    found_patterns.append(pattern)
            
            if found_patterns:
                print(f"\n✅ Found expected patterns: {', '.join(found_patterns)}")
            else:
                print(f"\n⚠️  No standard patterns found, but data may still be usable")
            
            return True
            
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return False
    
    def download_file(self, url: str, filename: str) -> bool:
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            filepath = self.base_dir / filename
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            
            print(f"✅ Downloaded: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return False
    
    def manual_download_instructions(self):
        """Print manual download instructions"""
        print("\n" + "="*70)
        print("📥 MANUAL DOWNLOAD REQUIRED")
        print("="*70)
        print("\nPlease follow these steps:\n")
        print("1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        print("2. Find 'Milling Data Set' or 'Mill Tool Wear'")
        print("3. Download all available files")
        print(f"4. Place them in: {self.base_dir.absolute()}\n")
        print("Expected files:")
        print("   - mill.txt or similar")
        print("   - c1.csv, c2.csv, c3.csv... (case files)")
        print("   - Any documentation files")
        print("\n5. After downloading, run: python data/download_data.py --check")
        print("="*70 + "\n")


class PHMDataDownloader:
    """Download and verify PHM 2010 dataset"""
    
    def __init__(self, base_dir: str = "data/raw/phm"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def check_existing_files(self) -> Dict[str, bool]:
        """Check for PHM data files"""
        print("\n🔍 Checking PHM 2010 data...\n")
        
        results = {
            "train_files": [],
            "test_files": [],
            "total_size_mb": 0
        }
        
        if not self.base_dir.exists():
            print(f"❌ Directory not found: {self.base_dir}")
            return results
        
        train_dir = self.base_dir / "train"
        test_dir = self.base_dir / "test"
        
        if train_dir.exists():
            results["train_files"] = [f.name for f in train_dir.glob("*") if f.is_file()]
        
        if test_dir.exists():
            results["test_files"] = [f.name for f in test_dir.glob("*") if f.is_file()]
        
        for file in self.base_dir.rglob("*"):
            if file.is_file():
                results["total_size_mb"] += file.stat().st_size / (1024 * 1024)
        
        if results["train_files"]:
            print(f"✅ Found {len(results['train_files'])} training files")
        else:
            print("❌ No training files found")
        
        if results["test_files"]:
            print(f"✅ Found {len(results['test_files'])} test files")
        else:
            print("⚠️  No test files found")
        
        print(f"\n📊 Total size: {results['total_size_mb']:.2f} MB")
        
        return results
    
    def manual_download_instructions(self):
        """Print PHM download instructions"""
        print("\n" + "="*70)
        print("📥 PHM 2010 DATASET - MANUAL DOWNLOAD")
        print("="*70)
        print("\nPlease follow these steps:\n")
        print("1. Search for: 'PHM Society 2010 Data Challenge'")
        print("2. Or visit: https://www.phmsociety.org/competition/phm/10")
        print("3. Download training and test data")
        print(f"4. Place in: {self.base_dir.absolute()}")
        print("\nCreate this structure:")
        print(f"   {self.base_dir.absolute()}/")
        print("   ├── train/")
        print("   │   └── [training files]")
        print("   └── test/")
        print("       └── [test files]")
        print("\n5. After downloading, run: python data/download_data.py --check")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NASA & PHM Dataset Management")
    parser.add_argument("--check", action="store_true", help="Check existing data files")
    parser.add_argument("--validate", action="store_true", help="Validate data structure")
    parser.add_argument("--instructions", action="store_true", help="Show download instructions")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🛠️  SPINN DATASET MANAGER")
    print("="*70)
    
    # Initialize downloaders
    nasa_downloader = NASAMillingDownloader()
    phm_downloader = PHMDataDownloader()
    
    if args.instructions or (not args.check and not args.validate):
        # Show instructions by default
        nasa_downloader.manual_download_instructions()
        phm_downloader.manual_download_instructions()
        return
    
    if args.check:
        print("\n📦 CHECKING NASA DATASET")
        print("-" * 70)
        nasa_results = nasa_downloader.check_existing_files()
        
        print("\n📦 CHECKING PHM DATASET")
        print("-" * 70)
        phm_results = phm_downloader.check_existing_files()
        
        # Summary
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        nasa_ok = len(nasa_results["csv_files"]) > 0
        phm_ok = len(phm_results["train_files"]) > 0
        
        if nasa_ok:
            print("✅ NASA dataset: READY")
        else:
            print("❌ NASA dataset: MISSING - Run with --instructions")
        
        if phm_ok:
            print("✅ PHM dataset: READY")
        else:
            print("⚠️  PHM dataset: MISSING (optional for validation)")
        
        if nasa_ok:
            print("\n🚀 Next step: Run validation with --validate")
        else:
            print("\n📥 Next step: Download datasets (see --instructions)")
        print("="*70 + "\n")
    
    if args.validate:
        print("\n🔬 VALIDATING NASA DATASET")
        print("-" * 70)
        if nasa_downloader.validate_data_structure():
            print("\n✅ Validation passed! Ready for preprocessing.")
            print("\n🚀 Next step: Run preprocessing:")
            print("   python data/preprocess.py")
        else:
            print("\n❌ Validation failed. Check data files.")


if __name__ == "__main__":
    main()
