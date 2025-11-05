"""
NASA Milling Dataset Preprocessing
Handles data cleaning, feature extraction, and train/val/test splitting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy not installed. Install with: pip install scipy")


class MillingDataPreprocessor:
    """Preprocess NASA milling dataset for PINN training"""
    
    def __init__(self, 
                 raw_data_dir: str = "data/raw/nasa",
                 processed_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.metadata = {}
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load NASA milling data from MATLAB or CSV files"""
        print("\n📂 Loading raw data files...")
        
        # Check for MATLAB file first (NASA dataset format)
        mat_files = list(self.raw_data_dir.glob("*.mat"))
        csv_files = list(self.raw_data_dir.glob("*.csv"))
        txt_files = list(self.raw_data_dir.glob("*.txt"))
        
        if mat_files and HAS_SCIPY:
            print(f"   Found MATLAB file: {mat_files[0].name}")
            return self._load_matlab_data(mat_files[0])
        elif csv_files or txt_files:
            return self._load_csv_data(csv_files, txt_files)
        else:
            raise FileNotFoundError(
                f"No data files found in {self.raw_data_dir}. "
                "Please download the dataset first."
            )
    
    def _load_matlab_data(self, mat_file: Path) -> pd.DataFrame:
        """Load NASA milling dataset from MATLAB file"""
        print(f"   Loading MATLAB file: {mat_file.name}")
        
        if not HAS_SCIPY:
            raise ImportError("scipy is required to load .mat files. Install with: pip install scipy")
        
        # Load MATLAB file
        mat_data = loadmat(str(mat_file))
        
        print(f"   📊 MATLAB file contents:")
        for key in mat_data.keys():
            if not key.startswith('__'):
                print(f"      - {key}: {type(mat_data[key])}")
        
        # NASA milling dataset structure: mat_data['mill'] contains experiments
        if 'mill' in mat_data:
            mill_data = mat_data['mill']
            print(f"   Found 'mill' data structure with shape: {mill_data.shape}")
            return self._parse_mill_structure(mill_data)
        else:
            # Try to parse any large array
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    if value.size > 100:  # Large enough to be data
                        print(f"   Using '{key}' as data array")
                        return self._parse_mill_structure(value)
        
        raise ValueError("Could not find valid data structure in MATLAB file")
    
    def _parse_mill_structure(self, mill_data) -> pd.DataFrame:
        """Parse the NASA mill data structure into DataFrame"""
        print("\n   🔍 Parsing mill data structure...")
        
        all_experiments = []
        
        # NASA mill dataset has structure: mill[0,case_number]
        # Each case contains: case, VB, time, DOC, feed, material, smcAC, smcDC, vib_table, vib_spindle, AE_table, AE_spindle
        n_cases = mill_data.shape[1] if len(mill_data.shape) > 1 else 1
        
        print(f"   Found {n_cases} experimental cases")
        
        for case_idx in range(n_cases):  # Process all cases (typically 167)
            try:
                case_data = mill_data[0, case_idx]
                
                # Extract experiment number
                case_num = int(case_data['case'][0, 0])
                
                # Extract tool wear (VB - flank wear in mm)
                vb = float(case_data['VB'][0, 0])
                
                # Extract cutting parameters
                doc = float(case_data['DOC'][0, 0])  # Depth of cut (mm)
                feed = float(case_data['feed'][0, 0])  # Feed rate (mm/rev)
                
                # Extract force measurements
                # smcAC and smcDC are force sensors
                # Using smcAC for primary force measurement
                force_ac = case_data['smcAC']  # Shape: (n_samples, 1)
                force_dc = case_data['smcDC']  # Shape: (n_samples, 1)
                
                # Get vibration data (can be used as additional features)
                vib_table = case_data['vib_table']  # Shape: (n_samples, 1)
                vib_spindle = case_data['vib_spindle']  # Shape: (n_samples, 1)
                
                n_samples = force_ac.shape[0]
                
                # Downsample to reduce data size (take every 10th sample)
                # NASA dataset has 9000 samples per case - too many
                downsample_factor = 100
                indices = np.arange(0, n_samples, downsample_factor)
                
                # Create DataFrame for this experiment
                exp_df = pd.DataFrame({
                    'experiment_id': case_num,
                    'case_index': case_idx,
                    'time': indices / 1000.0,  # Convert to seconds (assuming 1kHz sampling)
                    'tool_wear': vb,  # Tool wear in mm (constant for this case)
                    'depth_of_cut': doc,
                    'feed_rate': feed,
                    'force_ac': force_ac[indices].flatten(),  # AC force sensor
                    'force_dc': force_dc[indices].flatten(),  # DC force sensor
                    'vib_table': vib_table[indices].flatten(),  # Table vibration
                    'vib_spindle': vib_spindle[indices].flatten(),  # Spindle vibration
                })
                
                # Estimate 3-axis forces from available sensors
                # (NASA dataset doesn't have true Fx, Fy, Fz - we approximate)
                exp_df['force_x'] = exp_df['force_ac']  # Primary cutting force
                exp_df['force_y'] = exp_df['force_dc']  # Secondary force
                exp_df['force_z'] = exp_df['vib_table']  # Approximate from vibration
                
                # Add spindle speed (constant for this dataset - typical value)
                exp_df['spindle_speed'] = 3000.0  # RPM (typical for this dataset)
                
                all_experiments.append(exp_df)
                
                if (case_idx + 1) % 20 == 0:
                    print(f"      Processed {case_idx + 1}/{n_cases} cases...")
                
            except Exception as e:
                print(f"      ⚠️  Skipping case {case_idx + 1}: {e}")
                continue
        
        if not all_experiments:
            raise ValueError("No valid experiments could be parsed")
        
        combined_df = pd.concat(all_experiments, ignore_index=True)
        print(f"\n   ✅ Loaded {len(combined_df)} samples from {len(all_experiments)} experiments")
        print(f"   📊 Tool wear range: {combined_df['tool_wear'].min():.3f} - {combined_df['tool_wear'].max():.3f} mm")
        print(f"   📊 Unique experiments: {combined_df['experiment_id'].nunique()}")
        
        return combined_df
    
    def _load_csv_data(self, csv_files: List[Path], txt_files: List[Path]) -> pd.DataFrame:
        """Load data from CSV/TXT files (fallback method)"""
        all_data = []
        
        # Try loading CSV files
        for file in csv_files:
            print(f"   Loading: {file.name}")
            try:
                df = pd.read_csv(file)
                # Add file identifier
                df['experiment_id'] = file.stem
                all_data.append(df)
                print(f"      ✅ Loaded {len(df)} rows")
            except Exception as e:
                print(f"      ⚠️ Failed: {e}")
        
        if not all_data:
            raise ValueError("Could not load any data files successfully")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ Total data loaded: {len(combined_df)} rows from {len(all_data)} files")
        
        return combined_df
    
    def identify_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Automatically identify important columns"""
        print("\n🔍 Identifying column types...")
        
        column_map = {
            'time': None,
            'force_x': None,
            'force_y': None,
            'force_z': None,
            'tool_wear': None,
            'spindle_speed': None,
            'feed_rate': None,
            'depth_of_cut': None
        }
        
        # Common patterns for each column type
        patterns = {
            'time': ['time', 't', 'timestamp', 'seconds'],
            'force_x': ['fx', 'force_x', 'forcex', 'x_force', 'smcac'],
            'force_y': ['fy', 'force_y', 'forcey', 'y_force', 'smcdc'],
            'force_z': ['fz', 'force_z', 'forcez', 'z_force', 'smcalarmac'],
            'tool_wear': ['vb', 'wear', 'tool_wear', 'flank_wear'],
            'spindle_speed': ['speed', 'spindle', 'rpm', 'spindlespeed'],
            'feed_rate': ['feed', 'feedrate', 'feed_rate', 'vf'],
            'depth_of_cut': ['doc', 'depth', 'ae', 'ap', 'depth_of_cut']
        }
        
        # Try to match columns
        for col in df.columns:
            col_lower = col.lower().strip()
            for key, pattern_list in patterns.items():
                if any(pattern in col_lower for pattern in pattern_list):
                    column_map[key] = col
                    break
        
        # Print results
        print("\n📋 Column mapping:")
        for key, value in column_map.items():
            status = "✅" if value else "❌"
            print(f"   {status} {key}: {value}")
        
        return column_map
    
    def create_features(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """Create standardized feature set"""
        print("\n⚙️ Creating feature set...")
        
        # Check if data already has the correct columns (from MATLAB parsing)
        expected_cols = ['time', 'force_x', 'force_y', 'force_z', 'tool_wear', 
                        'spindle_speed', 'feed_rate', 'depth_of_cut']
        
        if all(col in df.columns for col in expected_cols):
            print("   ✅ Data already has correct column structure")
            features_df = df.copy()
        else:
            # Need to map columns
            features_df = pd.DataFrame()
            
            # Handle time/index
            if column_map['time']:
                features_df['time'] = df[column_map['time']]
            else:
                # Create synthetic time based on row index
                features_df['time'] = np.arange(len(df))
                print("   ⚠️ No time column found, using row index")
            
            # Force measurements
            for force_type in ['force_x', 'force_y', 'force_z']:
                if column_map[force_type]:
                    features_df[force_type] = df[column_map[force_type]]
                else:
                    # If missing, create placeholder (will be filtered later)
                    features_df[force_type] = 0
                    print(f"   ⚠️ {force_type} not found, using zeros")
            
            # Tool wear (target variable)
            if column_map['tool_wear']:
                features_df['tool_wear'] = df[column_map['tool_wear']]
            else:
                # Try to infer from data progression
                print("   ⚠️ Tool wear not found, attempting to create synthetic wear")
                # Simple linear wear model based on time
                features_df['tool_wear'] = features_df['time'] * 0.1  # Placeholder
            
            # Process parameters
            for param in ['spindle_speed', 'feed_rate', 'depth_of_cut']:
                if column_map[param]:
                    features_df[param] = df[column_map[param]]
                else:
                    # Use default values if missing
                    defaults = {
                        'spindle_speed': 3000,
                        'feed_rate': 0.5,
                        'depth_of_cut': 1.5
                    }
                    features_df[param] = defaults[param]
                    print(f"   ⚠️ {param} not found, using default: {defaults[param]}")
        
        # Derived features
        features_df['force_magnitude'] = np.sqrt(
            features_df['force_x']**2 + 
            features_df['force_y']**2 + 
            features_df['force_z']**2
        )
        
        # Material removal rate (simplified)
        features_df['mrr'] = (features_df['spindle_speed'] * 
                              features_df['feed_rate'] * 
                              features_df['depth_of_cut'])
        
        # Cumulative material removed (integrate over time)
        if 'experiment_id' in df.columns:
            features_df['experiment_id'] = df['experiment_id']
            features_df['cumulative_mrr'] = (
                features_df.groupby('experiment_id')['mrr']
                .cumsum()
            )
        else:
            features_df['cumulative_mrr'] = features_df['mrr'].cumsum()
        
        # Thermal proxy (heat generation estimate)
        features_df['heat_generation'] = (
            features_df['force_magnitude'] * 
            features_df['spindle_speed'] * 0.001  # Scaling factor
        )
        
        # Thermal displacement (simplified model)
        # Assuming linear relationship with cumulative heat
        features_df['cumulative_heat'] = (
            features_df.groupby('experiment_id')['heat_generation'].cumsum()
            if 'experiment_id' in features_df.columns
            else features_df['heat_generation'].cumsum()
        )
        
        # Thermal displacement estimate (using thermal expansion coefficient)
        # ΔL = α * L * ΔT, simplified for estimation
        alpha = 11.7e-6  # Steel thermal expansion coefficient
        L_tool = 100  # Tool length in mm
        # Assume temperature rise proportional to cumulative heat
        features_df['thermal_displacement'] = (
            alpha * L_tool * features_df['cumulative_heat'] * 0.01
        )
        
        print(f"\n✅ Created {len(features_df.columns)} features:")
        for col in features_df.columns:
            print(f"   - {col}")
        
        return features_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter data"""
        print("\n🧹 Cleaning data...")
        
        initial_len = len(df)
        
        # Remove NaN and inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        print(f"   Removed {initial_len - len(df)} rows with NaN/inf values")
        
        # Light outlier removal (only extreme outliers)
        # Use a more lenient z-score threshold
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        for col in numerical_cols:
            if col not in ['time', 'experiment_id', 'case_index']:
                # Only remove if std > 0 (avoid division by zero)
                if df[col].std() > 0:
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    before = len(df)
                    df = df[z_scores < 10]  # Very lenient - only extreme outliers
                    outlier_count += (before - len(df))
        
        print(f"   Removed {outlier_count} extreme outliers")
        
        # Ensure tool wear is non-negative
        if 'tool_wear' in df.columns:
            df = df[df['tool_wear'] >= 0]
        
        # Cap thermal displacement at reasonable maximum (based on NASA data: max ~0.36 mm)
        if 'thermal_displacement' in df.columns:
            before = len(df)
            df = df[df['thermal_displacement'] < 1.0]  # Remove corrupted values > 1mm
            removed = before - len(df)
            if removed > 0:
                print(f"   Removed {removed} rows with corrupted thermal_displacement values")
        
        print(f"✅ Clean data: {len(df)} rows remaining")
        
        return df.reset_index(drop=True)
    
    def normalize_data(self, 
                       train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Normalize features using StandardScaler"""
        print("\n📊 Normalizing data...")
        
        # Columns to normalize (ONLY INPUT FEATURES - exclude outputs, time, experiment_id)
        output_cols = ['tool_wear', 'thermal_displacement']
        cols_to_normalize = [col for col in train_df.columns 
                            if col not in ['time', 'experiment_id'] + output_cols]
        
        print(f"   Normalizing {len(cols_to_normalize)} INPUT features only")
        print(f"   Keeping outputs in original scale: {output_cols}")
        
        # Fit scaler on training data only
        self.scaler.fit(train_df[cols_to_normalize])
        
        # Transform all splits (inputs only)
        train_df[cols_to_normalize] = self.scaler.transform(train_df[cols_to_normalize])
        val_df[cols_to_normalize] = self.scaler.transform(val_df[cols_to_normalize])
        test_df[cols_to_normalize] = self.scaler.transform(test_df[cols_to_normalize])
        
        # Save scaler
        scaler_path = self.processed_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   Saved scaler to: {scaler_path}")
        
        return train_df, val_df, test_df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test sets"""
        print("\n✂️ Splitting data (70% train, 15% val, 15% test)...")
        
        # If we have multiple experiments, try to split by experiment
        if 'experiment_id' in df.columns:
            unique_experiments = df['experiment_id'].unique()
            print(f"   Found {len(unique_experiments)} experiments")
            
            if len(unique_experiments) >= 3:
                # Split experiments
                train_exp, temp_exp = train_test_split(
                    unique_experiments, test_size=0.3, random_state=42
                )
                val_exp, test_exp = train_test_split(
                    temp_exp, test_size=0.5, random_state=42
                )
                
                train_df = df[df['experiment_id'].isin(train_exp)]
                val_df = df[df['experiment_id'].isin(val_exp)]
                test_df = df[df['experiment_id'].isin(test_exp)]
            else:
                # Too few experiments, split within experiments
                train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
                val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        else:
            # Random split
            train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        print(f"   Train: {len(train_df)} samples")
        print(f"   Val:   {len(val_df)} samples")
        print(f"   Test:  {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, 
                           train_df: pd.DataFrame,
                           val_df: pd.DataFrame,
                           test_df: pd.DataFrame):
        """Save processed data to disk"""
        print("\n💾 Saving processed data...")
        
        train_path = self.processed_dir / "train.csv"
        val_path = self.processed_dir / "val.csv"
        test_path = self.processed_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"   ✅ Saved: {train_path}")
        print(f"   ✅ Saved: {val_path}")
        print(f"   ✅ Saved: {test_path}")
        
        # Save metadata
        metadata = {
            'n_features': len(train_df.columns),
            'feature_names': list(train_df.columns),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'statistics': {
                'tool_wear': {
                    'min': float(train_df['tool_wear'].min()),
                    'max': float(train_df['tool_wear'].max()),
                    'mean': float(train_df['tool_wear'].mean())
                },
                'thermal_displacement': {
                    'min': float(train_df['thermal_displacement'].min()),
                    'max': float(train_df['thermal_displacement'].max()),
                    'mean': float(train_df['thermal_displacement'].mean())
                }
            }
        }
        
        metadata_path = self.processed_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Saved metadata: {metadata_path}")
    
    def process_all(self):
        """Run complete preprocessing pipeline"""
        print("\n" + "="*70)
        print("🚀 STARTING DATA PREPROCESSING")
        print("="*70)
        
        try:
            # Load raw data
            df = self.load_raw_data()
            
            # Identify columns
            column_map = self.identify_columns(df)
            
            # Create features
            features_df = self.create_features(df, column_map)
            
            # Clean data
            clean_df = self.clean_data(features_df)
            
            # Split data
            train_df, val_df, test_df = self.split_data(clean_df)
            
            # Normalize
            train_df, val_df, test_df = self.normalize_data(train_df, val_df, test_df)
            
            # Save
            self.save_processed_data(train_df, val_df, test_df)
            
            print("\n" + "="*70)
            print("✅ PREPROCESSING COMPLETE!")
            print("="*70)
            print("\n🚀 Next steps:")
            print("   1. Review processed data in: data/processed/")
            print("   2. Check metadata.json for statistics")
            print("   3. Run training: python experiments/train_baseline.py")
            print("\n")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print("\n💡 Tip: Run 'python data/download_data.py --validate' first")


if __name__ == "__main__":
    preprocessor = MillingDataPreprocessor()
    preprocessor.process_all()
