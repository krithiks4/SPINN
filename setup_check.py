import sys
import os
from pathlib import Path

def check_python_version():
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro}")
        print("   Required: Python 3.8+")
        return False

def check_packages():
    required = ['torch', 'numpy', 'pandas', 'matplotlib', 'sklearn', 'tqdm']
    installed = []
    missing = []
    for package in required:
        try:
            __import__(package if package != 'sklearn' else 'sklearn')
            installed.append(package)
        except ImportError:
            missing.append(package)
    if installed:
        print(f"\n✅ Installed packages: {', '.join(installed)}")
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\n   Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages installed!")
        return True

def check_directories():
    required_dirs = [
        'data/raw/nasa',
        'data/processed',
        'models',
        'experiments',
        'utils',
        'results/figures',
        'results/metrics',
        'results/models'
    ]
    print("\n📁 Checking directory structure...")
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} (will be created)")
            path.mkdir(parents=True, exist_ok=True)
            all_exist = False
    if all_exist:
        print("\n✅ All directories exist!")
    else:
        print("\n⚠️  Created missing directories")
    return True

def check_dataset():
    nasa_dir = Path('data/raw/nasa')
    phm_dir = Path('data/raw/phm')
    print("\n📊 Checking datasets...")
    nasa_files = list(nasa_dir.glob('*.csv')) if nasa_dir.exists() else []
    phm_files = list(phm_dir.rglob('*.csv')) if phm_dir.exists() else []
    dataset_ready = False
    if nasa_files:
        print(f"   ✅ NASA dataset: {len(nasa_files)} CSV files found")
        dataset_ready = True
    else:
        print(f"   ❌ NASA dataset: NOT FOUND")
        print(f"      Download from: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        print(f"      Place in: {nasa_dir.absolute()}")
    if phm_files:
        print(f"   ✅ PHM dataset: {len(phm_files)} CSV files found")
    else:
        print(f"   ⚠️  PHM dataset: NOT FOUND (optional)")
    return dataset_ready

def check_jupyter():
    try:
        import jupyter
        print("\n✅ Jupyter Notebook: Available")
        return True
    except ImportError:
        print("\n⚠️  Jupyter Notebook: NOT INSTALLED")
        print("   Install with: pip install jupyter")
        return False

def main():
    print("="*70)
    print("🔍 SPINN PROJECT SETUP VERIFICATION")
    print("="*70)
    checks = {
        'Python Version': check_python_version(),
        'Python Packages': check_packages(),
        'Directory Structure': check_directories(),
        'Dataset': check_dataset(),
        'Jupyter': check_jupyter()
    }
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    for name, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name}")
    print("\n" + "="*70)
    if all(checks.values()):
        print("✅ ALL CHECKS PASSED - READY TO START!")
        print("\n🚀 Next step: Open 01_train_baseline.ipynb")
        print("   Command: jupyter notebook 01_train_baseline.ipynb")
    elif checks['Dataset']:
        print("⚠️  ALMOST READY - Just need to install packages")
        print("\n🔧 Run: pip install -r requirements.txt")
    else:
        print("⚠️  SETUP INCOMPLETE - Please complete the following:")
        print("\n📥 1. Download NASA dataset")
        print("      URL: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        print(f"      Place in: {Path('data/raw/nasa').absolute()}")
        print("\n📦 2. Install Python packages")
        print("      Command: pip install -r requirements.txt")
        print("\n📖 3. Read START_HERE.md for detailed instructions")
    print("="*70)

if __name__ == "__main__":
    main()
