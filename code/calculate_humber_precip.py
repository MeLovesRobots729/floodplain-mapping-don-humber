# AI assistance: this file was written with the assistance of DeepSeek V3 (2026).
# The code was reviewed, tested, and modified by Alexander Zhang.

import pandas as pd
import numpy as np
from pathlib import Path

# Import paths from central configuration
from paths import PRECIP_DATA_DIR

# Path to your downloaded CSV files
precip_dir = PRECIP_DATA_DIR

# List of Humber gauges with their actual file names
gauges = [
    {"id": "HY038", "name": "East Humber at Mill Road", "file": "precip-total-finalhy038.csv"},
    {"id": "HY067", "name": "West Humber at HWY 7", "file": "precip-total-finalhy067.csv"},
]

precip_data = {}

for gauge in gauges:
    gauge_id = gauge["id"]
    file_name = gauge["file"]
    file_path = precip_dir / file_name
    
    print(f"\n📊 Processing: {gauge['name']} ({gauge_id})")
    print(f"   Looking for: {file_path}")
    
    # Check if file exists
    if not file_path.exists():
        print(f"   ❌ File not found: {file_name}")
        print(f"   Please check the file name and try again.")
        continue
    
    # Read the CSV and find the exact data start
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the line that says "timestamp,value"
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == 'timestamp,value' or line.strip().startswith('timestamp,value'):
            data_start_idx = i + 1
            break
    
    if data_start_idx is None:
        print(f"   ⚠️ Could not find 'timestamp,value' header. Skipping...")
        continue
    
    print(f"   Found data starting at row {data_start_idx + 1}")
    
    timestamps = []
    values = []
    
    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 2:
            ts = parts[0].strip('"')
            val = parts[1].strip('"')
            try:
                values.append(float(val))
                timestamps.append(ts)
            except ValueError:
                continue
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    if len(df) == 0:
        print(f"   ⚠️ No data rows found. Skipping...")
        continue
    
    print(f"   Total rows: {len(df)}")
    print(f"   First few rows:")
    print(df.head(5))
    
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f%z', errors='coerce')
    except:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    df['Year'] = df['timestamp'].dt.year
    df = df.dropna(subset=['Year'])
    
    if len(df) == 0:
        print(f"   ⚠️ No valid timestamp data. Skipping...")
        continue
    
    annual_totals = df.groupby('Year')['value'].sum()
    annual_totals = annual_totals[annual_totals > 100]
    
    if len(annual_totals) == 0:
        print(f"   ⚠️ No valid years found (all totals < 100mm).")
        continue
    
    mean_annual = annual_totals.mean()
    
    precip_data[gauge_id] = {
        'name': gauge['name'],
        'mean_annual': mean_annual,
        'years_of_data': len(annual_totals),
        'annual_totals': annual_totals
    }
    
    print(f"   ✅ Mean annual precipitation: {mean_annual:.2f} mm")
    print(f"   📅 Years of data: {len(annual_totals)}")
    print(f"   Yearly totals (mm):")
    for year, total in sorted(annual_totals.items()):
        print(f"      {year}: {total:.2f}")

# Summary
print("\n" + "="*60)
print("SUMMARY: HUMBER RIVER PRECIPITATION")
print("="*60)

if len(precip_data) == 0:
    print("⚠️ No data processed successfully.")
else:
    for gauge_id, data in precip_data.items():
        print(f"{data['name']} ({gauge_id}):")
        print(f"  Mean Annual: {data['mean_annual']:.2f} mm")
        print(f"  Years: {data['years_of_data']}")
        print()

    print("\n📌 VALUES FOR IDW INTERPOLATION:")
    for gauge_id, data in precip_data.items():
        print(f"{gauge_id}: {data['mean_annual']:.2f} mm")
