"""Download completed IPUMS extract."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

from ipumspy import IpumsApiClient, readers

IPUMS_API_KEY = os.environ.get("IPUMS_API_KEY")
if not IPUMS_API_KEY:
    raise ValueError("IPUMS_API_KEY not found in environment. Check .env file.")

ipums = IpumsApiClient(IPUMS_API_KEY)

# Read extract number
data_dir = Path(__file__).parent.parent / "data"
extract_file = data_dir / "extract_info.txt"

if not extract_file.exists():
    raise FileNotFoundError(f"No extract info found at {extract_file}. Run submit_ipums_extract.py first.")

with open(extract_file) as f:
    lines = f.readlines()
    extract_num = int([l for l in lines if l.startswith("number:")][0].split(":")[1].strip())

print(f"Downloading extract #{extract_num}...")

# Wait for completion (if not already complete)
print("Waiting for extract to complete (this may take a while for large extracts)...")
ipums.wait_for_extract(extract=extract_num, collection="cps")

# Download
print(f"Downloading to {data_dir}...")
ipums.download_extract(
    extract=extract_num,
    collection="cps",
    download_dir=data_dir
)

print(f"\n✓ Downloaded to {data_dir}")
print("\nDownloaded files:")
for f in data_dir.glob("cps_*"):
    print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

print("\nNext steps:")
print("1. Extract and load the data")
print("2. Apply transition filters from transition_filters.py")
print("3. Run the mobility analysis")
