"""Check status of submitted IPUMS extract."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

from ipumspy import IpumsApiClient

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

print(f"Checking status of extract #{extract_num}...")

# Get status
extract_info = ipums.extract_status(extract=extract_num, collection="cps")
print(f"\nExtract {extract_num} status: {extract_info}")

# Provide guidance based on status
status = str(extract_info).lower()
if "completed" in status:
    print("\n✓ Extract is ready for download!")
    print("Run: python temp/cps_mobility/scripts/download_extract.py")
elif "queued" in status:
    print("\n⏳ Extract is queued. Check back later.")
elif "started" in status or "running" in status:
    print("\n⏳ Extract is being processed. Check back later.")
elif "failed" in status:
    print("\n✗ Extract failed. Check IPUMS website for details.")
elif "canceled" in status:
    print("\n✗ Extract was canceled.")
else:
    print(f"\nStatus: {extract_info}")
