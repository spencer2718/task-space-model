"""
Submit IPUMS CPS extract for occupation mobility analysis.
Per feasibility report v2, specification in ipums_extract_specification.md
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

from ipumspy import IpumsApiClient, MicrodataExtract

IPUMS_API_KEY = os.environ.get("IPUMS_API_KEY")
if not IPUMS_API_KEY:
    raise ValueError("IPUMS_API_KEY not found in environment. Check .env file.")

ipums = IpumsApiClient(IPUMS_API_KEY)

# Build sample list for 2015-2019 and 2022-2024 (excluding 2020-2021)
# Get available samples from API and filter to our target years
print("Fetching available CPS samples...")
available_samples = set(ipums.get_all_sample_info('cps'))

# Build desired sample list
desired_samples = []
for year in range(2015, 2020):  # 2015-2019
    for month in range(1, 13):
        desired_samples.append(f"cps{year}_{month:02d}s")

for year in range(2022, 2025):  # 2022-2024
    for month in range(1, 13):
        desired_samples.append(f"cps{year}_{month:02d}s")

# Filter to only available samples
samples = [s for s in desired_samples if s in available_samples]
missing = [s for s in desired_samples if s not in available_samples]

print(f"Requested {len(desired_samples)} samples, {len(samples)} available")
if missing:
    print(f"Missing samples: {missing[:10]}{'...' if len(missing) > 10 else ''}")
print(f"Sample range: {samples[0]} to {samples[-1]}")

# Variables per specification
variables = [
    "CPSIDP",      # Person linking key
    "CPSIDV",      # Validated person linking key
    "MISH",        # Month in sample
    "OCC2010",     # Harmonized occupation (primary)
    "OCC",         # Original occupation codes (robustness)
    "EMPSTAT",     # Employment status
    "LABFORCE",    # Labor force status
    "CLASSWKR",    # Class of worker (employer change proxy)
    "AGE",         # Demographics
    "SEX",
    "RACE",
    "EDUC",
    "PANLWT",      # Panel weight
    "STATEFIP",    # State (for clustering)
]

print(f"Requesting {len(variables)} variables: {variables}")

extract = MicrodataExtract(
    collection="cps",
    description="CPS mobility analysis: occupation transitions 2015-2019, 2022-2024",
    samples=samples,
    variables=variables,
)

# Submit
print("\nSubmitting extract...")
submitted = ipums.submit_extract(extract)
print(f"Extract submitted successfully!")
print(f"Extract number: {submitted.number}")

# Save extract number for later retrieval
data_dir = Path(__file__).parent.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

with open(data_dir / "extract_info.txt", "w") as f:
    f.write(f"collection: cps\n")
    f.write(f"number: {submitted.number}\n")
    f.write(f"samples: {len(samples)}\n")
    f.write(f"variables: {len(variables)}\n")
    f.write(f"description: CPS mobility analysis: occupation transitions 2015-2019, 2022-2024\n")

print(f"\nExtract info saved to: {data_dir / 'extract_info.txt'}")
print("\nNext steps:")
print("1. Run check_extract_status.py to monitor progress")
print("2. When complete, run download_extract.py to download data")
