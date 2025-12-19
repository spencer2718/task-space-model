"""
IPUMS Census extract definition and download for Test B.

Requires IPUMS_API_KEY in environment or .env file.

Downloads Census 5% samples for 1980, 1990, 2000 with:
- Geography (CNTYGP98 for 1980, PUMA for 1990/2000)
- Occupation codes
- Person weights
- Employment filter variables
"""

import os
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def get_api_client():
    """Get authenticated IPUMS API client."""
    from ipumspy import IpumsApiClient

    api_key = os.environ.get("IPUMS_API_KEY")
    if not api_key:
        raise ValueError("IPUMS_API_KEY not found in environment")

    return IpumsApiClient(api_key)


def create_extract_1980():
    """Create 1980 Census 5% extract definition."""
    from ipumspy import MicrodataExtract

    return MicrodataExtract(
        collection="usa",
        description="Test B: 1980 Census 5% for CZ employment",
        samples=["us1980a"],  # 1980 5% state sample
        variables=[
            "YEAR",
            "STATEFIP",
            "CNTYGP98",    # County group (1980 geography)
            "OCC1950",     # Occupation code (harmonized to 1950 basis)
            "PERWT",       # Person weight
            "AGE",
            "SEX",
            "EMPSTAT",     # Employment status
            "LABFORCE",    # Labor force status
        ],
    )


def create_extract_1990():
    """Create 1990 Census 5% extract definition."""
    from ipumspy import MicrodataExtract

    return MicrodataExtract(
        collection="usa",
        description="Test B: 1990 Census 5% for CZ employment",
        samples=["us1990a"],  # 1990 5% sample
        variables=[
            "YEAR",
            "STATEFIP",
            "PUMA",        # Public Use Microdata Area
            "OCC1990",     # 1990 occupation code
            "PERWT",
            "AGE",
            "SEX",
            "EMPSTAT",
            "LABFORCE",
        ],
    )


def create_extract_2000():
    """Create 2000 Census 5% extract definition."""
    from ipumspy import MicrodataExtract

    return MicrodataExtract(
        collection="usa",
        description="Test B: 2000 Census 5% for CZ employment",
        samples=["us2000a"],  # 2000 5% sample
        variables=[
            "YEAR",
            "STATEFIP",
            "PUMA",
            "OCC",         # 2000 occupation code
            "PERWT",
            "AGE",
            "SEX",
            "EMPSTAT",
            "LABFORCE",
        ],
    )


def submit_extracts(client, extracts: list) -> list[str]:
    """Submit extracts and return extract IDs."""
    extract_ids = []

    for extract in extracts:
        submitted = client.submit_extract(extract)
        extract_id = submitted.extract_id
        extract_ids.append(extract_id)
        print(f"Submitted: {extract.description} -> ID: {extract_id}")

    return extract_ids


def wait_for_extracts(client, extract_ids: list[str], poll_interval: int = 60):
    """Wait for all extracts to complete."""
    from ipumspy import IpumsExtractNotReady

    pending = set(extract_ids)
    completed = []

    while pending:
        for extract_id in list(pending):
            try:
                # Check if ready
                extract_info = client.extract_status(
                    collection="usa",
                    extract_id=extract_id
                )
                status = extract_info.get("status", "unknown")

                if status == "completed":
                    print(f"Extract {extract_id}: COMPLETED")
                    pending.remove(extract_id)
                    completed.append(extract_id)
                elif status == "failed":
                    print(f"Extract {extract_id}: FAILED")
                    pending.remove(extract_id)
                else:
                    print(f"Extract {extract_id}: {status}")
            except Exception as e:
                print(f"Extract {extract_id}: checking... ({e})")

        if pending:
            print(f"Waiting {poll_interval}s... ({len(pending)} pending)")
            time.sleep(poll_interval)

    return completed


def download_extracts(
    client,
    extract_ids: list[str],
    download_dir: Path,
):
    """Download completed extracts."""
    download_dir.mkdir(parents=True, exist_ok=True)

    for extract_id in extract_ids:
        print(f"Downloading extract {extract_id}...")
        client.download_extract(
            collection="usa",
            extract_id=extract_id,
            download_dir=download_dir,
        )
        print(f"Downloaded to {download_dir}")


def main():
    """Submit all extracts for Test B."""
    client = get_api_client()

    # Create extract definitions
    extracts = [
        create_extract_1980(),
        create_extract_1990(),
        create_extract_2000(),
    ]

    # Submit
    print("Submitting IPUMS extracts...")
    extract_ids = submit_extracts(client, extracts)

    print(f"\nExtract IDs: {extract_ids}")
    print("\nExtracts submitted. Use wait_and_download() to poll for completion.")

    return extract_ids


def wait_and_download(extract_ids: Optional[list[str]] = None):
    """Wait for extracts and download when ready."""
    client = get_api_client()

    if extract_ids is None:
        # Try to get from a saved file or prompt
        raise ValueError("Provide extract_ids from previous submission")

    download_dir = Path("data/external/ipums/census")

    # Wait
    print("Waiting for extracts to complete...")
    completed = wait_for_extracts(client, extract_ids)

    # Download
    if completed:
        print(f"\nDownloading {len(completed)} extracts...")
        download_extracts(client, completed, download_dir)
        print("Done!")
    else:
        print("No extracts completed successfully.")


if __name__ == "__main__":
    main()
