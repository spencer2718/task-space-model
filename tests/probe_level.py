"""
Probe O*NET V2 API for Level Scores

The "Importance" score tells us how frequently a task is performed.
The "Level" score tells us how complex/difficult the task is.

For proper manifold construction, we may need both to distinguish:
- Software Developer (high Level on "Working with Computers")
- Data Entry Clerk (high Importance but low Level on same task)

This script investigates if/how Level data is available in the V2 API.
"""

import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import requests
from dotenv import load_dotenv

load_dotenv()


def probe_work_activity_structure():
    """Inspect the full JSON structure of work activities."""
    api_key = os.getenv('ONET_API_KEY')
    base_url = "https://api-v2.onetcenter.org"
    headers = {
        'Accept': 'application/json',
        'X-API-Key': api_key
    }

    soc_code = '15-1252.00'  # Software Developers

    print("=" * 70)
    print("PROBING O*NET V2 API FOR LEVEL SCORES")
    print("=" * 70)

    # Step 1: Get full work_activities response
    print("\n[1] Fetching work_activities for Software Developers...")
    url = f"{base_url}/online/occupations/{soc_code}/details/work_activities"
    r = requests.get(url, headers=headers)
    data = r.json()

    print(f"    Status: {r.status_code}")
    print(f"    Top-level keys: {list(data.keys())}")
    print(f"    Total elements: {data.get('total', '?')}")

    # Step 2: Print FULL structure of first element
    print("\n[2] Full JSON structure of first Work Activity element:")
    print("-" * 70)
    if data.get('element'):
        first_elem = data['element'][0]
        print(json.dumps(first_elem, indent=2))
    print("-" * 70)

    # Step 3: Check all keys across all elements
    print("\n[3] All unique keys found in work activity elements:")
    all_keys = set()
    for elem in data.get('element', []):
        all_keys.update(elem.keys())
    print(f"    {sorted(all_keys)}")

    # Step 4: Look for Level-related fields
    print("\n[4] Searching for 'level' in element data...")
    for elem in data.get('element', []):
        for key, value in elem.items():
            if 'level' in key.lower():
                print(f"    Found: {key} = {value}")
            if isinstance(value, dict) and 'level' in str(value).lower():
                print(f"    Found in {key}: {value}")

    # Step 5: Try the 'related' URL to see if it has more data
    print("\n[5] Probing 'related' URL for deeper element data...")
    if data.get('element') and 'related' in data['element'][0]:
        related_url = data['element'][0]['related']
        print(f"    URL: {related_url}")
        r2 = requests.get(related_url, headers=headers)
        print(f"    Status: {r2.status_code}")
        if r2.status_code == 200:
            related_data = r2.json()
            print(f"    Keys: {list(related_data.keys())}")
            print("\n    Full related response:")
            print("-" * 70)
            print(json.dumps(related_data, indent=2))
            print("-" * 70)

    # Step 6: Try alternative endpoint patterns for scales/level
    print("\n[6] Probing alternative endpoints for Level data...")
    alt_endpoints = [
        f"/online/occupations/{soc_code}/details/work_activities?scale=LV",
        f"/online/occupations/{soc_code}/details/work_activities/scales",
        f"/online/occupations/{soc_code}/summary/work_activities",
    ]

    for endpoint in alt_endpoints:
        url = f"{base_url}{endpoint}"
        r = requests.get(url, headers=headers)
        print(f"    {endpoint:<60} -> {r.status_code}")
        if r.status_code == 200:
            alt_data = r.json()
            print(f"        Keys: {list(alt_data.keys())}")
            if alt_data.get('element'):
                sample = alt_data['element'][0]
                print(f"        Sample element keys: {list(sample.keys())}")

    # Step 7: Check if abilities have level
    print("\n[7] Checking abilities for Level scores...")
    url = f"{base_url}/online/occupations/{soc_code}/details/abilities"
    r = requests.get(url, headers=headers)
    data = r.json()

    if data.get('element'):
        print("    First ability element:")
        print(json.dumps(data['element'][0], indent=2))

    # Step 8: Compare two occupations on same element
    print("\n[8] Comparing 'Working with Computers' across occupations...")

    occupations = [
        ('15-1252.00', 'Software Developers'),
        ('43-9021.00', 'Data Entry Keyers'),  # Alternative if not in default list
        ('43-4051.00', 'Customer Service Reps'),
    ]

    for soc, name in occupations:
        url = f"{base_url}/online/occupations/{soc}/details/work_activities"
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print(f"    {name}: Failed to fetch ({r.status_code})")
            continue

        data = r.json()
        for elem in data.get('element', []):
            if 'Working with Computers' in elem.get('name', ''):
                print(f"    {name}:")
                print(f"        importance: {elem.get('importance')}")
                # Check for any level-like fields
                for k, v in elem.items():
                    if k not in ['id', 'related', 'name', 'description', 'importance']:
                        print(f"        {k}: {v}")
                break

    print("\n" + "=" * 70)
    print("PROBE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    probe_work_activity_structure()
