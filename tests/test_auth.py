"""
O*NET V2 API Authentication Probe

Tests connectivity to the V2 API and explores available endpoints.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import requests
from dotenv import load_dotenv

load_dotenv()


def test_auth():
    """Test basic authentication against V2 API."""
    api_key = os.getenv('ONET_API_KEY')
    if not api_key:
        print("ERROR: ONET_API_KEY not found in .env")
        return False

    print(f"API Key: {api_key[:10]}... (length: {len(api_key)})")

    base_url = "https://api-v2.onetcenter.org"
    headers = {
        'Accept': 'application/json',
        'X-API-Key': api_key
    }

    # Test 1: /about endpoint
    print("\n[Test 1] Checking /about endpoint...")
    url = f"{base_url}/about/"
    r = requests.get(url, headers=headers)
    print(f"  URL: {url}")
    print(f"  Status: {r.status_code}")

    if r.status_code == 200:
        print("  SUCCESS! API key is valid.")
        data = r.json()
        print(f"  Response keys: {list(data.keys())}")
        if 'api_version' in data:
            print(f"  API Version: {data['api_version']}")
    else:
        print(f"  FAILED: {r.text[:200]}")
        return False

    # Test 2: Explore root/help to find endpoints
    print("\n[Test 2] Exploring available endpoints...")
    for endpoint in ['/', '/help/', '/ws/', '/ws/online/', '/ws/mnm/']:
        url = f"{base_url}{endpoint}"
        r = requests.get(url, headers=headers)
        print(f"  {endpoint:<20} -> {r.status_code}")
        if r.status_code == 200:
            try:
                data = r.json()
                if isinstance(data, dict):
                    print(f"    Keys: {list(data.keys())[:5]}")
            except:
                pass

    # Test 3: Try to fetch occupation data for Software Developers
    print("\n[Test 3] Fetching occupation data for 15-1252.00 (Software Developers)...")

    # Try various endpoint patterns
    occupation_patterns = [
        '/ws/online/occupations/15-1252.00/',
        '/ws/mnm/careers/15-1252.00/',
        '/occupations/15-1252.00/',
        '/online/occupations/15-1252.00/',
    ]

    for pattern in occupation_patterns:
        url = f"{base_url}{pattern}"
        r = requests.get(url, headers=headers)
        print(f"  {pattern:<45} -> {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"    SUCCESS! Keys: {list(data.keys())}")
            break

    # Test 4: Try work_activities endpoint
    print("\n[Test 4] Looking for work_activities endpoint...")

    wa_patterns = [
        '/ws/online/occupations/15-1252.00/work_activities/',
        '/ws/mnm/careers/15-1252.00/work_activities/',
        '/occupations/15-1252.00/work_activities/',
    ]

    for pattern in wa_patterns:
        url = f"{base_url}{pattern}"
        r = requests.get(url, headers=headers)
        print(f"  {pattern:<50} -> {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"    SUCCESS! Keys: {list(data.keys())}")
            if 'element' in data:
                print(f"    Found {len(data['element'])} work activity elements")
            break

    return True


if __name__ == "__main__":
    success = test_auth()
    sys.exit(0 if success else 1)
