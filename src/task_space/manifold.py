import abc
import os
import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from dotenv import load_dotenv


class TaskManifold(abc.ABC):
    """
    Abstract base class representing the measure space (T, d, mu).
    Decouples the mathematical operator logic from the specific data source (O*NET vs Synthetic).
    """
    def __init__(self):
        self.task_vectors = None  # The matrix X (n_tasks, n_features)
        self.task_ids = None      # Labels or IDs for the tasks
        self.adj_matrix = None    # The sparse graph operator
        self.n_tasks = 0

    @abc.abstractmethod
    def load_data(self, **kwargs):
        """
        Ingest raw data and populate self.task_vectors.
        """
        pass

    def build_graph(self, k=10, sigma=1.0):
        """
        Constructs the numerical realization graph (Remark 3.1).
        Uses k-NN with a Gaussian kernel to approximate local operators.
        """
        if self.task_vectors is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print(f"Building mutual-{k}NN graph for {self.n_tasks} tasks...")

        # 1. Standardize the metric space
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(self.task_vectors)

        # 2. Compute k-NN
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_norm)
        self.adj_matrix = nbrs.kneighbors_graph(X_norm, mode='connectivity')

        # Note: In the future, we can apply the Gaussian kernel here: exp(-d^2 / sigma)
        return self.adj_matrix


class SyntheticManifold(TaskManifold):
    """
    Generates plausible dummy data for testing the differential inclusions
    without waiting for O*NET API keys.
    """
    def load_data(self, n_samples=1000, n_features=50, n_clusters=20):
        print(f"Generating synthetic task space: {n_samples} tasks, {n_clusters} occupation clusters.")

        # Generate 'blobs' to mimic how tasks cluster into occupations
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

        # Add some noise to make it realistic (tasks aren't perfectly aligned)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.5, X.shape)

        self.task_vectors = X + noise
        self.task_ids = [f"TASK_{i:04d}" for i in range(n_samples)]
        self.n_tasks = n_samples
        print("Synthetic data loaded.")


class OnetManifold(TaskManifold):
    """
    Realization using live O*NET Web Services data.
    Fetches Work Activities and Abilities to construct task vectors.
    """

    # Default diverse occupation sample for testing
    DEFAULT_OCCUPATIONS = [
        '15-1252.00',  # Software Developers
        '51-4041.00',  # Machinists
        '29-1141.00',  # Registered Nurses
        '41-3031.00',  # Securities Sales Agents
        '53-3032.00',  # Heavy Truck Drivers
        '25-1011.00',  # Business Teachers, Postsecondary
        '47-2111.00',  # Electricians
        '13-2011.00',  # Accountants and Auditors
        '35-2014.00',  # Cooks, Restaurant
        '43-4051.00',  # Customer Service Representatives
    ]

    def __init__(self, api_key=None):
        super().__init__()
        load_dotenv()
        self.api_key = api_key or os.getenv('ONET_API_KEY')
        if not self.api_key:
            raise ValueError("O*NET API key required. Set ONET_API_KEY in .env")
        # V2 API base URL (not the legacy /ws/ endpoint)
        self.base_url = "https://api-v2.onetcenter.org"
        self.occupation_data = {}  # Raw data keyed by SOC code
        self.element_ids = []      # Ordered list of all element IDs (feature names)
        self.occupation_codes = []  # Ordered list of occupation codes

    def _fetch_endpoint(self, soc_code, endpoint):
        """
        Fetch data from a specific O*NET V2 API endpoint.
        Handles pagination to get all elements.
        """
        url = f"{self.base_url}/online/occupations/{soc_code}/details/{endpoint}"
        headers = {
            'Accept': 'application/json',
            'X-API-Key': self.api_key
        }

        all_elements = []
        while url:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            all_elements.extend(data.get('element', []))
            # Handle pagination
            url = data.get('next')

        return {'element': all_elements}

    def fetch_work_activities(self, soc_code, include_level=False):
        """
        Fetch Work Activities for an occupation from V2 API.

        Args:
            soc_code: SOC occupation code
            include_level: If True, also fetch level data (requires extra API calls)

        Returns:
            dict mapping element IDs to {name, importance, level (if requested), score}
        """
        data = self._fetch_endpoint(soc_code, 'work_activities')
        elements = {}
        for item in data.get('element', []):
            element_id = item.get('id', '')
            name = item.get('name', '')
            importance = item.get('importance')
            if importance is not None:
                elements[f"WA_{element_id}"] = {
                    'name': name,
                    'importance': float(importance),
                    'level': None,  # Will be filled if include_level=True
                    'score': float(importance)  # Default: importance only
                }
        return elements

    def fetch_abilities(self, soc_code, include_level=False):
        """
        Fetch Abilities for an occupation from V2 API.

        Args:
            soc_code: SOC occupation code
            include_level: If True, also fetch level data (requires extra API calls)

        Returns:
            dict mapping element IDs to {name, importance, level (if requested), score}
        """
        data = self._fetch_endpoint(soc_code, 'abilities')
        elements = {}
        for item in data.get('element', []):
            element_id = item.get('id', '')
            name = item.get('name', '')
            importance = item.get('importance')
            if importance is not None:
                elements[f"AB_{element_id}"] = {
                    'name': name,
                    'importance': float(importance),
                    'level': None,
                    'score': float(importance)
                }
        return elements

    def _fetch_element_levels(self, element_type, element_id, occupation_codes):
        """
        Fetch Level scores for a specific element across occupations.

        The V2 API provides Level data through the element-centric endpoint:
        /online/onet_data/{element_type}/{element_id}

        Args:
            element_type: 'work_activities' or 'abilities'
            element_id: The O*NET element ID (e.g., '4.A.3.b.1')
            occupation_codes: List of SOC codes to look for

        Returns:
            dict mapping SOC codes to level scores
        """
        url = f"{self.base_url}/online/onet_data/{element_type}/{element_id}"
        headers = {
            'Accept': 'application/json',
            'X-API-Key': self.api_key
        }

        occupation_set = set(occupation_codes)
        levels = {}

        while url and occupation_set:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                break

            data = response.json()
            for occ in data.get('occupation', []):
                code = occ.get('code')
                if code in occupation_set:
                    levels[code] = occ.get('level')
                    occupation_set.discard(code)

            url = data.get('next')

        return levels

    def _enrich_with_levels(self, occupation_codes, score_mode='combined'):
        """
        Enrich occupation data with Level scores.

        Args:
            occupation_codes: List of SOC codes
            score_mode: How to compute final score:
                - 'importance': Use importance only (0-100)
                - 'level': Use level only (0-100)
                - 'combined': importance * level / 100 (0-100 scale)

        This requires fetching from element-centric endpoints, which is slower
        but provides the Level dimension needed for proper manifold construction.
        """
        print("  Enriching with Level data...")

        # Collect all unique element IDs
        all_elements = {}
        for soc_code, elements in self.occupation_data.items():
            for elem_id, elem_data in elements.items():
                if elem_id not in all_elements:
                    # Determine element type from prefix
                    if elem_id.startswith('WA_'):
                        all_elements[elem_id] = ('work_activities', elem_id[3:])
                    elif elem_id.startswith('AB_'):
                        all_elements[elem_id] = ('abilities', elem_id[3:])

        total_elements = len(all_elements)
        print(f"    Fetching levels for {total_elements} elements...")

        for idx, (elem_id, (elem_type, raw_id)) in enumerate(all_elements.items()):
            if (idx + 1) % 20 == 0:
                print(f"    Progress: {idx + 1}/{total_elements} elements...")

            levels = self._fetch_element_levels(elem_type, raw_id, occupation_codes)

            # Update occupation data with levels
            for soc_code, level in levels.items():
                if soc_code in self.occupation_data and elem_id in self.occupation_data[soc_code]:
                    elem = self.occupation_data[soc_code][elem_id]
                    elem['level'] = float(level) if level is not None else None

                    # Compute final score based on mode
                    if score_mode == 'importance':
                        elem['score'] = elem['importance']
                    elif score_mode == 'level':
                        elem['score'] = elem['level'] if elem['level'] is not None else 0
                    elif score_mode == 'combined':
                        if elem['level'] is not None:
                            # Combined score: importance * level / 100
                            # This keeps the score in 0-100 range
                            elem['score'] = elem['importance'] * elem['level'] / 100
                        else:
                            elem['score'] = elem['importance']

        print("    Level enrichment complete.")

    def load_data(self, occupation_codes=None, n_components=None,
                  include_level=False, score_mode='importance'):
        """
        Load O*NET data for specified occupations.

        Args:
            occupation_codes: List of SOC codes (e.g., ['15-1252.00', '51-4041.00'])
                             If None, uses DEFAULT_OCCUPATIONS.
            n_components: If set, apply PCA to reduce dimensionality (Phase I).
            include_level: If True, fetch Level data in addition to Importance.
                          This requires additional API calls but provides better
                          discrimination between occupations. (See Remark 4.1)
            score_mode: How to compute the final score for each element:
                - 'importance': Use Importance only (how frequently performed)
                - 'level': Use Level only (how complex/difficult)
                - 'combined': Importance * Level / 100 (default when include_level=True)

        The distinction between Importance and Level is critical:
        - A Data Entry Clerk has high Importance but low Level for "Working with Computers"
        - A Software Developer has high Importance AND high Level for the same task
        Using only Importance would incorrectly place these occupations close together.
        """
        occupation_codes = occupation_codes or self.DEFAULT_OCCUPATIONS
        self.occupation_codes = occupation_codes

        # If including level, default to combined score mode
        if include_level and score_mode == 'importance':
            score_mode = 'combined'

        print(f"Fetching O*NET data for {len(occupation_codes)} occupations...")
        print(f"  Score mode: {score_mode}" + (" (with Level)" if include_level else ""))

        # Collect all unique element IDs across occupations
        all_elements = set()

        for soc_code in occupation_codes:
            print(f"  Fetching {soc_code}...")
            try:
                work_activities = self.fetch_work_activities(soc_code)
                abilities = self.fetch_abilities(soc_code)

                self.occupation_data[soc_code] = {
                    **work_activities,
                    **abilities
                }
                all_elements.update(self.occupation_data[soc_code].keys())
            except requests.HTTPError as e:
                print(f"    Warning: Failed to fetch {soc_code}: {e}")
                continue

        # Enrich with Level data if requested
        if include_level:
            self._enrich_with_levels(occupation_codes, score_mode)

        # Create ordered element list (feature names)
        self.element_ids = sorted(list(all_elements))
        print(f"Total elements (features): {len(self.element_ids)}")

        # Build task vectors matrix (occupations x elements)
        n_occupations = len(self.occupation_data)
        n_features = len(self.element_ids)

        self.task_vectors = np.zeros((n_occupations, n_features))
        valid_codes = []

        for i, soc_code in enumerate(occupation_codes):
            if soc_code not in self.occupation_data:
                continue
            valid_codes.append(soc_code)
            row_idx = len(valid_codes) - 1
            for j, element_id in enumerate(self.element_ids):
                if element_id in self.occupation_data[soc_code]:
                    self.task_vectors[row_idx, j] = self.occupation_data[soc_code][element_id]['score']

        # Trim to valid occupations only
        self.task_vectors = self.task_vectors[:len(valid_codes), :]
        self.task_ids = valid_codes
        self.n_tasks = len(valid_codes)

        print(f"Loaded {self.n_tasks} occupations with {n_features} features.")

        # Apply PCA if requested (Phase I dimensionality reduction)
        if n_components and n_components < n_features:
            print(f"Applying PCA: {n_features} -> {n_components} dimensions...")
            pca = PCA(n_components=n_components)
            self.task_vectors = pca.fit_transform(self.task_vectors)
            self.pca_model = pca
            print(f"Variance explained: {sum(pca.explained_variance_ratio_):.2%}")

        print("O*NET data loaded successfully.")

    def get_element_name(self, element_id):
        """Get human-readable name for an element ID."""
        for soc_code, elements in self.occupation_data.items():
            if element_id in elements:
                return elements[element_id]['name']
        return element_id

    def get_element_index(self, element_id):
        """Get the feature index for an element ID."""
        if element_id in self.element_ids:
            return self.element_ids.index(element_id)
        return None

    def find_elements_by_name(self, name_substring):
        """
        Find element IDs containing a substring in their name.
        Useful for locating specific work activities or abilities.
        """
        matches = []
        for soc_code, elements in self.occupation_data.items():
            for element_id, data in elements.items():
                if name_substring.lower() in data['name'].lower():
                    matches.append((element_id, data['name']))
        return list(set(matches))
