"""
Configuration module for the project.

Contains all hyperparameters, discretization thresholds, and paths.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for d in [REPORTS_DIR, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Data files
MATCH_DATA_FILE = DATA_DIR / "matchData.csv"
MATCH_IDS_FILE = DATA_DIR / "match_ids.csv"
PLAYERS_FILE = DATA_DIR / "players_8-14-25.csv"
MATCH_DATA_JSONL = DATA_DIR / "match_data.jsonl"

# Rank buckets
RANK_BUCKETS = {
    "Platinum": ["PLATINUM"],
    "Diamond": ["DIAMOND"],
    "Master": ["MASTER"],
    "Elite": ["GRANDMASTER", "CHALLENGER"]  # Merge GM + Challenger
}

# All possible ranks
ALL_RANKS = ["Platinum", "Diamond", "Master", "Elite"]

# Variable definitions
VARIABLES = [
    "FB",       # First Blood
    "FT",       # First Tower
    "Gold10",   # Gold diff @10 min
    "Kills10",  # Kill diff @10 min
    "Herald",   # Rift Herald by 14 min
    "Gold20",   # Gold diff @20 min
    "Kills20",  # Kill diff @20 min
    "Drakes",   # Dragon count by 25 min
    "Soul",     # Dragon soul by 30 min
    "Baron",    # Baron kills by 30 min
    "Inhibs",   # Inhibitor diff @25 min
    "Towers",   # Tower diff @25 min
    "Win"       # Team win
]

# Discretization thresholds
DISCRETIZATION_CONFIG = {
    "Gold10": {
        "bins": [-float('inf'), -1000, 1000, float('inf')],
        "labels": ["low", "neutral", "high"]
    },
    "Kills10": {
        "bins": [-float('inf'), -2.5, 2.5, float('inf')],
        "labels": ["behind", "even", "ahead"]
    },
    "Herald": {
        "values": [0, 1]
    },
    "Gold20": {
        "bins": [-float('inf'), -3000, 3000, float('inf')],
        "labels": ["low", "neutral", "high"]
    },
    "Kills20": {
        "bins": [-float('inf'), -5.5, 5.5, float('inf')],
        "labels": ["behind", "even", "ahead"]
    },
    "Drakes": {
        "bins": [-0.5, 0.5, 1.5, 2.5, 3.5, 100],  # Right edge inclusive
        "labels": ["0", "1", "2", "3", "4+"]
    },
    "Soul": {
        # Categorical mapping
        "values": ["None", "Infernal", "Mountain", "Ocean", "Cloud", "Hextech", "Chemtech"]
    },
    "Baron": {
        "bins": [-0.5, 0.5, 1.5, 100],
        "labels": ["0", "1", "2+"]
    },
    "Towers": {
        "bins": [-100, -1.5, 1.5, 100],
        "labels": ["<=-2", "-1_to_1", ">=2"]
    },
    "Inhibs": {
        "bins": [-100, -0.5, 0.5, 100],
        "labels": ["<=-1", "0", ">=1"]
    },
    "FB": {
        "values": [0, 1]
    },
    "FT": {
        "values": [0, 1]
    },
    "Win": {
        "values": [0, 1]
    }
}

# Domain constraints (blacklist edges)
# Format: (from, to) pairs that should NOT exist
FORBIDDEN_EDGES = [
    # Win cannot cause anything
    ("Win", "FB"),
    ("Win", "FT"),
    ("Win", "Gold10"),
    ("Win", "Kills10"),
    ("Win", "Herald"),
    ("Win", "Gold20"),
    ("Win", "Kills20"),
    ("Win", "Drakes"),
    ("Win", "Soul"),
    ("Win", "Baron"),
    ("Win", "Inhibs"),
    ("Win", "Towers"),
    
    # Temporal constraints: later events cannot cause earlier events
    # Late (order=2) -> Early (order=0) forbidden
    ("Baron", "FB"),
    ("Baron", "FT"),
    ("Baron", "Gold10"),
    ("Baron", "Kills10"),
    ("Baron", "Herald"),
    ("Soul", "FB"),
    ("Soul", "FT"),
    ("Soul", "Gold10"),
    ("Soul", "Kills10"),
    ("Soul", "Herald"),
    ("Inhibs", "FB"),
    ("Inhibs", "FT"),
    ("Inhibs", "Gold10"),
    ("Inhibs", "Kills10"),
    ("Inhibs", "Herald"),
    
    # Mid (order=1) -> Early (order=0) forbidden
    ("Gold20", "FB"),
    ("Gold20", "FT"),
    ("Gold20", "Gold10"),
    ("Gold20", "Kills10"),
    ("Gold20", "Herald"),
    ("Kills20", "FB"),
    ("Kills20", "FT"),
    ("Kills20", "Gold10"),
    ("Kills20", "Kills10"),
    ("Kills20", "Herald"),
    ("Drakes", "FB"),
    ("Drakes", "FT"),
    ("Drakes", "Gold10"),
    ("Drakes", "Kills10"),
    ("Drakes", "Herald"),
    ("Towers", "FB"),
    ("Towers", "FT"),
    ("Towers", "Gold10"),
    ("Towers", "Kills10"),
    ("Towers", "Herald"),
]

# GES parameters
GES_PARAMS = {
    "score_func": "local_score_BIC",  # BIC score (works with discrete data when properly encoded)
    "maxP": 4,  # Limit parents to avoid overfitting with discrete data
    "parameters": {
        'kfold': 1,  # No cross-validation for speed
        'lambda': 0.01  # Small regularization
    }
}

# CPT estimation parameters
CPT_PARAMS = {
    "estimator_type": "BayesianEstimator",  # or "MaximumLikelihoodEstimator"
    "prior_type": "BDeu",
    "equivalent_sample_size": 10  # Dirichlet prior strength
}

# Visualization parameters
VIZ_PARAMS = {
    "figure_size": (12, 8),
    "node_size": 3000,
    "font_size": 10,
    "edge_width": 2.0,
    "dpi": 150,
    "layout": "spring"  # or "hierarchical"
}

# Query examples
EXAMPLE_QUERIES = [
    {
        "name": "High Baron and Gold Advantage",
        "evidence": {"Baron": "1", "Gold20": "high"}
    },
    {
        "name": "Infernal Soul with Low Gold",
        "evidence": {"Soul": "Infernal", "Gold20": "low"}
    },
    {
        "name": "Multiple Drakes, First Tower, and Tower Advantage",
        "evidence": {"Drakes": "3", "FT": 1, "Towers": ">=2"}
    },
    {
        "name": "First Blood Only",
        "evidence": {"FB": 1}
    },
    {
        "name": "Baron and Soul",
        "evidence": {"Baron": "1", "Soul": "Mountain"}
    },
    {
        "name": "Herald and Early Kill Lead",
        "evidence": {"Herald": "1", "Kills10": "ahead"}
    },
    {
        "name": "Behind in Kills but Strong Objectives",
        "evidence": {"Kills20": "behind", "Drakes": "3", "Herald": "1"}
    },
    {
        "name": "Late Game Domination",
        "evidence": {"Baron": "1", "Inhibs": ">=1", "Soul": "Infernal"}
    }
]

# Random seed for reproducibility
RANDOM_SEED = 42

# Data sampling (for development/testing)
SAMPLE_SIZE = None  # None = use all data, or set to e.g., 5000 for testing

