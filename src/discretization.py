"""
Discretization utilities for continuous and categorical variables.

Converts raw match data into discrete bins suitable for Bayesian networks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from .config import DISCRETIZATION_CONFIG


def discretize_gold_diff(values: pd.Series, time: str) -> pd.Series:
    """
    Discretize gold difference at a specific time point.
    
    Args:
        values: Gold difference values
        time: Either 'Gold10' or 'Gold20'
    
    Returns:
        Discretized series with labels
    """
    config = DISCRETIZATION_CONFIG[time]
    return pd.cut(
        values,
        bins=config["bins"],
        labels=config["labels"],
        include_lowest=True
    ).astype(str)


def discretize_drakes(values: pd.Series) -> pd.Series:
    """Discretize dragon count."""
    config = DISCRETIZATION_CONFIG["Drakes"]
    return pd.cut(
        values,
        bins=config["bins"],
        labels=config["labels"],
        include_lowest=True
    ).astype(str)


def discretize_baron(values: pd.Series) -> pd.Series:
    """Discretize baron kills."""
    config = DISCRETIZATION_CONFIG["Baron"]
    return pd.cut(
        values,
        bins=config["bins"],
        labels=config["labels"],
        include_lowest=True
    ).astype(str)


def discretize_towers(values: pd.Series) -> pd.Series:
    """Discretize tower difference."""
    config = DISCRETIZATION_CONFIG["Towers"]
    return pd.cut(
        values,
        bins=config["bins"],
        labels=config["labels"],
        include_lowest=True
    ).astype(str)


def discretize_kills(values: pd.Series, time: str) -> pd.Series:
    """
    Discretize kill difference at a specific time point.
    
    Args:
        values: Kill difference values
        time: Either 'Kills10' or 'Kills20'
    
    Returns:
        Discretized series with labels
    """
    config = DISCRETIZATION_CONFIG[time]
    return pd.cut(
        values,
        bins=config["bins"],
        labels=config["labels"],
        include_lowest=True
    ).astype(str)


def discretize_inhibs(values: pd.Series) -> pd.Series:
    """Discretize inhibitor difference."""
    config = DISCRETIZATION_CONFIG["Inhibs"]
    return pd.cut(
        values,
        bins=config["bins"],
        labels=config["labels"],
        include_lowest=True
    ).astype(str)


def discretize_soul(values: pd.Series) -> pd.Series:
    """
    Discretize/categorize soul type.
    
    Maps soul values to standard categories.
    """
    # Fill missing with 'None'
    values = values.fillna("None")
    
    # Map any variations to standard categories
    soul_map = {
        "none": "None",
        "None": "None",
        "infernal": "Infernal",
        "Infernal": "Infernal",
        "mountain": "Mountain",
        "Mountain": "Mountain",
        "ocean": "Ocean",
        "Ocean": "Ocean",
        "cloud": "Cloud",
        "Cloud": "Cloud",
        "hextech": "Hextech",
        "Hextech": "Hextech",
        "chemtech": "Chemtech",
        "Chemtech": "Chemtech"
    }
    
    return values.map(lambda x: soul_map.get(str(x), "None")).astype(str)


def discretize_binary(values: pd.Series) -> pd.Series:
    """Discretize binary variables (0/1)."""
    return values.fillna(0).astype(int).astype(str)


def discretize_all_variables(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Apply discretization to all variables in a dataframe.
    
    Args:
        df: DataFrame with raw variables
        inplace: Whether to modify df in place
    
    Returns:
        DataFrame with discretized variables
    """
    if not inplace:
        df = df.copy()
    
    # Discretize each variable
    discretization_map = {
        "FB": lambda x: discretize_binary(x),
        "FT": lambda x: discretize_binary(x),
        "Gold10": lambda x: discretize_gold_diff(x, "Gold10"),
        "Kills10": lambda x: discretize_kills(x, "Kills10"),
        "Herald": lambda x: discretize_binary(x),
        "Gold20": lambda x: discretize_gold_diff(x, "Gold20"),
        "Kills20": lambda x: discretize_kills(x, "Kills20"),
        "Drakes": lambda x: discretize_drakes(x),
        "Soul": lambda x: discretize_soul(x),
        "Baron": lambda x: discretize_baron(x),
        "Inhibs": lambda x: discretize_inhibs(x),
        "Towers": lambda x: discretize_towers(x),
        "Win": lambda x: discretize_binary(x)
    }
    
    for var, func in discretization_map.items():
        if var in df.columns:
            df[var] = func(df[var])
    
    return df


def validate_discretization(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate discretized data and return statistics.
    
    Returns:
        Dictionary with validation results
    """
    from .variables import VARIABLE_SCHEMA
    
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {}
    }
    
    for var_name in VARIABLE_SCHEMA.keys():
        if var_name not in df.columns:
            results["warnings"].append(f"Variable {var_name} not found in dataframe")
            continue
        
        # Check for missing values
        missing = df[var_name].isna().sum()
        if missing > 0:
            results["warnings"].append(f"{var_name} has {missing} missing values")
        
        # Check for invalid values
        valid_values = set(VARIABLE_SCHEMA[var_name].values)
        actual_values = set(df[var_name].dropna().unique())
        invalid = actual_values - valid_values
        
        if invalid:
            results["errors"].append(
                f"{var_name} has invalid values: {invalid}"
            )
            results["valid"] = False
        
        # Value distribution
        results["statistics"][var_name] = df[var_name].value_counts().to_dict()
    
    return results


def get_discretization_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get a summary of discretized variable distributions.
    
    Returns:
        DataFrame with value counts for each variable
    """
    summary_data = []
    
    for var in DISCRETIZATION_CONFIG.keys():
        if var in df.columns:
            counts = df[var].value_counts()
            for value, count in counts.items():
                summary_data.append({
                    "Variable": var,
                    "Value": value,
                    "Count": count,
                    "Percentage": f"{100 * count / len(df):.2f}%"
                })
    
    return pd.DataFrame(summary_data)


