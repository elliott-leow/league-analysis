import numpy as np
import pandas as pd
from typing import Dict, Any
from .config import DISCRETIZATION_CONFIG


def discretize_gold_diff(values: pd.Series, time: str) -> pd.Series:
    config = DISCRETIZATION_CONFIG[time]
    return pd.cut(
        values,
        bins=config["bins"],
        labels=config["labels"],
        include_lowest=True
    ).astype(str)


def discretize_drakes(values: pd.Series) -> pd.Series:
    config = DISCRETIZATION_CONFIG["Drakes"]
    return pd.cut(
        values,
        bins=config["bins"],
        labels=config["labels"],
        include_lowest=True
    ).astype(str)


def discretize_baron(values: pd.Series) -> pd.Series:
    config = DISCRETIZATION_CONFIG["Baron"]
    return pd.cut(
        values,
        bins=config["bins"],
        labels=config["labels"],
        include_lowest=True
    ).astype(str)


def discretize_towers(values: pd.Series) -> pd.Series:
    config = DISCRETIZATION_CONFIG["Towers"]
    return pd.cut(
        values,
        bins=config["bins"],
        labels=config["labels"],
        include_lowest=True
    ).astype(str)


def discretize_kills(values: pd.Series, time: str) -> pd.Series:
    config = DISCRETIZATION_CONFIG[time]
    return pd.cut(
        values,
        bins=config["bins"],
        labels=config["labels"],
        include_lowest=True
    ).astype(str)


def discretize_inhibs(values: pd.Series) -> pd.Series:
    config = DISCRETIZATION_CONFIG["Inhibs"]
    return pd.cut(
        values,
        bins=config["bins"],
        labels=config["labels"],
        include_lowest=True
    ).astype(str)


def discretize_soul(values: pd.Series) -> pd.Series:
    # fill missing with 'none'
    values = values.fillna("None")
    
    # map any variations to standard categories
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
    return values.fillna(0).astype(int).astype(str)


def discretize_all_variables(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    
    # discretize each variable
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
        
        # check for missing values
        missing = df[var_name].isna().sum()
        if missing > 0:
            results["warnings"].append(f"{var_name} has {missing} missing values")
        
        # check for invalid values
        valid_values = set(VARIABLE_SCHEMA[var_name].values)
        actual_values = set(df[var_name].dropna().unique())
        invalid = actual_values - valid_values
        
        if invalid:
            results["errors"].append(
                f"{var_name} has invalid values: {invalid}"
            )
            results["valid"] = False
        
        # value distribution
        results["statistics"][var_name] = df[var_name].value_counts().to_dict()
    
    return results


def get_discretization_summary(df: pd.DataFrame) -> pd.DataFrame:
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


