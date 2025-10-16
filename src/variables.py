from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class Variable:
    name: str
    description: str
    values: List[str]
    temporal_order: int  # 0=early, 1=mid, 2=late, 3=outcome


# variable schema
VARIABLE_SCHEMA = {
    "FB": Variable(
        name="FB",
        description="First Blood - which team got the first kill",
        values=["0", "1"],
        temporal_order=0
    ),
    "FT": Variable(
        name="FT",
        description="First Tower - which team destroyed the first tower",
        values=["0", "1"],
        temporal_order=0
    ),
    "Gold10": Variable(
        name="Gold10",
        description="Gold difference at 10 minutes",
        values=["low", "neutral", "high"],
        temporal_order=0
    ),
    "Kills10": Variable(
        name="Kills10",
        description="Kill difference at 10 minutes",
        values=["behind", "even", "ahead"],
        temporal_order=0
    ),
    "Herald": Variable(
        name="Herald",
        description="Rift Herald secured by 14 minutes",
        values=["0", "1"],
        temporal_order=0
    ),
    "Gold20": Variable(
        name="Gold20",
        description="Gold difference at 20 minutes",
        values=["low", "neutral", "high"],
        temporal_order=1
    ),
    "Kills20": Variable(
        name="Kills20",
        description="Kill difference at 20 minutes",
        values=["behind", "even", "ahead"],
        temporal_order=1
    ),
    "Drakes": Variable(
        name="Drakes",
        description="Number of dragons secured by 25 minutes",
        values=["0", "1", "2", "3", "4+"],
        temporal_order=1
    ),
    "Soul": Variable(
        name="Soul",
        description="Dragon soul type obtained by 30 minutes",
        values=["None", "Infernal", "Mountain", "Ocean", "Cloud", "Hextech", "Chemtech"],
        temporal_order=2
    ),
    "Baron": Variable(
        name="Baron",
        description="Number of Baron Nashors killed by 30 minutes",
        values=["0", "1", "2+"],
        temporal_order=2
    ),
    "Inhibs": Variable(
        name="Inhibs",
        description="Inhibitor difference at 25 minutes",
        values=["<=-1", "0", ">=1"],
        temporal_order=2
    ),
    "Towers": Variable(
        name="Towers",
        description="Tower difference at 25 minutes",
        values=["<=-2", "-1_to_1", ">=2"],
        temporal_order=1
    ),
    "Win": Variable(
        name="Win",
        description="Match outcome (1=win, 0=loss)",
        values=["0", "1"],
        temporal_order=3
    )
}


def get_variable_info(var_name: str) -> Variable:
    return VARIABLE_SCHEMA[var_name]


def get_all_variables() -> List[str]:
    return list(VARIABLE_SCHEMA.keys())


def get_variables_by_temporal_order() -> Dict[int, List[str]]:
    groups = {}
    for name, var in VARIABLE_SCHEMA.items():
        order = var.temporal_order
        if order not in groups:
            groups[order] = []
        groups[order].append(name)
    return groups


def validate_variable_values(var_name: str, value: Any) -> bool:
    var = VARIABLE_SCHEMA.get(var_name)
    if var is None:
        return False
    return str(value) in var.values


def get_temporal_order(var_name: str) -> int:
    return VARIABLE_SCHEMA[var_name].temporal_order


def can_edge_exist(from_var: str, to_var: str) -> bool:
    # win cannot be a parent of any other variable
    # variables can only influence variables at the same or later temporal order
    if from_var == "Win":
        return False
    
    from_order = get_temporal_order(from_var)
    to_order = get_temporal_order(to_var)
    
    # can only go forward or stay at same time
    return from_order <= to_order


