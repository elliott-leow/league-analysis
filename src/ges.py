"""
Structure learning using GES (Greedy Equivalence Search) algorithm.

Uses causal-learn library to learn CPDAG structures from data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Set
import pickle
import logging

from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils

from .config import GES_PARAMS, FORBIDDEN_EDGES, MODELS_DIR
from .variables import get_all_variables, can_edge_exist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_forbidden_matrix(variables: List[str]) -> np.ndarray:
    """
    Create a forbidden edge matrix based on domain constraints.
    
    Args:
        variables: List of variable names in order
    
    Returns:
        Binary matrix where 1 indicates forbidden edge from i to j
    """
    n = len(variables)
    forbidden = np.zeros((n, n), dtype=int)
    
    var_to_idx = {var: i for i, var in enumerate(variables)}
    
    # Add explicit forbidden edges
    for from_var, to_var in FORBIDDEN_EDGES:
        if from_var in var_to_idx and to_var in var_to_idx:
            i = var_to_idx[from_var]
            j = var_to_idx[to_var]
            forbidden[i, j] = 1
    
    # Add temporal constraint violations
    for i, from_var in enumerate(variables):
        for j, to_var in enumerate(variables):
            if i != j and not can_edge_exist(from_var, to_var):
                forbidden[i, j] = 1
    
    logger.info(f"Created forbidden matrix with {forbidden.sum()} forbidden edges")
    return forbidden


def encode_data_for_ges(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Encode categorical data as integers for GES.
    
    Args:
        df: DataFrame with discretized categorical variables
    
    Returns:
        Tuple of (encoded data matrix, list of variable names)
    """
    variables = [col for col in df.columns if col in get_all_variables()]
    
    # Remove variables with zero variance (constant columns)
    vars_to_keep = []
    for var in variables:
        n_unique = df[var].nunique()
        if n_unique <= 1:
            logger.warning(f"Removing variable '{var}' - only {n_unique} unique value(s)")
        else:
            vars_to_keep.append(var)
    
    if len(vars_to_keep) == 0:
        raise ValueError("No variables with variance remaining after filtering!")
    
    variables = vars_to_keep
    
    # Encode each categorical variable as integer
    encoded = np.zeros((len(df), len(variables)), dtype=int)
    
    for i, var in enumerate(variables):
        # Convert to categorical codes
        encoded[:, i] = pd.Categorical(df[var]).codes
    
    logger.info(f"Encoded {len(variables)} variables with {len(df)} samples")
    return encoded, variables


def add_edges_to_outcome(
    edges: List[Tuple[str, str, str]],
    variables: List[str],
    df: pd.DataFrame,
    outcome_var: str = 'Win',
    top_k: int = 3
) -> List[Tuple[str, str, str]]:
    """
    Add edges to outcome variable based on mutual information.
    
    GES with BIC often doesn't add edges to outcome variables because they're
    predicted by the collective state. This post-processing adds the strongest
    direct predictors.
    
    Args:
        edges: Existing edge list
        variables: List of variable names
        df: Original data (for computing associations)
        outcome_var: Name of outcome variable
        top_k: Number of top predictors to add edges from
    
    Returns:
        Updated edge list with edges to outcome
    """
    from sklearn.metrics import mutual_info_score
    
    if outcome_var not in variables:
        return edges
    
    # Check if outcome already has incoming edges
    has_incoming = any(to_var == outcome_var for _, to_var, _ in edges)
    if has_incoming:
        return edges  # Already has edges, don't modify
    
    logger.info(f"Adding edges to outcome variable: {outcome_var}")
    
    # Compute mutual information with outcome for all other variables
    mi_scores = {}
    for var in variables:
        if var != outcome_var and var in df.columns:
            mi = mutual_info_score(df[var], df[outcome_var])
            mi_scores[var] = mi
    
    # Get top-k predictors
    top_predictors = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Add directed edges to outcome
    new_edges = list(edges)
    for var, mi in top_predictors:
        new_edges.append((var, outcome_var, 'directed'))
        logger.info(f"  Added edge: {var} → {outcome_var} (MI={mi:.3f})")
    
    return new_edges


def fit_ges(
    df: pd.DataFrame,
    score_func: str = 'local_score_BIC',
    use_constraints: bool = True,
    add_outcome_edges: bool = True
) -> dict:
    """
    Fit GES algorithm to learn CPDAG structure.
    
    Args:
        df: DataFrame with discretized variables
        score_func: Scoring function to use
        use_constraints: Whether to apply domain constraints
        add_outcome_edges: Whether to add edges to Win if missing
    
    Returns:
        Dictionary containing:
            - 'graph': GeneralGraph object (CPDAG)
            - 'variables': List of variable names
            - 'adjacency': Adjacency matrix
            - 'edges': List of edges
    """
    logger.info("Running GES structure learning")
    
    # Encode data
    data, variables = encode_data_for_ges(df)
    
    # Create forbidden matrix if using constraints
    forbidden = None
    if use_constraints:
        forbidden = create_forbidden_matrix(variables)
    
    # Run GES
    logger.info(f"Running GES with {len(variables)} variables and {len(data)} samples")
    
    try:
        # Note: causal-learn's GES doesn't directly support forbidden edges in the same way
        # We'll run GES and then filter edges post-hoc if needed
        record = ges(
            data,
            score_func=score_func,
            maxP=GES_PARAMS.get("maxP"),
            parameters=GES_PARAMS.get("parameters")
        )
        
        graph = record['G']
        
        # If constraints are used, remove forbidden edges
        if use_constraints and forbidden is not None:
            graph = remove_forbidden_edges(graph, forbidden)
        
        logger.info("GES completed successfully")
        
    except Exception as e:
        logger.error(f"Error running GES: {e}")
        raise
    
    # Extract edges
    edges = extract_edges(graph, variables)
    
    # Add edges to Win if missing
    if add_outcome_edges:
        edges = add_edges_to_outcome(edges, variables, df, outcome_var='Win', top_k=3)
    
    # Get adjacency matrix
    adjacency = graph.graph
    
    result = {
        'graph': graph,
        'variables': variables,
        'adjacency': adjacency,
        'edges': edges,
        'n_edges': len(edges)
    }
    
    logger.info(f"Learned structure with {len(edges)} edges")
    
    return result


def remove_forbidden_edges(graph, forbidden_matrix: np.ndarray):
    """
    Remove forbidden edges from a learned graph.
    
    Args:
        graph: GeneralGraph object
        forbidden_matrix: Binary matrix indicating forbidden edges
    
    Returns:
        Modified graph
    """
    n = forbidden_matrix.shape[0]
    
    for i in range(n):
        for j in range(n):
            if forbidden_matrix[i, j] == 1 and graph.graph[i, j] != 0:
                logger.debug(f"Removing forbidden edge {i} -> {j}")
                graph.graph[i, j] = 0
    
    return graph


def extract_edges(graph, variables: List[str]) -> List[Tuple[str, str, str]]:
    """
    Extract edges from graph object.
    
    Args:
        graph: GeneralGraph object
        variables: List of variable names
    
    Returns:
        List of tuples (from_var, to_var, edge_type)
        edge_type is one of: 'directed' (->), 'undirected' (--), 'bidirected' (<->)
    """
    edges = []
    adjacency = graph.graph
    n = len(variables)
    
    for i in range(n):
        for j in range(i + 1, n):  # Only check upper triangle to avoid duplicates
            edge_ij = adjacency[i, j]
            edge_ji = adjacency[j, i]
            
            if edge_ij == 0 and edge_ji == 0:
                # No edge
                continue
            elif edge_ij == -1 and edge_ji == 1:
                # Directed edge i -> j
                edges.append((variables[i], variables[j], 'directed'))
            elif edge_ij == 1 and edge_ji == -1:
                # Directed edge j -> i
                edges.append((variables[j], variables[i], 'directed'))
            elif edge_ij == -1 and edge_ji == -1:
                # Undirected edge i -- j
                edges.append((variables[i], variables[j], 'undirected'))
            elif edge_ij == 1 and edge_ji == 1:
                # Bidirected edge i <-> j (rare in CPDAG)
                edges.append((variables[i], variables[j], 'bidirected'))
    
    return edges


def save_ges_result(result: dict, rank: str, output_dir: Optional[Path] = None):
    """Save GES result to file."""
    if output_dir is None:
        output_dir = MODELS_DIR
    
    output_file = output_dir / f"ges_{rank}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    
    logger.info(f"Saved GES result to {output_file}")


def load_ges_result(rank: str, models_dir: Optional[Path] = None) -> dict:
    """Load GES result from file."""
    if models_dir is None:
        models_dir = MODELS_DIR
    
    input_file = models_dir / f"ges_{rank}.pkl"
    
    if not input_file.exists():
        raise FileNotFoundError(f"GES result not found: {input_file}")
    
    with open(input_file, 'rb') as f:
        result = pickle.load(f)
    
    logger.info(f"Loaded GES result from {input_file}")
    return result


def cpdag_to_dag(cpdag_edges: List[Tuple[str, str, str]], variables: List[str]) -> List[Tuple[str, str]]:
    """
    Convert CPDAG to a single DAG by orienting undirected edges.
    
    Uses a simple heuristic: orient undirected edges according to temporal order.
    
    Args:
        cpdag_edges: List of edges from CPDAG
        variables: List of variable names
    
    Returns:
        List of directed edges (from, to)
    """
    from .variables import get_temporal_order
    
    dag_edges = []
    
    for from_var, to_var, edge_type in cpdag_edges:
        if edge_type == 'directed':
            dag_edges.append((from_var, to_var))
        elif edge_type == 'undirected':
            # Orient based on temporal order
            from_order = get_temporal_order(from_var)
            to_order = get_temporal_order(to_var)
            
            if from_order < to_order:
                dag_edges.append((from_var, to_var))
            elif to_order < from_order:
                dag_edges.append((to_var, from_var))
            else:
                # Same temporal order - orient arbitrarily (alphabetically)
                if from_var < to_var:
                    dag_edges.append((from_var, to_var))
                else:
                    dag_edges.append((to_var, from_var))
        # Note: bidirected edges are problematic for DAG; we'll skip them
    
    logger.info(f"Converted CPDAG to DAG with {len(dag_edges)} directed edges")
    return dag_edges


def get_edge_set(edges: List[Tuple[str, str, str]]) -> Set[Tuple[str, str]]:
    """
    Get set of edges (ignoring direction for comparison).
    
    Returns:
        Set of (from, to) tuples where from < to alphabetically
    """
    edge_set = set()
    
    for from_var, to_var, edge_type in edges:
        # Normalize edge to always have alphabetically first variable first
        edge = tuple(sorted([from_var, to_var]))
        edge_set.add(edge)
    
    return edge_set

