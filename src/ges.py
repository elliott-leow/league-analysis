
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
    n = len(variables)
    forbidden = np.zeros((n, n), dtype=int)
    
    var_to_idx = {var: i for i, var in enumerate(variables)}
    
    # add explicit forbidden edges
    for from_var, to_var in FORBIDDEN_EDGES:
        if from_var in var_to_idx and to_var in var_to_idx:
            i = var_to_idx[from_var]
            j = var_to_idx[to_var]
            forbidden[i, j] = 1
    
    # add temporal constraint violations
    for i, from_var in enumerate(variables):
        for j, to_var in enumerate(variables):
            if i != j and not can_edge_exist(from_var, to_var):
                forbidden[i, j] = 1
    
    logger.info(f"Created forbidden matrix with {forbidden.sum()} forbidden edges")
    return forbidden


def encode_data_for_ges(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    variables = [col for col in df.columns if col in get_all_variables()]
    
    # remove variables with zero variance (constant columns)
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
    
    # encode each categorical variable as integer
    encoded = np.zeros((len(df), len(variables)), dtype=int)
    
    for i, var in enumerate(variables):
        # convert to categorical codes
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
    from sklearn.metrics import mutual_info_score
    
    if outcome_var not in variables:
        return edges
    
    # check if outcome already has incoming edges
    has_incoming = any(to_var == outcome_var for _, to_var, _ in edges)
    if has_incoming:
        return edges  # Already has edges, don't modify
    
    logger.info(f"Adding edges to outcome variable: {outcome_var}")
    
    # compute mutual information with outcome for all other variables
    mi_scores = {}
    for var in variables:
        if var != outcome_var and var in df.columns:
            mi = mutual_info_score(df[var], df[outcome_var])
            mi_scores[var] = mi
    
    # get top-k predictors
    top_predictors = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # add directed edges to outcome
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
    logger.info("Running GES structure learning")
    
    # encode data
    data, variables = encode_data_for_ges(df)
    
    # create forbidden matrix if using constraints
    forbidden = None
    if use_constraints:
        forbidden = create_forbidden_matrix(variables)
    
    # run GES
    logger.info(f"Running GES with {len(variables)} variables and {len(data)} samples")
    
    try:
        # note: causal-learn's GES doesn't directly support forbidden edges in the same way
        # we'll run GES and then filter edges post-hoc if needed
        record = ges(
            data,
            score_func=score_func,
            maxP=GES_PARAMS.get("maxP"),
            parameters=GES_PARAMS.get("parameters")
        )
        
        graph = record['G']
        
        # if constraints are used, remove forbidden edges
        if use_constraints and forbidden is not None:
            graph = remove_forbidden_edges(graph, forbidden)
        
        logger.info("GES completed successfully")
        
    except Exception as e:
        logger.error(f"Error running GES: {e}")
        raise
    
    # extract edges
    edges = extract_edges(graph, variables)
    
    # add edges to Win if missing
    if add_outcome_edges:
        edges = add_edges_to_outcome(edges, variables, df, outcome_var='Win', top_k=3)
    
    # get adjacency matrix
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
    n = forbidden_matrix.shape[0]
    
    for i in range(n):
        for j in range(n):
            if forbidden_matrix[i, j] == 1 and graph.graph[i, j] != 0:
                logger.debug(f"Removing forbidden edge {i} -> {j}")
                graph.graph[i, j] = 0
    
    return graph


def extract_edges(graph, variables: List[str]) -> List[Tuple[str, str, str]]:
    edges = []
    adjacency = graph.graph
    n = len(variables)
    
    for i in range(n):
        for j in range(i + 1, n):  # Only check upper triangle to avoid duplicates
            edge_ij = adjacency[i, j]
            edge_ji = adjacency[j, i]
            
            if edge_ij == 0 and edge_ji == 0:
                # no edge
                continue
            elif edge_ij == -1 and edge_ji == 1:
                # directed edge i -> j
                edges.append((variables[i], variables[j], 'directed'))
            elif edge_ij == 1 and edge_ji == -1:
                # directed edge j -> i
                edges.append((variables[j], variables[i], 'directed'))
            elif edge_ij == -1 and edge_ji == -1:
                # undirected edge i -- j
                edges.append((variables[i], variables[j], 'undirected'))
            elif edge_ij == 1 and edge_ji == 1:
                # bidirected edge i <-> j (rare in CPDAG)
                edges.append((variables[i], variables[j], 'bidirected'))
    
    return edges


def save_ges_result(result: dict, rank: str, output_dir: Optional[Path] = None):
    if output_dir is None:
        output_dir = MODELS_DIR
    
    output_file = output_dir / f"ges_{rank}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    
    logger.info(f"Saved GES result to {output_file}")


def load_ges_result(rank: str, models_dir: Optional[Path] = None) -> dict:
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
    from .variables import get_temporal_order
    
    dag_edges = []
    
    for from_var, to_var, edge_type in cpdag_edges:
        if edge_type == 'directed':
            dag_edges.append((from_var, to_var))
        elif edge_type == 'undirected':
            # orient based on temporal order
            from_order = get_temporal_order(from_var)
            to_order = get_temporal_order(to_var)
            
            if from_order < to_order:
                dag_edges.append((from_var, to_var))
            elif to_order < from_order:
                dag_edges.append((to_var, from_var))
            else:
                # same temporal order - orient arbitrarily (alphabetically)
                if from_var < to_var:
                    dag_edges.append((from_var, to_var))
                else:
                    dag_edges.append((to_var, from_var))
        # note: bidirected edges are problematic for DAG; we'll skip them
    
    logger.info(f"Converted CPDAG to DAG with {len(dag_edges)} directed edges")
    return dag_edges


def get_edge_set(edges: List[Tuple[str, str, str]]) -> Set[Tuple[str, str]]:
    edge_set = set()
    
    for from_var, to_var, edge_type in edges:
        # normalize edge to always have alphabetically first variable first
        edge = tuple(sorted([from_var, to_var]))
        edge_set.add(edge)
    
    return edge_set

