"""
Probabilistic query module.

Performs inference on learned Bayesian networks to answer probabilistic queries.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

from .config import EXAMPLE_QUERIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_inference(model: DiscreteBayesianNetwork) -> VariableElimination:
    """
    Set up inference engine for a Bayesian network.
    
    Args:
        model: DiscreteBayesianNetwork with learned CPTs
    
    Returns:
        VariableElimination inference object
    """
    return VariableElimination(model)


def p_win_given(
    evidence: Dict[str, Any],
    model: DiscreteBayesianNetwork,
    inference: Optional[VariableElimination] = None
) -> float:
    """
    Compute P(Win=1 | evidence).
    
    Args:
        evidence: Dictionary of variable values
        model: DiscreteBayesianNetwork
        inference: Optional pre-initialized inference engine
    
    Returns:
        Probability of winning given evidence
    """
    if inference is None:
        inference = setup_inference(model)
    
    try:
        # Convert evidence values to strings if needed
        evidence_str = {k: str(v) for k, v in evidence.items()}
        
        # Perform inference
        result = inference.query(variables=["Win"], evidence=evidence_str)
        
        # Get P(Win=1)
        prob = result.values[result.state_names["Win"].index("1")]
        
        return prob
    
    except Exception as e:
        logger.error(f"Error computing P(Win | {evidence}): {e}")
        return np.nan


def query_multiple(
    queries: List[Dict[str, Any]],
    model: DiscreteBayesianNetwork
) -> pd.DataFrame:
    """
    Run multiple queries on a model.
    
    Args:
        queries: List of query dictionaries
        model: DiscreteBayesianNetwork
    
    Returns:
        DataFrame with query results
    """
    inference = setup_inference(model)
    results = []
    
    for i, query in enumerate(queries):
        evidence = query.get("evidence", {})
        name = query.get("name", f"Query {i+1}")
        
        prob = p_win_given(evidence, model, inference)
        
        results.append({
            "Query": name,
            "Evidence": str(evidence),
            "P(Win=1)": prob,
            "P(Win=0)": 1 - prob if not np.isnan(prob) else np.nan
        })
    
    return pd.DataFrame(results)


def run_example_queries(model: DiscreteBayesianNetwork, rank: str) -> pd.DataFrame:
    """
    Run example queries on a model.
    
    Args:
        model: DiscreteBayesianNetwork
        rank: Rank name (for logging)
    
    Returns:
        DataFrame with results
    """
    logger.info(f"Running example queries for rank: {rank}")
    
    results = query_multiple(EXAMPLE_QUERIES, model)
    
    print(f"\n{'='*80}")
    print(f"QUERY RESULTS FOR {rank.upper()}")
    print(f"{'='*80}\n")
    print(results.to_string(index=False))
    print()
    
    return results


def compute_conditional_probability(
    target: str,
    target_value: Any,
    evidence: Dict[str, Any],
    model: DiscreteBayesianNetwork,
    inference: Optional[VariableElimination] = None
) -> float:
    """
    Compute P(target=target_value | evidence).
    
    Args:
        target: Target variable name
        target_value: Target variable value
        evidence: Evidence dictionary
        model: DiscreteBayesianNetwork
        inference: Optional inference engine
    
    Returns:
        Conditional probability
    """
    if inference is None:
        inference = setup_inference(model)
    
    try:
        evidence_str = {k: str(v) for k, v in evidence.items()}
        result = inference.query(variables=[target], evidence=evidence_str)
        
        target_value_str = str(target_value)
        if target_value_str in result.state_names[target]:
            prob = result.values[result.state_names[target].index(target_value_str)]
            return prob
        else:
            logger.warning(f"Value {target_value} not found for variable {target}")
            return np.nan
    
    except Exception as e:
        logger.error(f"Error computing P({target}={target_value} | {evidence}): {e}")
        return np.nan


def get_most_probable_explanation(
    evidence: Dict[str, Any],
    model: DiscreteBayesianNetwork
) -> Dict[str, Any]:
    """
    Find the most probable explanation (MPE) for unobserved variables.
    
    Args:
        evidence: Observed variables
        model: DiscreteBayesianNetwork
    
    Returns:
        Dictionary with most probable values for all variables
    """
    from pgmpy.inference import BeliefPropagation
    
    try:
        bp = BeliefPropagation(model)
        evidence_str = {k: str(v) for k, v in evidence.items()}
        
        # Get MAP for unobserved variables
        unobserved = [v for v in model.nodes() if v not in evidence_str]
        
        if not unobserved:
            return evidence_str
        
        mpe = bp.map_query(variables=unobserved, evidence=evidence_str)
        
        # Combine with evidence
        result = {**evidence_str, **mpe}
        
        return result
    
    except Exception as e:
        logger.error(f"Error computing MPE: {e}")
        return {}


def analyze_variable_influence(
    target: str,
    model: DiscreteBayesianNetwork,
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze how each variable influences the target variable.
    
    Args:
        target: Target variable name
        model: DiscreteBayesianNetwork
        data: Data for computing empirical distributions
    
    Returns:
        DataFrame with influence metrics
    """
    inference = setup_inference(model)
    results = []
    
    # Get marginal probability of target
    marginal = inference.query(variables=[target])
    p_target_base = marginal.values[1]  # Assuming binary with index 1 = positive outcome
    
    for var in model.nodes():
        if var == target:
            continue
        
        # For each value of the variable
        for value in data[var].unique():
            try:
                # Compute conditional probability
                evidence = {var: str(value)}
                result = inference.query(variables=[target], evidence=evidence)
                p_target_given = result.values[1]
                
                # Compute influence
                influence = p_target_given - p_target_base
                
                results.append({
                    "Variable": var,
                    "Value": value,
                    "P(Target=1|Var)": p_target_given,
                    "Influence": influence,
                    "Lift": p_target_given / p_target_base if p_target_base > 0 else np.nan
                })
            
            except Exception as e:
                logger.debug(f"Error analyzing {var}={value}: {e}")
                continue
    
    df = pd.DataFrame(results)
    df = df.sort_values("Influence", ascending=False)
    
    return df


def compare_query_across_ranks(
    evidence: Dict[str, Any],
    models: Dict[str, DiscreteBayesianNetwork]
) -> pd.DataFrame:
    """
    Compare the same query across multiple rank models.
    
    Args:
        evidence: Evidence dictionary
        models: Dictionary mapping rank names to DiscreteBayesianNetworks
    
    Returns:
        DataFrame comparing probabilities across ranks
    """
    results = []
    
    for rank, model in models.items():
        prob = p_win_given(evidence, model)
        results.append({
            "Rank": rank,
            "P(Win=1)": prob,
            "P(Win=0)": 1 - prob if not np.isnan(prob) else np.nan
        })
    
    return pd.DataFrame(results)


def save_query_results(results: pd.DataFrame, filename: str, output_dir):
    """Save query results to CSV."""
    output_path = output_dir / filename
    results.to_csv(output_path, index=False)
    logger.info(f"Saved query results to {output_path}")

