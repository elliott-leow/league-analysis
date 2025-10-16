
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pickle
import logging

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.factors.discrete import TabularCPD

from .config import CPT_PARAMS, MODELS_DIR
from .ges import cpdag_to_dag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_bayesian_network(dag_edges: List[Tuple[str, str]], variables: List[str]) -> DiscreteBayesianNetwork:
    logger.info(f"Creating Bayesian Network with {len(dag_edges)} edges")
    
    # create network
    model = DiscreteBayesianNetwork(dag_edges)
    
    # ensure all variables are added
    for var in variables:
        if var not in model.nodes():
            model.add_node(var)
    
    logger.info(f"Created BN with {len(model.nodes())} nodes and {len(model.edges())} edges")
    
    return model


def fit_cpts(
    model: DiscreteBayesianNetwork,
    data: pd.DataFrame,
    estimator_type: str = "BayesianEstimator"
) -> DiscreteBayesianNetwork:
    logger.info(f"Fitting CPTs using {estimator_type}")
    
    try:
        # ensure data only contains variables that are in the model
        model_vars = list(model.nodes())
        data_subset = data[model_vars].copy()
        
        if estimator_type == "MaximumLikelihoodEstimator":
            model.fit(data_subset, estimator=MaximumLikelihoodEstimator)
        elif estimator_type == "BayesianEstimator":
            model.fit(
                data_subset,
                estimator=BayesianEstimator,
                prior_type=CPT_PARAMS.get("prior_type", "BDeu"),
                equivalent_sample_size=CPT_PARAMS.get("equivalent_sample_size", 10)
            )
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
        
        logger.info(f"CPT fitting completed. Created {len(model.get_cpds())} CPDs for {len(model_vars)} nodes")
        
        # verify all nodes have CPDs and create them for isolated nodes
        nodes_with_cpds = {cpd.variable for cpd in model.get_cpds()}
        missing_cpds = set(model_vars) - nodes_with_cpds
        if missing_cpds:
            logger.warning(f"Missing CPDs for nodes: {missing_cpds}. Creating CPDs for isolated nodes...")
            for node in missing_cpds:
                # create marginal CPD for isolated node
                node_data = data_subset[node]
                cardinality = node_data.nunique()
                values = node_data.value_counts(normalize=True).sort_index().values.reshape(-1, 1)
                
                cpd = TabularCPD(
                    variable=node,
                    variable_card=cardinality,
                    values=values
                )
                model.add_cpds(cpd)
                logger.info(f"Created marginal CPD for isolated node '{node}' with cardinality {cardinality}")
        
        # log CPD information
        for cpd in model.get_cpds():
            logger.debug(f"CPT for {cpd.variable}: shape {cpd.cardinality}")
        
    except Exception as e:
        logger.error(f"Error fitting CPTs: {e}")
        raise
    
    return model


def learn_parameters_from_ges(
    ges_result: dict,
    data: pd.DataFrame,
    estimator_type: Optional[str] = None
) -> DiscreteBayesianNetwork:
    if estimator_type is None:
        estimator_type = CPT_PARAMS.get("estimator_type", "BayesianEstimator")
    
    # convert CPDAG to DAG
    dag_edges = cpdag_to_dag(ges_result['edges'], ges_result['variables'])
    
    # create network
    model = create_bayesian_network(dag_edges, ges_result['variables'])
    
    # filter data to only include variables in the model
    # (some may have been removed during structure learning due to zero variance)
    data_filtered = data[ges_result['variables']].copy()
    
    # fit CPTs
    model = fit_cpts(model, data_filtered, estimator_type)
    
    return model


def save_bayesian_network(model: DiscreteBayesianNetwork, rank: str, output_dir: Optional[Path] = None):
    if output_dir is None:
        output_dir = MODELS_DIR
    
    output_file = output_dir / f"bn_{rank}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved Bayesian Network to {output_file}")


def load_bayesian_network(rank: str, models_dir: Optional[Path] = None) -> DiscreteBayesianNetwork:
    if models_dir is None:
        models_dir = MODELS_DIR
    
    input_file = models_dir / f"bn_{rank}.pkl"
    
    if not input_file.exists():
        raise FileNotFoundError(f"Bayesian Network not found: {input_file}")
    
    with open(input_file, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Loaded Bayesian Network from {input_file}")
    return model


def print_cpt_summary(model: DiscreteBayesianNetwork):
    print("\n" + "="*80)
    print("CONDITIONAL PROBABILITY TABLES SUMMARY")
    print("="*80)
    
    for cpd in model.get_cpds():
        print(f"\n{cpd.variable}:")
        print(f"  Parents: {cpd.variables[1:] if len(cpd.variables) > 1 else 'None'}")
        print(f"  Cardinality: {cpd.cardinality}")
        print(f"  Table shape: {cpd.values.shape}")
        
        # print small CPTs in full
        if cpd.values.size <= 50:
            print(f"\n{cpd}")


def get_marginal_probabilities(model: DiscreteBayesianNetwork) -> Dict[str, pd.Series]:
    from pgmpy.inference import VariableElimination
    
    inference = VariableElimination(model)
    marginals = {}
    
    for var in model.nodes():
        result = inference.query(variables=[var])
        marginals[var] = pd.Series(
            result.values,
            index=result.state_names[var]
        )
    
    return marginals


def validate_cpts(model: DiscreteBayesianNetwork) -> Dict[str, any]:
    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    for cpd in model.get_cpds():
        # check if probabilities sum to 1
        prob_sums = cpd.values.sum(axis=0)
        
        if not np.allclose(prob_sums, 1.0, atol=1e-5):
            results["errors"].append(
                f"CPT for {cpd.variable} does not sum to 1: {prob_sums}"
            )
            results["valid"] = False
        
        # check for NaN or infinite values
        if np.any(np.isnan(cpd.values)) or np.any(np.isinf(cpd.values)):
            results["errors"].append(
                f"CPT for {cpd.variable} contains NaN or infinite values"
            )
            results["valid"] = False
        
        # check for zero probabilities (warning)
        if np.any(cpd.values == 0):
            results["warnings"].append(
                f"CPT for {cpd.variable} contains zero probabilities"
            )
    
    return results

