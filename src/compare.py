"""
Structural comparison module.

Compares learned graph structures across different ranks.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set
from pathlib import Path
import logging

from .ges import get_edge_set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def edge_set(edges: List[Tuple[str, str, str]]) -> Set[Tuple[str, str]]:
    """
    Convert edge list to set of undirected edges.
    
    Args:
        edges: List of (from, to, edge_type) tuples
    
    Returns:
        Set of (node1, node2) tuples where node1 < node2 alphabetically
    """
    return get_edge_set(edges)


def directed_edge_set(edges: List[Tuple[str, str, str]]) -> Set[Tuple[str, str]]:
    """
    Convert edge list to set of directed edges.
    
    Args:
        edges: List of (from, to, edge_type) tuples
    
    Returns:
        Set of (from, to) tuples for directed edges
    """
    return {(f, t) for f, t, et in edges if et == 'directed'}


def compute_jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
    
    Returns:
        Jaccard similarity (intersection / union)
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def compare_edges(graphs_by_rank: Dict[str, List[Tuple[str, str, str]]]) -> Dict:
    """
    Compare edges across multiple rank graphs.
    
    Args:
        graphs_by_rank: Dictionary mapping rank names to edge lists
    
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing graphs across {len(graphs_by_rank)} ranks")
    
    # Convert to edge sets
    edge_sets = {rank: edge_set(edges) for rank, edges in graphs_by_rank.items()}
    directed_sets = {rank: directed_edge_set(edges) for rank, edges in graphs_by_rank.items()}
    
    # Find common edges (present in all ranks)
    all_edges = set.union(*edge_sets.values()) if edge_sets else set()
    common_edges = set.intersection(*edge_sets.values()) if len(edge_sets) > 1 else all_edges
    
    # Find unique edges per rank
    unique_edges = {}
    for rank, edges in edge_sets.items():
        other_edges = set.union(*[e for r, e in edge_sets.items() if r != rank])
        unique_edges[rank] = edges - other_edges
    
    # Edge frequency (how many ranks have each edge)
    edge_frequency = {}
    for rank, edges in edge_sets.items():
        for edge in edges:
            edge_frequency[edge] = edge_frequency.get(edge, 0) + 1
    
    # Pairwise Jaccard similarities
    ranks = list(graphs_by_rank.keys())
    jaccard_matrix = np.zeros((len(ranks), len(ranks)))
    
    for i, rank1 in enumerate(ranks):
        for j, rank2 in enumerate(ranks):
            jaccard_matrix[i, j] = compute_jaccard_similarity(
                edge_sets[rank1],
                edge_sets[rank2]
            )
    
    # Directed edge agreement
    directed_agreement = {}
    for rank1 in ranks:
        for rank2 in ranks:
            if rank1 < rank2:  # Avoid duplicates
                common_undirected = edge_sets[rank1] & edge_sets[rank2]
                
                # For common edges, check if they have the same direction
                agreement = 0
                total = 0
                
                for node1, node2 in common_undirected:
                    # Check all possible directions
                    dir1_edges = directed_sets[rank1]
                    dir2_edges = directed_sets[rank2]
                    
                    has_12_in_1 = (node1, node2) in dir1_edges
                    has_21_in_1 = (node2, node1) in dir1_edges
                    has_12_in_2 = (node1, node2) in dir2_edges
                    has_21_in_2 = (node2, node1) in dir2_edges
                    
                    # If both have the edge directed the same way, count as agreement
                    if (has_12_in_1 and has_12_in_2) or (has_21_in_1 and has_21_in_2):
                        agreement += 1
                    total += 1
                
                if total > 0:
                    directed_agreement[f"{rank1}_vs_{rank2}"] = agreement / total
    
    results = {
        "n_ranks": len(graphs_by_rank),
        "total_unique_edges": len(all_edges),
        "common_edges": common_edges,
        "n_common_edges": len(common_edges),
        "unique_edges": unique_edges,
        "edge_frequency": edge_frequency,
        "jaccard_matrix": pd.DataFrame(
            jaccard_matrix,
            index=ranks,
            columns=ranks
        ),
        "directed_agreement": directed_agreement
    }
    
    logger.info(f"Found {len(common_edges)} common edges across all ranks")
    logger.info(f"Total unique edges: {len(all_edges)}")
    
    return results


def generate_comparison_summary(comparison: Dict) -> str:
    """
    Generate a markdown summary of the comparison.
    
    Args:
        comparison: Output from compare_edges
    
    Returns:
        Markdown-formatted string
    """
    md = []
    
    md.append("# Structural Comparison Across Ranks\n")
    md.append(f"**Number of ranks compared:** {comparison['n_ranks']}\n")
    md.append(f"**Total unique edges:** {comparison['total_unique_edges']}\n")
    md.append(f"**Common edges (all ranks):** {comparison['n_common_edges']}\n\n")
    
    # Common edges
    if comparison['common_edges']:
        md.append("## Common Edges (Present in All Ranks)\n")
        for edge in sorted(comparison['common_edges']):
            md.append(f"- {edge[0]} -- {edge[1]}\n")
        md.append("\n")
    else:
        md.append("## Common Edges\n")
        md.append("*No edges are common across all ranks.*\n\n")
    
    # Unique edges per rank
    md.append("## Rank-Specific Edges\n")
    for rank in sorted(comparison['unique_edges'].keys()):
        unique = comparison['unique_edges'][rank]
        md.append(f"\n### {rank} ({len(unique)} unique edges)\n")
        if unique:
            for edge in sorted(unique):
                md.append(f"- {edge[0]} -- {edge[1]}\n")
        else:
            md.append("*No unique edges for this rank.*\n")
    
    md.append("\n")
    
    # Edge frequency table
    md.append("## Edge Frequency Table\n")
    md.append("| Edge | Frequency | Ranks |\n")
    md.append("|------|-----------|-------|\n")
    
    sorted_freq = sorted(
        comparison['edge_frequency'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for edge, freq in sorted_freq:
        md.append(f"| {edge[0]} -- {edge[1]} | {freq}/{comparison['n_ranks']} | {freq/comparison['n_ranks']:.1%} |\n")
    
    md.append("\n")
    
    # Jaccard similarity matrix
    md.append("## Jaccard Similarity Matrix\n")
    md.append(comparison['jaccard_matrix'].to_markdown())
    md.append("\n\n")
    
    # Directed edge agreement
    if comparison['directed_agreement']:
        md.append("## Directed Edge Agreement\n")
        md.append("For edges present in multiple ranks, percentage with same direction:\n\n")
        for pair, agreement in sorted(comparison['directed_agreement'].items()):
            md.append(f"- {pair}: {agreement:.1%}\n")
        md.append("\n")
    
    # Interpretation
    md.append("## Interpretation\n")
    avg_jaccard = comparison['jaccard_matrix'].values[np.triu_indices_from(comparison['jaccard_matrix'].values, k=1)].mean()
    
    md.append(f"\n**Average pairwise Jaccard similarity:** {avg_jaccard:.3f}\n\n")
    
    if avg_jaccard > 0.7:
        md.append("The graph structures are **highly similar** across ranks, "
                 "suggesting consistent game dynamics regardless of skill level.\n\n")
    elif avg_jaccard > 0.4:
        md.append("The graph structures show **moderate similarity** across ranks, "
                 "suggesting some common patterns but also rank-specific differences.\n\n")
    else:
        md.append("The graph structures are **quite different** across ranks, "
                 "suggesting that game dynamics vary significantly with skill level.\n\n")
    
    return "".join(md)


def create_edge_comparison_table(graphs_by_rank: Dict[str, List[Tuple[str, str, str]]]) -> pd.DataFrame:
    """
    Create a table showing which edges appear in which ranks.
    
    Args:
        graphs_by_rank: Dictionary mapping rank names to edge lists
    
    Returns:
        DataFrame with edges as rows and ranks as columns
    """
    # Get all unique edges
    all_edges = set()
    for edges in graphs_by_rank.values():
        all_edges.update(edge_set(edges))
    
    # Create table
    data = []
    for edge in sorted(all_edges):
        row = {"Edge": f"{edge[0]} -- {edge[1]}"}
        
        for rank in sorted(graphs_by_rank.keys()):
            rank_edges = edge_set(graphs_by_rank[rank])
            row[rank] = "✓" if edge in rank_edges else ""
        
        # Add total count
        row["Count"] = sum(1 for rank in graphs_by_rank.keys() 
                          if edge in edge_set(graphs_by_rank[rank]))
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values("Count", ascending=False)
    
    return df


def save_comparison_results(
    comparison: Dict,
    output_file: Path
):
    """
    Save comparison results to markdown file.
    
    Args:
        comparison: Comparison results
        output_file: Output file path
    """
    summary = generate_comparison_summary(comparison)
    
    with open(output_file, 'w') as f:
        f.write(summary)
    
    logger.info(f"Saved comparison results to {output_file}")


