"""
Visualization module for Bayesian networks.

Creates visualizations of learned graph structures and comparisons across ranks.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
import logging

from .config import VIZ_PARAMS, FIGURES_DIR
from .variables import get_temporal_order

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_networkx_graph(edges: List[Tuple[str, str, str]]) -> nx.DiGraph:
    """
    Create a NetworkX directed graph from edge list.
    
    Args:
        edges: List of (from, to, edge_type) tuples
    
    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()
    
    for from_var, to_var, edge_type in edges:
        if edge_type == 'directed':
            G.add_edge(from_var, to_var)
        elif edge_type == 'undirected':
            # Add both directions for undirected (will style differently)
            G.add_edge(from_var, to_var, bidirectional=True)
            G.add_edge(to_var, from_var, bidirectional=True)
    
    return G


def get_hierarchical_layout(G: nx.DiGraph, variables: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Create a hierarchical layout based on temporal ordering.
    
    Args:
        G: NetworkX graph
        variables: List of variable names
    
    Returns:
        Dictionary mapping nodes to (x, y) positions
    """
    # Group variables by temporal order
    temporal_groups = {}
    for var in G.nodes():
        order = get_temporal_order(var)
        if order not in temporal_groups:
            temporal_groups[order] = []
        temporal_groups[order].append(var)
    
    pos = {}
    y_spacing = 3.0
    
    for order in sorted(temporal_groups.keys()):
        nodes = sorted(temporal_groups[order])  # Sort alphabetically
        n = len(nodes)
        x_positions = np.linspace(-n/2, n/2, n)
        
        for i, node in enumerate(nodes):
            pos[node] = (x_positions[i], -order * y_spacing)
    
    return pos


def plot_cpdag(
    edges: List[Tuple[str, str, str]],
    variables: List[str],
    title: str = "CPDAG",
    output_file: Optional[Path] = None,
    layout: str = "hierarchical"
) -> plt.Figure:
    """
    Plot a CPDAG (Completed Partially Directed Acyclic Graph).
    
    Args:
        edges: List of edges
        variables: List of variable names
        title: Plot title
        output_file: Optional output file path
        layout: Layout algorithm ('hierarchical' or 'spring')
    
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting CPDAG: {title}")
    
    # Create graph
    G = nx.DiGraph()
    G.add_nodes_from(variables)
    
    directed_edges = []
    undirected_pairs = set()
    
    for from_var, to_var, edge_type in edges:
        if edge_type == 'directed':
            directed_edges.append((from_var, to_var))
        elif edge_type == 'undirected':
            # Store as sorted tuple to avoid duplicates
            pair = tuple(sorted([from_var, to_var]))
            undirected_pairs.add(pair)
    
    # Add directed edges
    G.add_edges_from(directed_edges)
    
    # Create figure
    fig, ax = plt.subplots(
        figsize=VIZ_PARAMS["figure_size"],
        dpi=VIZ_PARAMS["dpi"]
    )
    
    # Get layout
    if layout == "hierarchical":
        pos = get_hierarchical_layout(G, variables)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=VIZ_PARAMS["node_size"],
        ax=ax
    )
    
    # Draw directed edges with better arrow visibility
    nx.draw_networkx_edges(
        G, pos,
        edgelist=directed_edges,
        edge_color='black',
        width=VIZ_PARAMS["edge_width"],
        arrows=True,
        arrowsize=30,  # Increased from 20
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1',  # Slight curve to show direction better
        node_size=VIZ_PARAMS["node_size"],  # Pass node size for proper arrow positioning
        min_source_margin=15,
        min_target_margin=15,
        ax=ax
    )
    
    # Draw undirected edges (no arrows)
    for node1, node2 in undirected_pairs:
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        ax.plot([x1, x2], [y1, y2], 'b--', linewidth=VIZ_PARAMS["edge_width"], alpha=0.6)
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=VIZ_PARAMS["font_size"],
        font_weight='bold',
        ax=ax
    )
    
    # Add legend
    directed_patch = mpatches.Patch(color='black', label='Directed edge (→)')
    undirected_patch = mpatches.Patch(color='blue', label='Undirected edge (--)')
    ax.legend(handles=[directed_patch, undirected_patch], loc='upper right')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        fig.savefig(output_file, dpi=VIZ_PARAMS["dpi"], bbox_inches='tight')
        logger.info(f"Saved figure to {output_file}")
    
    return fig


def plot_rank_comparison(
    rank_edges: Dict[str, List[Tuple[str, str, str]]],
    variables: List[str],
    output_file: Optional[Path] = None
) -> plt.Figure:
    """
    Plot side-by-side comparison of CPDAGs across ranks.
    
    Args:
        rank_edges: Dictionary mapping rank names to edge lists
        variables: List of variable names
        output_file: Optional output file path
    
    Returns:
        Matplotlib figure
    """
    logger.info("Creating rank comparison plot")
    
    n_ranks = len(rank_edges)
    ncols = min(2, n_ranks)
    nrows = (n_ranks + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(VIZ_PARAMS["figure_size"][0] * ncols / 2, 
                 VIZ_PARAMS["figure_size"][1] * nrows / 2),
        dpi=VIZ_PARAMS["dpi"]
    )
    
    if n_ranks == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (rank, edges) in enumerate(rank_edges.items()):
        ax = axes[idx]
        
        # Create graph
        G = nx.DiGraph()
        G.add_nodes_from(variables)
        
        directed_edges = [(f, t) for f, t, et in edges if et == 'directed']
        G.add_edges_from(directed_edges)
        
        # Layout
        pos = get_hierarchical_layout(G, variables)
        
        # Draw
        nx.draw(
            G, pos,
            node_color='lightblue',
            node_size=2000,
            with_labels=True,
            font_size=8,
            font_weight='bold',
            arrows=True,
            arrowsize=15,
            edge_color='black',
            width=1.5,
            ax=ax
        )
        
        ax.set_title(f"{rank} (n={len(edges)} edges)", fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(n_ranks, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Bayesian Network Structures by Rank", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file, dpi=VIZ_PARAMS["dpi"], bbox_inches='tight')
        logger.info(f"Saved comparison plot to {output_file}")
    
    return fig


def plot_edge_frequency(
    edge_freq: Dict[Tuple[str, str], int],
    n_ranks: int,
    output_file: Optional[Path] = None
) -> plt.Figure:
    """
    Plot edge frequency across ranks.
    
    Args:
        edge_freq: Dictionary mapping edges to frequency counts
        n_ranks: Total number of ranks
        output_file: Optional output file path
    
    Returns:
        Matplotlib figure
    """
    logger.info("Creating edge frequency plot")
    
    # Sort by frequency
    sorted_edges = sorted(edge_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 20
    top_edges = sorted_edges[:20]
    
    if not top_edges:
        logger.warning("No edges to plot")
        return None
    
    edge_labels = [f"{e[0]} → {e[1]}" for e, _ in top_edges]
    frequencies = [freq / n_ranks for _, freq in top_edges]
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=VIZ_PARAMS["dpi"])
    
    colors = ['green' if f == 1.0 else 'orange' if f >= 0.5 else 'red' for f in frequencies]
    
    ax.barh(edge_labels, frequencies, color=colors, alpha=0.7)
    ax.set_xlabel('Frequency (proportion of ranks)', fontsize=12)
    ax.set_ylabel('Edge', fontsize=12)
    ax.set_title('Edge Frequency Across Ranks', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    green_patch = mpatches.Patch(color='green', label='Common (all ranks)')
    orange_patch = mpatches.Patch(color='orange', label='Frequent (≥50% ranks)')
    red_patch = mpatches.Patch(color='red', label='Rare (<50% ranks)')
    ax.legend(handles=[green_patch, orange_patch, red_patch], loc='lower right')
    
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file, dpi=VIZ_PARAMS["dpi"], bbox_inches='tight')
        logger.info(f"Saved edge frequency plot to {output_file}")
    
    return fig


def plot_variable_distributions(
    data: pd.DataFrame,
    rank: str,
    output_file: Optional[Path] = None
) -> plt.Figure:
    """
    Plot distributions of discretized variables.
    
    Args:
        data: Discretized data
        rank: Rank name
        output_file: Optional output file path
    
    Returns:
        Matplotlib figure
    """
    logger.info(f"Creating variable distribution plot for {rank}")
    
    from .variables import get_all_variables
    variables = [v for v in get_all_variables() if v in data.columns]
    
    n_vars = len(variables)
    ncols = 3
    nrows = (n_vars + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3), dpi=100)
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    for idx, var in enumerate(variables):
        ax = axes[idx]
        
        value_counts = data[var].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        
        value_counts.plot(kind='bar', ax=ax, color=colors, alpha=0.7)
        ax.set_title(var, fontsize=12, fontweight='bold')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Variable Distributions - {rank}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file, dpi=100, bbox_inches='tight')
        logger.info(f"Saved distribution plot to {output_file}")
    
    return fig


def save_graph_as_dot(
    edges: List[Tuple[str, str, str]],
    variables: List[str],
    output_file: Path,
    title: str = "CPDAG",
    render_png: bool = True
):
    """
    Save graph as DOT file for Graphviz rendering.
    
    Args:
        edges: List of edges
        variables: List of variable names
        output_file: Output file path
        title: Graph title
        render_png: Whether to also render as PNG using Graphviz
    """
    with open(output_file, 'w') as f:
        f.write(f'digraph "{title}" {{\n')
        f.write('  rankdir=TB;\n')
        f.write('  node [shape=box, style=filled, fillcolor=lightblue, fontsize=12];\n')
        f.write('  edge [penwidth=2.0];\n\n')
        
        # Add nodes
        for var in variables:
            f.write(f'  {var};\n')
        
        f.write('\n')
        
        # Add edges
        for from_var, to_var, edge_type in edges:
            if edge_type == 'directed':
                f.write(f'  {from_var} -> {to_var} [arrowsize=1.5];\n')
            elif edge_type == 'undirected':
                f.write(f'  {from_var} -> {to_var} [dir=none, style=dashed];\n')
        
        f.write('}\n')
    
    logger.info(f"Saved DOT file to {output_file}")
    
    # Try to render with Graphviz if available and requested
    if render_png:
        try:
            import subprocess
            png_file = output_file.with_suffix('.graphviz.png')
            result = subprocess.run(
                ['dot', '-Tpng', str(output_file), '-o', str(png_file)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info(f"Rendered Graphviz PNG to {png_file}")
            else:
                logger.warning(f"Graphviz rendering failed: {result.stderr}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Graphviz not available or timed out: {e}")
        except Exception as e:
            logger.warning(f"Could not render with Graphviz: {e}")

