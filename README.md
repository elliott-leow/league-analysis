# League of Legends Bayesian Network Analysis

## Overview

This project uses causal discovery methods to understand what drives winning in League of Legends matches across different skill tiers. We analyzed thousands of Platinum through Challenger matches to learn rank-specific Bayesian network structures that reveal how early-game advantages (like First Blood and early gold leads) translate into mid-game objective control (towers, dragons) and ultimately victory. The core question we address is: what are the causal pathways to winning, and how do these pathways differ between lower-skilled and higher-skilled players?

## Methodology

We implemented a complete pipeline using the GES (Greedy Equivalence Search) algorithm to learn causal graph structures from match data. The pipeline preprocesses raw match statistics into 13 discrete game state variables spanning early game (0-14 min), mid game (14-25 min), and late game (25-30+ min), then applies structure learning with temporal constraints to ensure causes precede effects. We enforce domain knowledge through forbidden edges (e.g., winning cannot cause early-game events) and use BIC scoring to penalize spurious correlations. The result is a CPDAG (Completed Partially Directed Acyclic Graph) for each rank tier that represents causal equivalence classes of directed acyclic graphs. We then estimate conditional probability tables using Bayesian parameter estimation and compare structures across ranks to identify strategic differences.

## Key Findings

Our analysis reveals striking differences in strategic patterns across skill tiers. Lower ranks (Platinum) follow a combat-first strategy where kill advantages drive objective control (`Kills20 → Drakes`), while higher ranks (Diamond, Master) employ a structure-first approach where tower control creates kill opportunities (`Towers → Kills20`). Master players demonstrate the most sophisticated understanding with unique patterns like `Baron → Soul` (timing Baron to secure Dragon Soul) and `Gold20 → Kills20` (recognizing that gold advantages translate to combat power through itemization). Interestingly, Elite players (Grandmaster/Challenger) show the simplest graph structure with only 8 edges, suggesting either pattern internalization or strategic efficiency focused on essentials. All ranks agree on three universal win conditions: tower control, inhibitor pressure, and mid-game kill advantages.

## Technical Implementation

The codebase is organized into modular Python components: `preprocessing.py` handles data discretization using domain-informed thresholds (e.g., 1000 gold = item breakpoint), `ges.py` implements structure learning with temporal constraints, `parameters.py` estimates CPTs using BDeu priors, `queries.py` enables probabilistic inference, and `visualize.py` generates graph visualizations. The CLI (`cli.py`) provides a unified interface for running the full pipeline (`python -m src.cli full`) or individual steps. Models are saved as pickled objects (GES results and Bayesian networks) for each rank, and comprehensive reports with graph visualizations are generated in the `reports/` directory. The project uses `causal-learn` for structure learning, `pgmpy` for Bayesian network modeling, and standard scientific Python libraries for data processing.

## Results and Applications

The learned causal structures enable both strategic insights and probabilistic queries. Players looking to improve can extract actionable advice: Platinum players should shift from fighting for kills to taking towers first, Diamond players should use tower advantages to create favorable combat situations, and Master players should time Baron captures to coincide with Soul drake spawns. Analysts can use the models to answer queries like "What's the win probability if we get Baron and high gold at 20 minutes?" or "How much does early Herald control increase our chances?" The rank-specific graphs also validate known strategic concepts (macro play matters more at high ranks) while revealing nuances like the Elite's simplified decision tree. All code, models, and visualizations are available in this repository, with detailed figures showing CPDAGs for each rank in `reports/figures/`.

## Quick Start

**Install dependencies:** `pip install -r requirements.txt`  
**Run full pipeline:** `python -m src.cli full`  
**Run queries:** `python -m src.cli query --rank Diamond --example`  
**View results:** Check `reports/figures/` for graph visualizations and `reports/*.csv` for query outputs.
