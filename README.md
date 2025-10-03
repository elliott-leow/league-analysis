# League of Legends Bayesian Network Structure Learning

A comprehensive data science project that learns rank-specific Bayesian network structures using GES (Greedy Equivalence Search) on League of Legends Platinum+ match data. This project compares causal graph structures across different skill tiers and enables probabilistic inference about match outcomes.

## 🎯 Project Overview

This project implements a complete pipeline for:
- **Data Preprocessing**: Transforms raw match data into discrete variables representing game states
- **Structure Learning**: Uses GES algorithm to learn CPDAGs (Completed Partially Directed Acyclic Graphs) per rank
- **Parameter Estimation**: Learns Conditional Probability Tables (CPTs) using Bayesian estimation
- **Structural Comparison**: Analyzes differences in causal structures across rank tiers
- **Probabilistic Inference**: Answers queries about win probabilities given game states

## 📁 Project Structure

```
bdsproject/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/                             # Data files
│   ├── matchData.csv                 # Main match dataset
│   ├── match_ids.csv                 # Match IDs with ranks
│   ├── players_8-14-25.csv          # Player information
│   └── match_data.jsonl             # Raw API data (optional)
├── src/                              # Source code
│   ├── __init__.py
│   ├── config.py                     # Configuration and parameters
│   ├── variables.py                  # Variable schema definitions
│   ├── discretization.py             # Discretization utilities
│   ├── preprocessing.py              # Data preprocessing
│   ├── ges.py                        # GES structure learning
│   ├── parameters.py                 # CPT parameter estimation
│   ├── queries.py                    # Probabilistic inference
│   ├── visualize.py                  # Graph visualization
│   ├── compare.py                    # Structural comparison
│   └── cli.py                        # Command-line interface
├── notebooks/                        # Jupyter notebooks
│   ├── 01_explore.ipynb             # Data exploration
│   └── 02_structure_learning.ipynb  # Structure learning demo
├── reports/                          # Generated reports
│   ├── figures/                      # Visualizations
│   └── lol_ges_report.md            # Final report
└── models/                           # Saved models
    ├── ges_*.pkl                     # GES results
    └── bn_*.pkl                      # Bayesian networks
```

## 🔧 Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Graphviz system library for advanced visualizations

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd /home/kano/Documents/bdsproject
```

2. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **(Optional) Install Graphviz system library:**
```bash
# Ubuntu/Debian
sudo apt-get install graphviz graphviz-dev

# macOS
brew install graphviz

# Then install Python bindings
pip install pygraphviz
```

### Verify Installation

```bash
python -m src.cli --help
```

## 🚀 Usage

### Quick Start: Full Pipeline

Run the complete pipeline for all ranks:

```bash
python -m src.cli full
```

This will:
1. Preprocess data for all rank tiers
2. Learn graph structures using GES
3. Estimate CPT parameters
4. Run example queries
5. Compare structures across ranks
6. Generate comprehensive report

### Step-by-Step Usage

#### 1. Preprocess Data

For all ranks:
```bash
python -m src.cli preprocess --rank all
```

For a specific rank:
```bash
python -m src.cli preprocess --rank Diamond
```

With sample size (for testing):
```bash
python -m src.cli preprocess --rank Diamond --sample-size 5000
```

#### 2. Learn Structure with GES

```bash
python -m src.cli learn --rank Diamond
```

Options:
- `--no-constraints`: Disable domain constraints (not recommended)
- `--layout`: Choose layout (`hierarchical` or `spring`)

#### 3. Learn Parameters (CPTs)

```bash
python -m src.cli params --rank Diamond
```

Options:
- `--verbose`: Print detailed CPT information

#### 4. Run Queries

Example queries:
```bash
python -m src.cli query --rank Diamond --example
```

Custom query:
```bash
python -m src.cli query --rank Diamond --evidence "Baron=1,Gold20=high"
```

#### 5. Compare Structures Across Ranks

```bash
python -m src.cli compare --visualize
```

Options:
- `--ranks`: Specify which ranks to compare
- `--visualize`: Generate comparison visualizations

#### 6. Generate Report

```bash
python -m src.cli report
```

### Using Jupyter Notebooks

1. **Start Jupyter:**
```bash
jupyter notebook
```

2. **Open notebooks:**
   - `notebooks/01_explore.ipynb` - Data exploration and quality checks
   - `notebooks/02_structure_learning.ipynb` - Interactive structure learning demo

## 📊 Variables Modeled

The project models the following discrete variables:

| Variable | Description | Values | Temporal Order |
|----------|-------------|--------|----------------|
| **FB** | First Blood | {0, 1} | Early (0) |
| **FT** | First Tower | {0, 1} | Early (0) |
| **Gold10** | Gold difference @10min | {low, neutral, high} | Early (0) |
| **Kills10** | Kill difference @10min | {behind, even, ahead} | Early (0) |
| **Herald** | Rift Herald by 14min | {0, 1} | Early (0) |
| **Gold20** | Gold difference @20min | {low, neutral, high} | Mid (1) |
| **Kills20** | Kill difference @20min | {behind, even, ahead} | Mid (1) |
| **Drakes** | Dragon count by 25min | {0, 1, 2, 3, 4+} | Mid (1) |
| **Towers** | Tower difference @25min | {≤-2, -1_to_1, ≥2} | Mid (1) |
| **Soul** | Dragon soul by 30min | {None, Infernal, Mountain, Ocean, Cloud, Hextech, Chemtech} | Late (2) |
| **Baron** | Baron kills by 30min | {0, 1, 2+} | Late (2) |
| **Inhibs** | Inhibitor difference @25min | {≤-1, 0, ≥1} | Late (2) |
| **Win** | Match outcome | {0, 1} | Outcome (3) |

### Rank Buckets

- **Platinum**: PLATINUM tier
- **Diamond**: DIAMOND tier
- **Master**: MASTER tier
- **Elite**: GRANDMASTER + CHALLENGER tiers (merged due to sample size)

## 🧪 Domain Constraints

The structure learning enforces temporal ordering:
- **Win cannot cause anything** (no edges from Win to other variables)
- **Later events cannot cause earlier events** (e.g., Baron cannot cause First Blood)
- **Temporal layers**: 
  - Early (FB, FT, Gold10, Kills10, Herald) → 
  - Mid (Gold20, Kills20, Drakes, Towers) → 
  - Late (Soul, Baron, Inhibs) → 
  - Outcome (Win)

## 📈 Example Queries

After learning structures, you can ask probabilistic questions:

1. **"What's the win probability if we get Baron and have high gold at 20 minutes?"**
   ```bash
   python -m src.cli query --rank Diamond --evidence "Baron=1,Gold20=high"
   ```

2. **"How much does First Blood increase win chances?"**
   ```bash
   python -m src.cli query --rank Diamond --evidence "FB=1"
   ```

3. **"What if we get Infernal Soul but are behind in gold?"**
   ```bash
   python -m src.cli query --rank Diamond --evidence "Soul=Infernal,Gold20=low"
   ```

## 📝 Output Files

### Generated Figures

- `reports/figures/cpdag_<rank>.png` - CPDAG visualization for each rank
- `reports/figures/cpdag_<rank>.dot` - Graphviz DOT file
- `reports/figures/rank_comparison.png` - Side-by-side comparison
- `reports/figures/edge_frequency.png` - Edge frequency across ranks

### Reports

- `reports/lol_ges_report.md` - Comprehensive analysis report
- `reports/structure_comparison.md` - Detailed structural comparison
- `reports/edge_comparison_table.csv` - Edge presence by rank
- `reports/queries_<rank>.csv` - Query results per rank

### Models

- `models/ges_<rank>.pkl` - Learned CPDAG structures
- `models/bn_<rank>.pkl` - Bayesian networks with CPTs

## 🔬 Technical Details

### Algorithms

- **GES (Greedy Equivalence Search)**: Score-based structure learning algorithm that searches over equivalence classes of DAGs
- **BIC Scoring**: Bayesian Information Criterion for model selection
- **Bayesian Parameter Estimation**: BDeu prior with equivalent sample size of 10

### Libraries

- `causal-learn`: Structure learning (GES implementation)
- `pgmpy`: Bayesian network modeling and inference
- `networkx`: Graph manipulation
- `pandas`, `numpy`: Data processing
- `matplotlib`, `seaborn`: Visualization

## ⚠️ Assumptions and Limitations

### Assumptions

1. **I.I.D. Data**: Matches are assumed independent and identically distributed
2. **Team Perspective**: All features are from blue team's perspective
3. **Discrete Time**: Variables capture game state at specific time points
4. **No Temporal Dynamics**: Each match is treated as a static snapshot

### Limitations

1. **Timeline Approximation**: Without full timeline data, some variables (Gold10, Gold20) are approximated
2. **Discretization**: Continuous variables are binned, potentially losing information
3. **Sample Size**: Higher ranks (GM/Challenger) have fewer samples
4. **Causal Interpretation**: CPDAGs show correlation structure; causal claims require domain expertise
5. **Missing Data**: Dragon soul types are inferred from dragon counts

## 🧰 Configuration

Edit `src/config.py` to customize:

- **Discretization thresholds**: Adjust bin boundaries for gold differences
- **GES parameters**: Change scoring function or search parameters
- **CPT estimation**: Modify prior strength or estimator type
- **Visualization**: Adjust figure size, colors, layout
- **Domain constraints**: Add or remove forbidden edges

Example:
```python
# In src/config.py
DISCRETIZATION_CONFIG = {
    "Gold10": {
        "bins": [-float('inf'), -1500, 1500, float('inf')],  # Wider neutral zone
        "labels": ["low", "neutral", "high"]
    },
    # ...
}
```

## 🐛 Troubleshooting

### Issue: `causal-learn` installation fails
**Solution**: Ensure you have Python 3.10+ and try:
```bash
pip install --upgrade pip setuptools wheel
pip install causal-learn
```

### Issue: `pygraphviz` won't install
**Solution**: Install system Graphviz first:
```bash
sudo apt-get install graphviz graphviz-dev  # Ubuntu
brew install graphviz  # macOS
```

### Issue: Out of memory when loading data
**Solution**: Use sampling:
```bash
python -m src.cli preprocess --rank Diamond --sample-size 10000
```

### Issue: GES takes too long
**Solution**: Reduce sample size or disable some constraints (not recommended):
```bash
python -m src.cli learn --rank Diamond --no-constraints
```

## 📚 References

- **GES Algorithm**: Chickering, D. M. (2002). "Optimal Structure Identification With Greedy Search"
- **Bayesian Networks**: Koller & Friedman (2009). "Probabilistic Graphical Models"
- **causal-learn**: Zheng et al. (2020). "Causal-learn: Causal Discovery in Python"
- **pgmpy**: Ankan & Panda (2015). "pgmpy: Probabilistic Graphical Models using Python"

## 📧 Contact

For questions or issues, please refer to the project documentation or open an issue in the repository.

## 📄 License

This project is for educational and research purposes.

---

**Built with ❤️ by an Elite Data Scientist + Engineer**


# league-analysis
