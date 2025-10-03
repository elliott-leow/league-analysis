# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Using the setup script (recommended)
./setup.sh

# Or manually
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python3 test_basic.py
```

You should see:
```
✓ All tests passed! Project is ready to use.
```

### Step 3: Run a Quick Demo

Try preprocessing and learning on a single rank with a small sample:

```bash
# Activate virtual environment (if not already)
source venv/bin/activate

# Quick demo on Diamond rank (uses 5000 samples)
python -m src.cli preprocess --rank Diamond --sample-size 5000
python -m src.cli learn --rank Diamond
python -m src.cli params --rank Diamond
python -m src.cli query --rank Diamond --example
```

### Step 4: Run Full Pipeline

Process all ranks with full data:

```bash
python -m src.cli full
```

This will:
- ✓ Preprocess all ranks
- ✓ Learn structures with GES
- ✓ Estimate CPTs
- ✓ Run example queries
- ✓ Compare structures
- ✓ Generate comprehensive report

**Time estimate**: 20-60 minutes depending on your system

### Step 5: View Results

Check the generated files:

```bash
# View report
cat reports/lol_ges_report.md

# View learned structures
ls reports/figures/cpdag_*.png

# View comparison
cat reports/structure_comparison.md
```

## 📊 Example Commands

### Preprocess specific rank
```bash
python -m src.cli preprocess --rank Platinum
```

### Learn structure with constraints
```bash
python -m src.cli learn --rank Diamond --layout hierarchical
```

### Run custom query
```bash
python -m src.cli query --rank Master --evidence "FB=1,Baron=1,Gold20=high"
```

### Compare structures
```bash
python -m src.cli compare --visualize --ranks Diamond Master Elite
```

## 📓 Jupyter Notebooks

Launch Jupyter and explore interactively:

```bash
jupyter notebook
```

Open:
1. `notebooks/01_explore.ipynb` - Data exploration
2. `notebooks/02_structure_learning.ipynb` - Structure learning demo

## 🐛 Common Issues

### causallearn installation fails
```bash
pip install --upgrade pip setuptools wheel
pip install causal-learn
```

### Out of memory
```bash
# Use smaller sample size
python -m src.cli preprocess --rank Diamond --sample-size 10000
```

### Slow GES
This is normal for large datasets. Consider:
- Using smaller sample size for testing
- Processing one rank at a time
- Running on a machine with more RAM/CPU

## 📚 Next Steps

1. **Explore the data**: Run `notebooks/01_explore.ipynb`
2. **Learn structures**: Run `notebooks/02_structure_learning.ipynb`
3. **Customize**: Edit `src/config.py` to adjust parameters
4. **Compare ranks**: Use `python -m src.cli compare --visualize`
5. **Query models**: Ask probabilistic questions with `--evidence`

## 🎯 Project Goals Checklist

- [x] Clean, reproducible project structure
- [x] GES/FGES structure learning per rank
- [x] CPT estimation with Bayesian methods
- [x] Structural comparison across ranks
- [x] Probabilistic query interface
- [x] Clear visualizations
- [x] Comprehensive documentation
- [x] CLI for easy execution
- [x] Jupyter notebooks for exploration

## 💡 Tips

1. **Start small**: Use `--sample-size 5000` for quick testing
2. **One rank at a time**: Process ranks individually before full pipeline
3. **Check logs**: Look for warnings about sample sizes
4. **Visualize first**: Run structure learning before parameter estimation
5. **Save often**: Models and results are cached in `models/` and `reports/`

---

**Happy learning! 🎮📊**


