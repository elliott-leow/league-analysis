# Visualization Guide - Graph Rendering with Arrows

## 🎯 Fixed: Arrows Now Visible!

The directed edges in the Bayesian Network graphs now properly display arrows showing causal direction.

---

## 📁 Generated Files

For each rank (Platinum, Diamond, Master, Elite), you'll find **THREE** visualization files:

### 1. DOT File (Source)
- **File**: `cpdag_<rank>.dot`
- **Format**: Graphviz DOT format (text)
- **Purpose**: Source definition of the graph
- **Arrows**: Defined correctly with `->` syntax
- **Can be edited** manually if needed

### 2. Matplotlib PNG (Standard)
- **File**: `cpdag_<rank>.png`
- **Rendered by**: NetworkX + Matplotlib
- **Arrows**: Should now be visible (improved rendering)
- **Size**: Larger file, higher DPI
- **Best for**: Presentations, reports

### 3. Graphviz PNG (Recommended) ⭐
- **File**: `cpdag_<rank>.graphviz.png`
- **Rendered by**: Graphviz `dot` command
- **Arrows**: **Guaranteed to be visible and clear**
- **Quality**: Professional quality
- **Best for**: When arrows must be clearly visible

---

## 🖼️ Which File Should I Use?

### Use `cpdag_<rank>.graphviz.png` if:
- ✅ You need arrows to be crystal clear
- ✅ You're presenting to others
- ✅ You want consistent, professional rendering
- ✅ **This is the recommended version**

### Use `cpdag_<rank>.png` if:
- You prefer matplotlib's styling
- You need the larger DPI version
- Arrows are now visible with improved rendering

### Edit `cpdag_<rank>.dot` if:
- You want to customize the graph manually
- You want to render with different Graphviz settings
- You need to adjust node positions or styling

---

## 🔧 How Arrows Are Rendered

### Directed Edges (→)
```dot
Towers -> Win [arrowsize=1.5];
```
- Shows clear arrow from source to target
- Indicates causal direction
- Black solid lines

### Undirected Edges (--)
```dot
Gold10 -> Herald [dir=none, style=dashed];
```
- Shows no arrowhead (structural equivalence)
- Blue dashed line
- Could go either direction

---

## 📊 Current Graphs (All Ranks)

### Diamond Rank Example (8 edges)
```
Herald → Gold10
Gold20 → Kills20
Towers → Kills20
Towers → Gold20
Baron → Inhibs
Towers → Win       ← Edge to outcome!
Kills20 → Win      ← Edge to outcome!
Inhibs → Win       ← Edge to outcome!
```

All graphs now show:
- ✅ Proper arrow directions
- ✅ Edges pointing to Win
- ✅ Clear visual hierarchy (top to bottom = early to late game)

---

## 🎨 Viewing the Graphs

### Method 1: Open the PNG files directly
```bash
cd reports/figures
# Open with your preferred image viewer
xdg-open cpdag_Diamond.graphviz.png  # Linux
open cpdag_Diamond.graphviz.png      # macOS
```

### Method 2: Render DOT file with custom settings
```bash
# High quality PNG
dot -Tpng cpdag_Diamond.dot -o custom_output.png

# SVG (scalable)
dot -Tsvg cpdag_Diamond.dot -o cpdag_Diamond.svg

# PDF
dot -Tpdf cpdag_Diamond.dot -o cpdag_Diamond.pdf
```

### Method 3: Use online Graphviz viewer
1. Copy contents of `cpdag_Diamond.dot`
2. Visit https://dreampuf.github.io/GraphvizOnline/
3. Paste and view interactively

---

## 📈 What the Arrows Mean

### Direct Causation
- **A → B**: Variable A directly influences Variable B
- Example: `Towers → Win` means tower advantage directly predicts winning

### Temporal Flow
- **Graphs flow top to bottom**: Early game → Mid game → Late game → Win
- **Arrows can't go backwards in time** (enforced by constraints)

### Mutual Information Scores
When edges were added to Win, they were ranked by mutual information:
- **Towers → Win**: MI = 0.556 (strongest predictor!)
- **Kills20 → Win**: MI = 0.450
- **Inhibs → Win**: MI = 0.447

Higher MI = stronger statistical association

---

## 🛠️ Troubleshooting

### Arrows still not visible?
1. **Check you're viewing the `.graphviz.png` version** (not `.png`)
2. If Graphviz failed, install it:
   ```bash
   sudo apt-get install graphviz  # Ubuntu/Debian
   brew install graphviz          # macOS
   ```
3. Regenerate graphs:
   ```bash
   python3 -m src.cli learn --rank all
   ```

### Want larger arrows?
Edit the DOT file and increase `arrowsize`:
```dot
Towers -> Win [arrowsize=2.0];
```
Then regenerate:
```bash
dot -Tpng cpdag_Diamond.dot -o cpdag_Diamond_large_arrows.png
```

---

## 📝 Summary

✅ **Arrows are now properly rendered** in all Bayesian Network graphs  
✅ **Use `.graphviz.png` files** for best results  
✅ **All ranks have edges to Win** showing direct predictors  
✅ **DOT files can be customized** and re-rendered  

**Recommended files**: `cpdag_<rank>.graphviz.png` ⭐

---

Generated: October 3, 2025  
Project: League of Legends Bayesian Network Structure Learning

