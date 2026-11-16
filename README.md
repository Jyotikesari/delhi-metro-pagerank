# PageRank on the Delhi Metro (PRISM 2026)
Code for the PRISM 2026 paper "Which Metro Stations Matter Most? A PageRank Analysis of the Delhi Metro Network" by Jyoti Kesari.

1. Get the dataset:  `git clone https://github.com/shubhamk-co/Delhi-Metro-Analysis.git a`
   (the code reads `a/Data/raw_Delhi_metro.csv`)
2. `pip install numpy scipy networkx matplotlib pandas`
3. `mkdir out && python3 analysis.py`   # graph, PageRank, robustness -> out/results.json
4. `python3 figs.py && python3 posterfigs.py`   # figures
Random seed is fixed (7), so results are reproducible.

Data: shubhamk-co/Delhi-Metro-Analysis (github.com/shubhamk-co/Delhi-Metro-Analysis). Not redistributed here.
