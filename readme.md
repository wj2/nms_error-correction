# Codebase for nonlinear mixed selectivity analysis and simulations

This code underlies all of the figures in Johnston, Palmer, Freedman
(2019) https://doi.org/10.1101/577288. It is all written in Python 3, making
extensive use of the Python scientific computing environment. To generate
the main text figures, run:
```
python mixedselectivity_theory.figures_script.py
```
with [this additional repository](https://github.com/wj2/general-neural) checked
out in the same folder. Further dependencies include:
1. numpy  
2. scipy  
3. scikit-learn
4. matplotlib  

All are available via pip. One of the supplemental figures relies on computation
of the rate-distortion bound via the Blahut-Arimoto algorithm, implemented in
python [here](https://github.com/alonkipnis/BlahutArimoto).