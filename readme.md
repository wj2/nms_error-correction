## Nonlinear mixed selectivity analysis and simulations

This code relies on code from
[this additional repository](https://github.com/wj2/general-neural). It is all
written in Python 3, making
extensive use of the Python scientific computing environment. Further
dependencies include:
1. numpy  
2. scipy  
3. scikit-learn
4. matplotlib  

All are available via pip. With any questions on how to navigate this code or
about the manuscript, please
feel free to contact [me](https://wj2.github.io/).

### Generating figures from Johnston, Palmer, Freedman (2019)
This code underlies all of the figures in Johnston, Palmer, Freedman
(2019) https://doi.org/10.1101/577288. To generate
the main text figures, run:
```
python mixedselectivity_theory/figures_script.py
```
One of the supplemental figures relies on computation
of the rate-distortion bound via the Blahut-Arimoto algorithm, implemented in
python [here](https://github.com/alonkipnis/BlahutArimoto).

