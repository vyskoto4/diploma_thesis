# Inventory Optimization Based on Demand Prediction 

This repo contains the source code and text of my diploma thesis.
I also included some example figures and results of experiments.

Very brief introduction of the directory structure:

    .
    │   ├── source code             
    │   ├── python              # All python source code
    │   │   ├── forecasting     # Module dedicated to demand prediction
    │   │   ├── inventory       # Module dedicated to inventory optimization
    │   ├── tex                 # Latex files used to produce the thesis pdf
    │   └── unit                # Unit tests
    └── text.pdf                # Thesis text
    
The easiest way to replicate the python development environment is using Conda:
```
conda create -n new environment --file ./source\ code/python/requirements.txt
```

