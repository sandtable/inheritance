# Inheritance simulation

An agent-based model simulating the UK population, focusing on the effects of inherited wealth.

## Requirements

* Python 3.6
* Packages listed in `requirements.txt`

## Running the simulation

Before your first run, initialise the data:

```
make data
```

or 

```
python data.py
```

To run the simulation:

```
make run
```

or 

```
python adapter.py model.yaml
```

## Outputs

Running the simulation produces an `output.h5` HDF5 file, containing the following tables:

* `data`: top-level data at each tick (year) of the simulation
* `by_age`: data subdivided by the age of the agents, at each tick
* `history`: full history of each agent through the simulation
* `skill_dist`: number of agents at each skill level at each tick
* `tree`: full list of family links between pairs of agents
* `trusts`: data on each inheritance trust at each tick

A function is provided for plotting the life history of an individual agent:

```
import pandas as pd
import sinks

data = pd.HDFStore('output.h5')

sinks.plot_history(data, 123)
```

## Input parameters

Parameters of the model are set in `model.yaml`. Under `control_parameters` you can set the number of `ticks` and number of `agents`. The available `model_parameters` are:

* `inheritance`: type of inheritance, "direct" or "trust"
* `trust_first_generation`: first generation to benefit from trusts (1 = children)
* `trust_last_generation`: last generation to benefit from trusts (1 = children)
* `assortative_mating`: strength of assortative mating (0 = none, 1 = very strong)
* `savings_rate`: fraction of income that gets saved
* `interest_rate`: interest rate for wealth and trusts

## Files

* `README.md`: this file
* `requirements.txt`: python packages required
* `raw_data/*/*.xls`: raw data from the ONS and census
* `data.py`: script for processing raw data into numpy arrays
* `model.yaml`: model specification
* `inheritance.py`: core model logic
* `adapter.py`: wrapper for `inheritance.py` that loads in the data and parameters
* `scenarios.py`: initialises agent population
* `sinks.py`: processes output data


## Resources

PyData London 2017

* Slides: https://www.slideshare.net/Sandtable/forecasting-social-inequality-using-agentbased-modelling
* Video: https://www.youtube.com/watch?v=RglNX4c_dfc