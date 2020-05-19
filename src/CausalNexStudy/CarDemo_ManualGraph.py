
# %% codecell
import os
from typing import *


os.getcwd()
# Setting the baseline:
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonProbabilisticGraphicalModels')


curPath: str = os.getcwd() + "/src/CausalNexStudy/"

dataPath: str = os.getcwd() + "/src/_data/"


print("curPath = ", curPath, "\n")
print("dataPath = ", dataPath, "\n")
# %% codecell
import sys
# Making files in utils folder visible here: to import my local print functions for nn.Module objects
sys.path.append(os.getcwd() + "/src/utils/")
# For being able to import files within CausalNex folder
sys.path.append(curPath)
sys.path.append(curPath + 'fonts/')

sys.path

# %% markdown
# Importing
# %% codecell

from causalnex.structure import StructureModel
from causalnex.network import BayesianNetwork

import pandas as pd
from pandas.core.frame import DataFrame

from src.utils.DataUtil import *
from src.utils.GraphvizUtil import *


import collections

# %% markdown
# ## Step 1: Creating the data
# %% codecell

# Creating some names for the random variables (nodes) in the graph, to clarify meaning.

# Create named tuple class with names "Names" and "Objects"
RandomVariable = collections.namedtuple("RandomVariable", ["var", "states"])


ProcessType = RandomVariable(var = "ProcessType", states = ['Accel-Pedal',
                                                            'Door-Mount',
                                                            'Engine-Mount',
                                                            'Engine-Wiring',
                                                            'Oil-Fill',
                                                            'Sun-Roof-Housing'])

ToolType = RandomVariable(var = "ToolType", states = ['Forklift', 'Front-Right-Door', 'Oil', 'Power-Gun'])

InjuryType = RandomVariable(var = "InjuryType", states = ['Chemical-Burn',
                                                          'Contact-Contusion',
                                                          'Electrical-Burn',
                                                          'Electrical-Shock',
                                                          'Fall-Gtm'])

#AbsenteeismLevel = RandomVariable(var = "AbsenteeismLevel", states =  ['Absenteeism-00',
#                                                                       'Absenteeism-01',
#                                                                       'Absenteeism-02',
#                                                                       'Absenteeism-03'])
AbsenteeismLevel = RandomVariable(var = "AbsenteeismLevel", states =  ['Low', 'Medium', 'High'])


# Make 30 days to represent 1 month
Time = RandomVariable(var = "Time", states = list(map(lambda day : str(day), range(1, 31))))

#TrainingLevel = RandomVariable(var = "TrainingLevel", states = ['Training-00',
#                                                                'Training-01',
#                                                                'Training-02',
#                                                                'Training-03'])
TrainingLevel = RandomVariable(var = "TrainingLevel", states = ['Low', 'Medium', 'High'])

#ExertionLevel = RandomVariable(var = "ExertionLevel", states = ['Exertion-00',
#                                                                'Exertion-01',
#                                                                'Exertion-02',
#                                                                'Exertion-03'])
ExertionLevel = RandomVariable(var = "ExertionLevel", states = ['Low', 'Medium', 'High'])

#ExperienceLevel = RandomVariable(var = "ExperienceLevel", states = ['Experience-00',
#                                                                    'Experience-01',
#                                                                    'Experience-02',
#                                                                    'Experience-03'])
ExperienceLevel = RandomVariable(var = "ExperienceLevel", states = ['Low', 'Medium', 'High'])

#WorkCapacity = RandomVariable(var = "WorkCapacity", states = ['WorkCapacity-00',
#                                                              'WorkCapacity-01',
#                                                              'WorkCapacity-02',
#                                                              'WorkCapacity-03'])
WorkCapacity = RandomVariable(var = "WorkCapacity", states = ['Low', 'Medium', 'High'])

dataDict = {Time.var : Time.states,
            TrainingLevel.var : TrainingLevel.states,
            ExertionLevel.var : ExertionLevel.states,
            ExperienceLevel.var : ExperienceLevel.states,
            WorkCapacity.var : WorkCapacity. states,
            ProcessType.var : ProcessType.states,
            ToolType.var : ToolType.states,
            InjuryType.var : InjuryType.states,
            AbsenteeismLevel.var : AbsenteeismLevel.states}

dataDict
# %% codecell
#data: DataFrame = makeWhiteNoiseDataFrame(dataValues = {Time.var : Time.states,
#                                                        TrainingLevel.var : TrainingLevel.states,
#                                                        ExertionLevel.var : ExertionLevel.states,
#                                                        ExperienceLevel.var : ExperienceLevel.states,
#                                                        WorkCapacity.var : WorkCapacity. states,
#                                                        ProcessType.var : ProcessType.states,
#                                                        ToolType.var : ToolType.states,
#                                                        InjuryType.var : InjuryType.states,
#                                                        AbsenteeismLevel.var : AbsenteeismLevel.states},
#                                          dataPath = dataPath + 'fullRawCombData.csv')

# %% codecell
signalDict = {Time.var : Time.states, ProcessType.var : ['Engine-Mount', 'Engine-Wiring', 'Oil-Fill'],
              ToolType.var : ToolType.states,
              InjuryType.var : ['Fall-Gtm', 'Contact-Contusion', 'Chemical-Burn', 'Electrical-Shock'],
              AbsenteeismLevel.var : AbsenteeismLevel.states}


whiteNoiseData: DataFrame = makeWhiteNoiseData(dataDict = dataDict, signalDict = signalDict) #, fileName = dataPath +"whitenoise.csv")

whiteNoiseData

# %% codecell
# Reading in the use case data
# NOTE: reading in every column as string type so the Time variable will come out string
usecaseData: DataFrame = pd.read_csv(dataPath + 'WIKI_USECASES_4_5.csv', delimiter = ',', dtype = str)
usecaseData = usecaseData.dropna()

# Now convert the Time to int:
usecaseData['Time'] = usecaseData['Time'].astype(int)

usecaseData
# %% codecell
# Concatenate the two data frames to create the final full data set
whiteNoiseData['Time'] = whiteNoiseData['Time'].astype(int)

# NOTE: inspected results but ElectricalBurn turns out too high probability (in Injury CPD later on) probabily just because of the frequency of appearances, so the whole 'whitenoisedata' concept cannot be correct. Even using a single state value for  white noise, just using the non-signal values to generate combinations of whtie noise values will STILL bias the results, even if the end ndoe variable (Absenteeism) is still 01,2,3 evenly spread out among that single state value.
#data: DataFrame = pd.concat([usecaseData, whiteNoiseData],
#                                keys=['UsecaseData', 'StubData'],
#                                names=['Type of Data', 'Row ID'])

# To see the data
#data.to_csv(path_or_buf = dataPath + 'WIKI_USECASES_4_5_fulldata.csv' , sep =',')


data = usecaseData



# %% markdown
# ## Step 2: Creating the Network Structure


# %% codecell


carModel: StructureModel = StructureModel()

carModel.add_edges_from([
    (ExertionLevel.var, WorkCapacity.var),
    (ExperienceLevel.var, WorkCapacity.var),
    (TrainingLevel.var, WorkCapacity.var),
    (WorkCapacity.var, AbsenteeismLevel.var),

    (Time.var, WorkCapacity.var),
    (Time.var, AbsenteeismLevel.var),
    (Time.var, ExertionLevel.var),
    (Time.var, ExperienceLevel.var),
    (Time.var, TrainingLevel.var),

    (ProcessType.var, ToolType.var),
    (ToolType.var, InjuryType.var),
    (ProcessType.var, InjuryType.var),
    (ProcessType.var, AbsenteeismLevel.var),
    (InjuryType.var, AbsenteeismLevel.var)
])


structToGraph(weightedGraph = carModel)
# %% markdown [markdown]
# Now visualize:
# %% codecell
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

# Now visualize it:
viz = plot_structure(
    carModel,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)

filename_demo = curPath + "demo.png"


viz.draw(filename_demo)

Image(filename_demo)


# %% markdown
# ## Step 3: Create the Bayesian Model and Fit CPDs
# %% codecell
# Checking the structure is acyclic before passing it to bayesian network:
import networkx as nx

assert nx.is_directed_acyclic_graph(carModel)

# Now fit bayesian model
bayesNet: BayesianNetwork = BayesianNetwork(carModel)

# %% markdown
# Fit node states
# %% codecell
bayesNet.fit_node_states(df = data)
bayesNet.node_states

# %% markdown [markdown]
# Fitting the conditional probability distributions
# %% codecell
bayesNet.fit_cpds(data, method="BayesianEstimator", bayes_prior="K2")


# %% markdown [markdown]
# Because `Time` has no incoming nodes, only outgoing nodes, its conditional distribution is also its *fully* marginal distribution - it is not conditional on any other variable.
# %% codecell
bayesNet.cpds[Time.var]
# %% codecell
bayesNet.cpds[ExertionLevel.var]
# %% codecell
bayesNet.cpds[WorkCapacity.var]
# %% markdown
# $\color{green}{\text{SUCCESS: }}$ The data-biasing has worked! The use cases have been verified. Below we see that:
#
# * (a) Given process-type = `Engine-Mount` and tool-type = `Forklift` the most likely injury-type = `Contact-Contusion` or even `Fall-Gtm` rather than things like `Chemical-Burn`. $\color{red}{\text{Actually, Chemical-Burn turns out high probablility ...? todo}}$ 
# * (b) Given process-type = `Engine-Wiring` and uses-op = `Power-Gun` the most likely injury-type = `Electrical-Shock` than things like `Contact-Contusion`.
# * (e) Given process-type = `Oil-Fill` and uses-op = `Oil`, the most likely injury-type = either `Chemical-Burn` or `Electrical-Shock`. (NOTE: focused on `Chemical-Burn` in data set so that is why the other option does not have high probability)
# %% codecell
bayesNet.cpds[InjuryType.var]
# %% codecell
bayesNet.cpds[AbsenteeismLevel.var]

# %% markdown [markdown]
# But `uses_op` has `process_type` as an incoming node, so its conditional distribution shows the values of `uses_op` conditional on values of `process_type`:
# %% codecell
bayesNet.cpds['uses_op']
# %% markdown [markdown]
# `injury_type` is conditional on two variables, and its table reflects this:
# %% codecell
bayesNet.cpds['injury_type']

# %% markdown [markdown]
# `absenteeism_level` is only **directly** conditional on two variables, the `injury_type` and `process_type`, which is visible in its conditional probability distribution table below:
# %% codecell
bayesNet.cpds['absenteeism_level']

# %% markdown
# Showing the final rendered graph with the conditional probability distributions alongside the nodes:
# %% codecell
#Image(filename = curPath + 'modelWithCPDs.png')
graph = structToGraph(weightedGraph = carModel)
#graphProbs = renderGraphProbabilities(givenGraph = graph, variables = ???)
