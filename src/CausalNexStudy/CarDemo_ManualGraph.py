
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

# %% codecell
from causalnex.structure import StructureModel

model: StructureModel = StructureModel()

model.add_weighted_edges_from([
    ('process_type', 'injury_type', 8.343),
    ('uses_op', 'injury_type', 9.43),
    ('injury_type', 'absenteeism_level', 5.4123),

    ('process_type', 'absenteeism_level', 0.0001),
    ('process_type', 'uses_op', 8.9),
    ('uses_op', 'process_type', 1.1)
])
# %% markdown [markdown]
# Now visualize:
# %% codecell
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

# Now visualize it:
viz = plot_structure(
    model,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename_demo = curPath + "demo.png"


viz.draw(filename_demo)
Image(filename_demo)
# %% markdown [markdown]
# Showing with graphviz (my function)
# %% codecell
from src.utils.GraphvizUtil import *


structToGraph(weightedGraph = model)

# %% markdown [markdown]
# Showing the nodes:
# %% codecell
model.nodes
# %% markdown [markdown]
# Showing the unique edges, which takes into account bidirectionality, as we see there is an edge from `process_type` --> `uses_op` and vice versa:
# %% codecell
model.edges
# %% markdown [markdown]
# Seeing the adjacency graph:
# %% codecell
model.adj

# %% markdown [markdown]
# Must remove one edge from the nodes `process_type` --> `uses_op` because the `BayesianNetwork` class must take an a directed **acyclic** graph:
# %% codecell
#from src.utils.VisualUtil import *


modelAcyclic: StructureModel = model.copy()
modelAcyclic.remove_edge(u = 'uses_op', v = 'process_type')

structToGraph(weightedGraph = modelAcyclic)
# %% markdown [markdown]
# Fit the bayesian network. No need to create the `StructureModel` from data because we created it by hand and have already set its edge weights.
# %% codecell
from causalnex.network import BayesianNetwork

bayesNet: BayesianNetwork = BayesianNetwork(modelAcyclic)


# %% markdown [markdown]
# Fitting node states, using the input data:
# %% codecell
import pandas as pd
from pandas.core.frame import DataFrame

from src.utils.DataUtil import *

import collections

# Creating some names for the random variables (nodes) in the graph, to clarify meaning.

# Create named tuple class with names "Names" and "Objects"
RandomVariable = collections.namedtuple("RandomVariable", ["var", "states"])


ProcessType = RandomVariable(var = "ProcessTYpe", states = ['Accel-Pedal',
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

AbsenteeismLevel = RandomVariable(var = "AbsenteeismLevel", states =  ['Absenteeism-00',
                                                                       'Absenteeism-01',
                                                                       'Absenteeism-02',
                                                                       'Absenteeism-03'])


# Make 30 days to represent 1 month
Time = RandomVariable(var = "Time", states = list(map(lambda day : str(day), range(31))))

TrainingLevel = RandomVariable(var = "TrainingLevel", states = ['Training-00',
                                                                'Training-01',
                                                                'Training-02',
                                                                'Training-03'])

ExertionLevel = RandomVariable(var = "ExertionLevel", states = ['Exertion-00',
                                                                'Exertion-01',
                                                                'Exertion-02',
                                                                'Exertion-03'])

ExperienceLevel = RandomVariable(var = "ExperienceLevel", states = ['Experience-00',
                                                                    'Experience-01',
                                                                    'Experience-02',
                                                                    'Experience-03'])

WorkCapacity = RandomVariable(var = "WorkCapacity", states = ['WorkCapacity-00',
                                                              'WorkCapacity-01',
                                                              'WorkCapacity-02',
                                                              'WorkCapacity-03'])

dataValues = {Time.var : Time.states,
            TrainingLevel.var : TrainingLevel.states,
            ExertionLevel.var : ExertionLevel.states,
            ExperienceLevel.var : ExperienceLevel.states,
            WorkCapacity.var : WorkCapacity. states,
            ProcessType.var : ProcessType.states,
            ToolType.var : ToolType.states,
            InjuryType.var : InjuryType.states,
            AbsenteeismLevel.var : AbsenteeismLevel.states}

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
# Making some white-noise combinations between proces, tool, injury, absenteeism out of the states which are NOT for the use cases 4, 5
processUseCaseStates: Set[State] = set(['Engine-Wiring', 'Engine-Mount', 'Oil-Fill'])
processStates: Set[State] = set(ProcessType.states) - processUseCaseStates; processStates

# NOTE filling this up for same reason as absenteeism
toolUseCaseStates: Set[State] = set(['Forklift', 'Power-Gun', 'Front-Right-Door', 'Oil'])
toolStates: Set[State] = set(ToolType.states) - toolUseCaseStates; toolStates
toolStates = ToolType.states if toolStates == set() else toolStates; toolStates

injuryUseCaseStates: Set[State] = set(['Fall-Gtm', 'Contact-Contusion', 'Chemical-Burn', 'Electrical-Shock'])
injuryStates: Set[State] = set(InjuryType.states) - injuryUseCaseStates; injuryStates

#absenteeismUseCaseStates: Set[State] = set(['Engine-Wiring', 'Engine-Mount', 'Oil-Fill'])
#absenteeismStates: Set[State] = set(AbsenteeismLevel.states) - absenteeismUseCaseStates; absenteeismStates
## NOTE: due to the nature of this use case (4, 5) we must have some absenteeism values so we set it to the the original set of values, otherwise cannot have white noise!
absenteeismStates: Set[State] = set(AbsenteeismLevel.states)

whiteNoiseData: DataFrame = makeData(dataValues = {ProcessType.var : list(processStates),
                                                   ToolType.var : list(toolStates),
                                                   InjuryType.var : list(injuryStates),
                                                   AbsenteeismLevel.var : list(absenteeismStates)},
                                     fileName=dataPath + 'whitenoisefills.csv')
whiteNoiseData


# TODO: encapsulate above logic in a makeWhiteNOiseData function (to extract values that should be part of white noise automatically, from the data values that make up the test cases)

# %% markdown [markdown]
# Fit all node states:
# %% codecell
# bayesNet = bayes
bayesNet.fit_node_states(df = data)
bayesNet.node_states

# %% markdown [markdown]
# Fitting the conditional probability distributions
# %% codecell
bayesNet.fit_cpds(data, method="BayesianEstimator", bayes_prior="K2")

# %% markdown [markdown]
# The learned bayesian network still stores the underlying `StructureModel` and the edge values are the same as before - they are just edge **weights** and not edge **probabilities**.
# %% codecell
list(bayesNet.structure.adjacency())

# %% markdown [markdown]
# Showing the graph again for reference
# %% codecell

structToGraph(weightedGraph = modelAcyclic)

# %% markdown [markdown]
# Because `process_type` has no incoming nodes, only outgoing nodes, its conditional distribution is also its *fully* marginal distribution - it is not conditional on any other variable.
# %% codecell
bayesNet.cpds['process_type']
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
graph = structToGraph(weightedGraph = model)
#graphProbs = renderGraphProbabilities(givenGraph = graph, variables = ???)
