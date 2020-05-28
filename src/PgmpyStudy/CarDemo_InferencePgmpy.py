# %% markdown [markdown]
# # Car Bayesian Network
# Creating bayesian network to model use cases in [https://synergo.atlassian.net/wiki/spaces/CLNTMMC/pages/1812529153/RFP+-+Extra+use+cases+-+Appendix+A](https://synergo.atlassian.net/wiki/spaces/CLNTMMC/pages/1812529153/RFP+-+Extra+use+cases+-+Appendix+A).

# %% markdown [markdown]
# Doing path-setting:
# %% codecell
import os
import sys


os.getcwd()
# Setting the baseline:
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonProbabilisticGraphicalModels')


curPath: str = os.getcwd() + "/src/PgmpyStudy/"

dataPath: str = os.getcwd() + "/src/_data/"
imagePath: str = curPath + 'images/'

print("curPath = ", curPath, "\n")
print("dataPath = ", dataPath, "\n")
print('imagePath = ', imagePath, "\n")


# Making files in utils folder visible here: to import my local print functions for nn.Module objects
sys.path.append(os.getcwd() + "/src/utils/")
# For being able to import files within PgmpyStudy folder
sys.path.append(curPath)

#sys.path.remove('/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP/src/utils/')
#sys.path.remove('/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP/src/PgmpyStudy/')

sys.path

# %% markdown [markdown]
# Science imports:
# %% codecell
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import JointProbabilityDistribution
from pgmpy.factors.discrete.DiscreteFactor import DiscreteFactor
from pgmpy.independencies import Independencies
from pgmpy.independencies.Independencies import IndependenceAssertion


from operator import mul
from functools import reduce

import itertools
import collections

from src.utils.GraphvizUtil import *
from src.utils.NetworkUtil import *
from src.utils.DataUtil import *
from src.utils.GenericUtil import *

from typing import *

# My type alias for clarity
from src.utils.TypeAliases import *

import pandas as pd
from pandas.core.frame import DataFrame


# %% markdown [markdown]
# ## Step 1: Creating / Loading Data
# %% codecell

# Create named tuple class with names "Names" and "Objects"
#RandomVariable = collections.namedtuple("RandomVariable", ["var", "states"]) # already in type aliases


Process = RandomVariable(var ="Process", states = {'Accel-Pedal',
                                                    'Door-Mount',
                                                    'Engine-Mount',
                                                    'Engine-Wiring',
                                                    'Oil-Fill',
                                                    'Sun-Roof-Housing'})

Tool = RandomVariable(var ="Tool", states = {'Forklift', 'Front-Right-Door', 'Oil', 'Power-Gun'})

Injury = RandomVariable(var ="Injury", states = {'Chemical-Burn',
                                                  'Contact-Contusion',
                                                  'Electrical-Burn',
                                                  'Electrical-Shock',
                                                  'Fall-Gtm'})

#Absenteeism = RandomVariable(var = "Absenteeism", states =  ['Absenteeism-00',
#                                                                       'Absenteeism-01',
#                                                                       'Absenteeism-02',
#                                                                       'Absenteeism-03'])
Absenteeism = RandomVariable(var ="Absenteeism", states =  {'Low', 'Medium', 'High'})


# Make 30 days to represent 1 month
Time = RandomVariable(var = "Time", states = set(range(1, 31))) # list(map(lambda day : str(day), range(1, 31))))

#Training = RandomVariable(var = "Training", states = ['Training-00',
#                                                                'Training-01',
#                                                                'Training-02',
#                                                                'Training-03'])
Training = RandomVariable(var ="Training", states = {'Low', 'Medium', 'High'})

#Exertion = RandomVariable(var = "Exertion", states = ['Exertion-00',
#                                                                'Exertion-01',
#                                                                'Exertion-02',
#                                                                'Exertion-03'])
Exertion = RandomVariable(var ="Exertion", states = {'Low', 'Medium', 'High'})

#Experience = RandomVariable(var = "Experience", states = ['Experience-00',
#                                                                    'Experience-01',
#                                                                    'Experience-02',
#                                                                    'Experience-03'])
Experience = RandomVariable(var ="Experience", states = {'Low', 'Medium', 'High'})

#WorkCapacity = RandomVariable(var = "WorkCapacity", states = ['WorkCapacity-00',
#                                                              'WorkCapacity-01',
#                                                              'WorkCapacity-02',
#                                                              'WorkCapacity-03'])
WorkCapacity = RandomVariable(var = "WorkCapacity", states = {'Low', 'Medium', 'High'})

dataDict = {Time.var : Time.states,
            Training.var : Training.states,
            Exertion.var : Exertion.states,
            Experience.var : Experience.states,
            WorkCapacity.var : WorkCapacity. states,
            Process.var : Process.states,
            Tool.var : Tool.states,
            Injury.var : Injury.states,
            Absenteeism.var : Absenteeism.states}

# %% codecell
# Reading in the use case data
# NOTE: reading in every column as string type so the Time variable will come out string
usecaseData: DataFrame = pd.read_csv(dataPath + 'WIKI_USECASES_4_5.csv', delimiter = ',', dtype = str)
usecaseData = cleanData(usecaseData)

# Now convert the Time to int:
usecaseData[Time.var] = usecaseData[Time.var].astype(int)

data = usecaseData
# TODO: Option to later concat with white noise data (like in CarDemo Manual from CausalnexStudy)
data


# %% markdown [markdown]
# ## Step 2: Create Network Structure

# %% codecell

carModel: BayesianModel = BayesianModel([
    (Exertion.var, WorkCapacity.var),
    (Experience.var, WorkCapacity.var),
    (Training.var, WorkCapacity.var),
    (WorkCapacity.var, Absenteeism.var),

    (Time.var, WorkCapacity.var),
    (Time.var, Absenteeism.var),
    (Time.var, Exertion.var),
    (Time.var, Experience.var),
    (Time.var, Training.var),

    (Process.var, Tool.var),
    (Tool.var, Injury.var),
    (Process.var, Injury.var),
    (Process.var, Absenteeism.var),
    (Injury.var, Absenteeism.var)
])


drawGraph(model = carModel)

# %% markdown [markdown]
# ## Step 3: Estimate CPDs
# %% codecell
from pgmpy.estimators import BayesianEstimator

#est: BayesianEstimator = BayesianEstimator(model = carModel, data = data)

assert carModel.get_cpds() == [], "Check cpds are empty beforehand"

carModel.fit(data, estimator = BayesianEstimator,
             prior_type = "BDeu",
             equivalent_sample_size = 10)


# %% codecell
conditionalDistDf(carModel, query= Time)
# %% codecell
conditionalDistDf(carModel, query= Process)
# %% codecell
conditionalDistDf(carModel, query= Tool)
# %% codecell
conditionalDistDf(carModel, query= Experience)
# %% codecell
conditionalDistDf(carModel, query= WorkCapacity)
# %% codecell
conditionalDistDf(carModel, query= Injury)

# %% codecell
conditionalDistDf(carModel, query= Absenteeism)





# %% markdown [markdown]
# ## Step 4: Inference in Bayesian Car Model
#
# Now let us verify active trails or independencies, for each kind of chain (causal, evidential, common cause, and common evidence) that can be found along the paths of the car model graph
#
# ### 1/ Causal Reasoning in the Car Model
# For a causal model $A \rightarrow B \rightarrow C$, there are two cases:
#   * **Marginal Dependence:** ($B$ unknown): When $B$ is unknown / unobserved, there is an active trail between $A$ and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa). We can say $P(A) \ne P(A \; | \; C)$
#   * **Conditional Independence:** ($B$ fixed): When $B$ is fixed, there is NO active trail between $A$ and $C$, so they are independent, which means the probability of $A$ won't influence probability of $C$ (and vice versa). We can say $P(A) = P(A \; | \; C)$


# %% codecell
drawGraph(carModel)
# %% markdown [markdown]
# #### Testing conditional independence:
# $$
# \color{DodgerBlue}{\text{WorkCapacity (observed)}: \;\;\;\;\;\;\;  \text{Experience} \; \bot \; \text{Absenteeism} \; | \; \text{WorkCapacity}}
# $$

# Given that **WorkCapacity**'s state is observed, we can make the following equivalent statements:
# * there is NO active trail between **Experience** and **Absenteeism**.
# * **Experience** and **Absenteeism** are locally independent.
# * the probability of **Experience** won't influence probability of **Absenteeism** (and vice versa).
#

# %% codecell
elim: VariableElimination = VariableElimination(model = carModel)

# %% markdown [markdown]
# **Testing Conditional Independence:** Using Active Trails Methods
# %% codecell
assert carModel.is_active_trail(start = Experience.var, end = Absenteeism.var, observed = None)

assert carModel.is_active_trail(start = Experience.var, end = Absenteeism.var, observed = [WorkCapacity.var]), "Check: still need to condition on extra variable for this not to be an active trail"

# Finding out which extra variable to condition on:
assert observedVars(carModel, start= Experience, end= Absenteeism) == [{Time.var, WorkCapacity.var}], "Check: all list of extra variables to condition on to nullify active trail between Experience and Absenteeism"

# Check trail is nullified
assert not carModel.is_active_trail(start = Experience.var, end = Absenteeism.var, observed =[WorkCapacity.var] + [Time.var]), "Check: active trail between Experience and Absenteeism is nullified with the extra variable observed"

# See, there is no active trail from Experience to Absenteeism when observing WorkCapacity and time.
showActiveTrails(carModel, variables = [Experience, Absenteeism], observed = [WorkCapacity, Time])

# %% markdown [markdown]
# **Testing Conditional Independence:** Using Probabilities
# %% codecell
OBS_STATE_WORKCAPACITY: State = 'Low'
OBS_STATE_TIME: int = 23

backdoorStates: Dict[Name, Set[State]] = {WorkCapacity.var : {OBS_STATE_WORKCAPACITY}, Time.var : {OBS_STATE_TIME}}


EWA: DiscreteFactor = elim.query(variables = [Absenteeism.var], evidence = backdoorStates)

EWA_1: DiscreteFactor = elim.query(variables = [Absenteeism.var], evidence = backdoorStates |plus| {Experience.var : {'High'}})

EWA_2: DiscreteFactor = elim.query(variables = [Absenteeism.var], evidence = backdoorStates |plus| {Experience.var : {'Medium'}})

EWA_3: DiscreteFactor = elim.query(variables = [Absenteeism.var], evidence = backdoorStates |plus| {Experience.var : {'Low'}})

print(EWA)
# %% markdown
# The probabilities above are stated formulaically as follows:
#
# $$
# \begin{array}{ll}
# P(\text{Absenteeism} = \text{High} \; | \; \Big\{  \text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23  \Big\}) \\
# = P(\text{Absenteeism} = \text{High} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{Low})  \\
# = P(\text{Absenteeism} = \text{High} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{Medium}) \\
# = P(\text{Absenteeism} = \text{High} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{High}) \\
# = 0.4989
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{Absenteeism} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\}) \\
# = P(\text{Absenteeism} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{Low})  \\
# = P(\text{Absenteeism} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{Medium}) \\
# = P(\text{Absenteeism} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{High}) \\
# = 0.3994
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{Absenteeism} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\}) \\
# = P(\text{Absenteeism} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{Low})  \\
# = P(\text{Absenteeism} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{Medium}) \\
# = P(\text{Absenteeism} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{High}) \\
# = 0.1017
# \end{array}
# $$
#
# Since all the above stated probabilities are equal for each state of `Absenteeism` = `Low`, `Medium`, `High`, we can assert that the random variables `Experience` and `Absenteeism` are independent of each other, when observing `WorkCapacity` state (and also observing the state of `Time` to adjust for backdoors). Arbitrarily choosing the states `backdoorStates` = `{WorkCapacity = Low, Time = 23}`, we can write:
# $$
# P(\text{Absenteeism} \; | \; \{\texttt{backdoorStates} \}) = P(\text{Absenteeism} \; | \; \{ \texttt{backdoorStates} \} \; \cap \; \text{Experience})
# $$
# %% codecell
assert allEqual(EWA.values, EWA_1.values, EWA_2.values, EWA_3.values), "Check: the random variables Experience and Absenteeism are independent, when intermediary node WorkCapacity is observed (while accounting for backdoors)"


dfEWA: DataFrame = eliminateSlice(carModel, query = Absenteeism, evidence = {WorkCapacity.var :['Low'], Time.var :[23], Experience.var : ['High', 'Low', 'Medium']})


dfEWA






# %% markdown [markdown]
# #### Testing marginal dependence:
# $$
# \color{Green}{\text{WorkCapacity (unobserved)}: \;\;\;\;\;\;\;  \text{Experience} \longrightarrow \text{WorkCapacity} \longrightarrow \text{Absenteeism}}
# $$
# Given that **WorkCapacity**'s state is unobserved, we can make the following equivalent statements:
# * there IS active trail between **Experience** and **Absenteeism**.
# * **Experience** and **Absenteeism** are dependent.
# * the probability of **Experience** influences probability of **Absenteeism** (and vice versa).
#


# %% markdown [markdown]
# **Testing Marginal Dependence:** Using Active Trails Methods
# %% codecell
assert carModel.is_active_trail(start = Experience.var, end = Absenteeism.var, observed = None)


# See, there is active trail from Experience to Absenteeism when not observing WorkCapacity variable
showActiveTrails(carModel, variables = [Experience, Absenteeism], observed = None)

# %% markdown [markdown]
# **Testing Marginal Dependence:** Using Probabilities
# %% codecell
OBS_STATE_TIME: int = 23

# TODO left off here: must make backdoor states type compatible with |plus| but also with elim.query() arguments
backdoorStates: Dict[Name, Set[State]] = {Time.var : {OBS_STATE_TIME}}

EA: DiscreteFactor = elim.query(variables = [Absenteeism.var], evidence = backdoorStates)
print(EA)
# %% codecell
EA_1: DiscreteFactor = elim.query(variables = [Absenteeism.var], evidence = backdoorStates |plus| {Experience.var : {'High'}})
print(EA_1)
# %% codecell
EA_2: DiscreteFactor = elim.query(variables = [Absenteeism.var], evidence = backdoorStates |plus| {Experience.var : {'Medium'}})
print(EA_2)
# %% codecell
EA_3: DiscreteFactor = elim.query(variables = [Absenteeism.var], evidence = backdoorStates |plus| {Experience.var : {'Low'}})
print(EA_3)
# %% markdown
# The probabilities above are stated formulaically as follows:
# $$
# \begin{array}{ll}
# P(\text{Absenteeism} = \text{High} \; | \; \Big\{ \text{Time} = 23  \Big\}) = 0.4965 \\
# \ne P(\text{Absenteeism} = \text{High} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{Low}) = 0.3885  \\
# \ne P(\text{Absenteeism} = \text{High} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{Medium}) = 0.3885 \\
# \ne P(\text{Absenteeism} = \text{High} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{High}) = 0.4973
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{Absenteeism} = \text{Low} \; | \; \Big\{ \text{Time} = 23  \Big\}) = 0.4965 \\
# \ne P(\text{Absenteeism} = \text{Low} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{Low}) = 0.3553 \\
# \ne P(\text{Absenteeism} = \text{Low} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{Medium}) = 0.3553 \\
# \ne P(\text{Absenteeism} = \text{Low} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{High}) = 0.3987
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{Absenteeism} = \text{Medium} \; | \; \Big\{ \text{Time} = 23  \Big\}) = 0.4965 \\
# \ne P(\text{Absenteeism} = \text{Medium} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{Low}) = 0.2561 \\
# \ne P(\text{Absenteeism} = \text{Medium} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{Medium}) = 0.2561 \\
# \ne P(\text{Absenteeism} = \text{Medium} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{Experience} = \text{High}) = 0.1040
# \end{array}
# $$
#
# Since not all the above stated probabilities are equal for each state of `Absenteeism` = `Low`, `Medium`, `High`, we can assert that the random variables `Experience` and `Absenteeism` are dependent of each other, when not observing `WorkCapacity` state (while  observing the state of `Time` to adjust for backdoors). Arbitrarily choosing the state `backdoorStates` = `{Time = 23}`, we can write:
# $$
# P(\text{Absenteeism} \; | \; \{\texttt{backdoorStates} \}) \ne P(\text{Absenteeism} \; | \; \{ \texttt{backdoorStates} \} \; \cap \; \text{Experience})
# $$
# %% codecell

assert not allEqual(EA.values, EA_1.values, EA_2.values, EA_3.values), "Check: the random variables Experience and Absenteeism are dependent, when intermediary node WorkCapacity is NOT observed (while accounting for backdoors)"



backdoorStates: Dict[Name, Set[State]] = {Time.var : {OBS_STATE_TIME}}

dfEA = eliminateSlice(carModel, query = Absenteeism, evidence = backdoorStates |plus| {Experience.var : Experience.states})
dfEA


# %% markdown
# ### Causal Reasoning: Experience - Absenteeism Effect
# Since the probabilities of `Absenteeism = High` are NOT the same, across all varying conditions of `Time` and `Experience`, this means that there is an active trail between `Experience` and `Absenteeism`.

# %% codecell
backdoorStates: Dict[Name, Set[State]] = {Time.var : {2, 15 , 30}}

dfEA: DataFrame = eliminateSlice(carModel, query = Absenteeism, evidence = backdoorStates |plus| {Experience.var : Experience.states})
dfEA
# %% markdown
# ### Causal Reasoning: Exertion - Absenteeism Effect
# Since the probabilities of `Absenteeism = High` are NOT the same, across all varying conditions of `Time` and `Exertion`, this means that there is an active trail between `Exertion` and `Absenteeism`.

# %% codecell
backdoorStates: Dict[Name, Set[State]] = {Time.var : {2, 15, 30}}

dfXA: DataFrame = eliminateSlice(carModel, query = Absenteeism, evidence = backdoorStates |plus| {Exertion.var : Exertion.states})
dfXA
# %% markdown
# ### Causal Reasoning: Training - Absenteeism Effect
# Since the probabilities of `Absenteeism = High` are NOT the same, across all varying conditions of `Time` and `Training`, this means that there is an active trail between `Training` and `Absenteeism`.

# %% codecell
backdoorStates: Dict[Name, Set[State]] = {Time.var : {2, 15, 30}}


dfTA: DataFrame = eliminateSlice(carModel, query = Absenteeism, evidence = backdoorStates |plus| {Training.var : Training.states})
dfTA

# %% markdown
# ### Causal Reasoning: Experience / Exertion / Training - Absenteeism
# %% codecell

#carModel.is_active_trail(start = [Exertion.var, Training.var, Experience.var], end = Absenteeism.var)
backdoorStates: Dict[Name, Set[State]] = {Time.var : {2, 30}}

dfEETA: DataFrame = eliminateSlice(carModel,
                                   query = Absenteeism,
                                   evidence = backdoorStates |plus| {Exertion.var : Exertion.states,
                                                                     Training.var : Training.states,
                                                                     Experience.var : Experience.states})
dfEETA

















# %% markdown
# ### 4/ Inter-Causal Reasoning in the Car Model (Common Effect Chains)
# For a common effect model $A \rightarrow B \leftarrow C$, there are two cases:
#   * **Marginal Independence:** ($B$ unknown): When $B$ is unknown / unobserved, there is NO active trail between $A$ and $C$, so they are independent, which means the probability of $A$ won't influence probability of $C$ (and vice versa). We can say $P(A) = P(A \; | \; C)$
#   * **Conditional Dependence:** ($B$ fixed): When $B$ is fixed, there IS an active trail between $A$ and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa). We can say $P(A) \ne P(A \; | \; C)$


# %% codecell
drawGraph(carModel)
# %% markdown [markdown]
# #### Testing marginal independence:
# $$
# \color{DodgerBlue}{\text{WorkCapacity (unobserved)}: \;\;\;\;\;\;\;  \text{Exertion} \; \bot \; \text{Training}}
# $$

# Given that **WorkCapacity**'s state is NOT observed, we can make the following equivalent statements:
# * there is NO active trail between **Exertion** and **Training**.
# * **Exertion** and **Training** are locally independent.
# * the probability of **Exertion** won't influence probability of **Training** (and vice versa).
#


# %% markdown [markdown]
# **Testing Marginal Independence:** Using Active Trails Methods
# %% codecell
# When NOT observing the state of the middle node, there is NO active trail (but need to bserve the Time var state because this is a backdoor)
assert not carModel.is_active_trail(start = Exertion.var, end = Training.var, observed = [Time.var])

# When observing the state, there is IS an active trail (also must always account for the backdoor, Time)
assert carModel.is_active_trail(start = Exertion.var, end = Training.var, observed = [WorkCapacity.var, Time.var])

assert carModel.is_active_trail(start = Exertion.var, end = Training.var, observed = [WorkCapacity.var]), "Check: still need to condition on extra variable for this not to be an active trail"

# Finding out which extra variable to condition on: this is the backdoor
assert observedVars(carModel, start= Exertion, end= Training) == [{'Time'}], "Check: all list of extra variables (backdoors) to condition on to ACTIVATE active trail between Exertion and Training"


# See, there is no active trail from Exertion to Training when not observing WorkCapacity.
showActiveTrails(carModel, variables = [Exertion, Training], observed = [Time])
# %% codecell

# See, there IS active trail from Exertion to Training when observing WorkCapacity.
showActiveTrails(carModel, variables = [Exertion, Training], observed = [WorkCapacity, Time])

# %% markdown [markdown]
# **Testing Marginal Independence:** Using Probabilities
# %% codecell
# OBS_STATE_WORKCAPACITY: State = 'Low' # remember, not observing the state of the middle node.
OBS_STATE_TIME: int = 23

backdoorStates: Dict[Name, Set[State]] = {Time.var : {OBS_STATE_TIME}}

TE: DiscreteFactor = elim.query(variables = [Exertion.var], evidence = backdoorStates)

TE_1: DiscreteFactor = elim.query(variables = [Exertion.var], evidence = backdoorStates |plus| {Training.var : {'High'}})

TE_2: DiscreteFactor = elim.query(variables = [Exertion.var], evidence = backdoorStates |plus| {Training.var : {'Medium'}})

TE_3: DiscreteFactor = elim.query(variables = [Exertion.var], evidence = backdoorStates |plus| {Training.var : {'Low'}})
print(TE)

# %% markdown
# Summary of above eliminations, in one chart:
# %% codecell
backdoorStates: Dict[Name, Set[State]] = {Time.var : {OBS_STATE_TIME}}

dfTE = eliminateSlice(carModel,
                      query = Exertion,
                      evidence = backdoorStates |plus| {Training.var : Training.states})
dfTE

# %% markdown [markdown]
#
# The probabilities above are stated formulaically as follows:
# $$
# \begin{array}{ll}
# P(\text{Exertion} = \text{High} \; | \; \Big\{ \text{Time} = 23  \Big\}) \\
# = P(\text{Exertion} = \text{High} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{Low})  \\
# = P(\text{Exertion} = \text{High} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{Medium}) \\
# = P(\text{Exertion} = \text{High} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{High}) \\
# = 0.9927
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{Exertion} = \text{Low} \; | \; \Big\{ \text{Time} = 23 \Big\}) \\
# = P(\text{Exertion} = \text{Low} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{Low})  \\
# = P(\text{Exertion} = \text{Low} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{Medium}) \\
# = P(\text{Exertion} = \text{Low} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{High}) \\
# = 0.0037
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{Exertion} = \text{Medium} \; | \; \Big\{ \text{Time} = 23 \Big\}) \\
# = P(\text{Exertion} = \text{Medium} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{Low})  \\
# = P(\text{Exertion} = \text{Medium} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{Medium}) \\
# = P(\text{Exertion} = \text{Medium} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{High}) \\
# = 0.0037
# \end{array}
# $$
#
# Since all the above stated probabilities are equal for each state of `Exertion` = `Low`, `Medium`, `High`, we can assert that the random variables `Training` and `Exertion` are independent of each other, when NOT observing `WorkCapacity` state (but also observing the state of `Time` to adjust for backdoors). Arbitrarily choosing the `backdoorStates` = `{Time = 23}`, we can write:
# $$
# P(\text{Exertion} \; | \; \{\texttt{backdoorStates} \}) = P(\text{Exertion} \; | \; \{ \texttt{backdoorStates} \} \; \cap \; \text{Training})
# $$
#
# %% codecell
assert allEqual(TE.values, TE_1.values, TE_2.values, TE_3.values), "Check: the random variables Exertion and Training are independent, when intermediary node WorkCapacity is NOT observed (while accounting for backdoors)"






# %% markdown [markdown]
# #### Testing conditional dependence:
# $$
# \color{Chartreuse}{\text{WorkCapacity (observed)}: \;\;\;\;\;\;\;  \text{Exertion} \longrightarrow \text{WorkCapacity} \longrightarrow \text{Training}}
# $$
# $$
# \color{LimeGreen}{\text{WorkCapacity (observed)}: \;\;\;\;\;\;\;  \text{Exertion} \longrightarrow \text{WorkCapacity} \longrightarrow \text{Training}}
# $$
# $$
# \color{Green}{\text{WorkCapacity (observed)}: \;\;\;\;\;\;\;  \text{Exertion} \longrightarrow \text{WorkCapacity} \longrightarrow \text{Training}}
# $$
# Given that **WorkCapacity**'s state is observed, we can make the following equivalent statements:
# * there IS active trail between **Exertion** and **Training**.
# * **Exertion** and **Training** are dependent.
# * the probability of **Exertion** influences probability of **Training** (and vice versa).
#


# %% markdown [markdown]
# **Testing Conditional Dependence:** Using Active Trails Methods
# %% codecell
assert carModel.is_active_trail(start = Exertion.var, end = Training.var, observed = [WorkCapacity.var, Time.var])

# See, there is active trail from Experience to Absenteeism when not observing WorkCapacity variable
showActiveTrails(carModel, variables = [Exertion, Training], observed = [WorkCapacity, Time])

# %% markdown [markdown]
# **Testing Conditional Dependence:** Using Probabilities

# %% codecell
OBS_STATE_WORKCAPACITY: State = 'Low'
OBS_STATE_TIME: int = 23

backdoorStates: Dict[Name, Set[State]] = {Time.var: {OBS_STATE_TIME}, WorkCapacity.var : {OBS_STATE_WORKCAPACITY} }

TWE: DiscreteFactor = elim.query(variables = [Exertion.var],
                                 evidence = backdoorStates)
print(TWE)
# %% codecell

TWE_1: DiscreteFactor = elim.query(variables = [Exertion.var],
                                   evidence = backdoorStates |plus| {Training.var : {'High'}})
print(TWE_1)
# %% codecell
TWE_2: DiscreteFactor = elim.query(variables = [Exertion.var],
                                   evidence = backdoorStates |plus| {Training.var : {'Medium'}})
print(TWE_2)
# %% codecell
TWE_3: DiscreteFactor = elim.query(variables = [Exertion.var],
                                   evidence = backdoorStates |plus| {Training.var : {'Low'}})
print(TWE_3)
# %% markdown [markdown]
#
# $$
# \begin{array}{ll}
# P(\text{Exertion} = \text{High} \; | \; \Big\{  \text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23  \Big\}) = 0.9975 \\
# \ne P(\text{Exertion} = \text{High} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{Low})  = 0.9927 \\
# \ne P(\text{Exertion} = \text{High} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{Medium}) = 0.9927 \\
# \ne P(\text{Exertion} = \text{High} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{High})  = 0.9975
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{Exertion} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\}) = 0.0012 \\
# \ne P(\text{Exertion} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{Low})  = 0.0037 \\
# \ne P(\text{Exertion} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{Medium}) = 0.0037 \\
# \ne P(\text{Exertion} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{High}) = 0.0012
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{Exertion} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\}) = 0.0012 \\
# \ne P(\text{Exertion} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{Low}) = 0.0037 \\
# \ne P(\text{Exertion} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{Medium}) = 0.0037 \\
# \ne P(\text{Exertion} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{Training} = \text{High}) = 0.0012
# \end{array}
# $$
#
# Since not all the above stated probabilities are equal for each state of `Exertion` = `Low`, `Medium`, `High`, we can assert that the random variables `Training` and `Exertion` are dependent, when observing `WorkCapacity` state (and also observing the state of `Time` to adjust for backdoors). Arbitrarily choosing the states `backdoorStates` = `{WorkCapacity = Low, Time = 23}`, we can write:
# $$
# P(\text{Exertion} \; | \; \{\texttt{backdoorStates} \}) \ne P(\text{Exertion} \; | \; \{ \texttt{backdoorStates} \} \; \cap \; \text{Training})
# $$
# %% codecell

assert not allEqual(TWE.values, TWE_1.values, TWE_2.values, TWE_3.values), "Check: the random variables Exertion and Training are dependent, when intermediary node WorkCapacity is observed (while accounting for backdoors)"



backdoorStates: Dict[Name, Set[State]] = {Time.var: {OBS_STATE_TIME}, WorkCapacity.var : {OBS_STATE_WORKCAPACITY}}

dfTWE = eliminateSlice(carModel, query = Exertion,
                       evidence = backdoorStates |plus| {Training.var : Training.states})
dfTWE


# %% markdown
# ### Common Effect Reasoning: WorkCapacity --> Absenteeism <-- Time
# %% codecell
# 4
# TODO left off here

Time_EarlyLate = RandomVariable(var = "Time", states = {2, 30})

observedVars(carModel, start= WorkCapacity, end= Time)
#backdoorAdjustSets(model = carModel, endVar = Absenteeism.var)
#inf = CausalInference(carModel)
#inf.get_all_backdoor_adjustment_sets(X = WorkCapacity.var, Y = Time.var)
carModel.is_active_trail(start = WorkCapacity.var, end = Time.var, observed = [Absenteeism.var, WorkCapacity.var, Time.var])
# %% codecell
df4 = eliminate(carModel, query = WorkCapacity, evidence = [Absenteeism, Experience, Time_EarlyLate])
df4
# %% codecell
# 5
observedVars(carModel, start= Time, end= Exertion)
# %% codecell
# 6
observedVars(carModel, start= Time, end= Experience)
# %% codecell
# 7
observedVars(carModel, start= Time, end= Training)
# %% codecell
# 9
observedVars(carModel, start= Process, end= Injury)
# %% codecell
# 10
observedVars(carModel, start= Tool, end= Process)
# %% codecell
# 13
observedVars(carModel, start= WorkCapacity, end= Injury)
# %% codecell
# 14
observedVars(carModel, start= Time, end= Process)
# %% codecell
