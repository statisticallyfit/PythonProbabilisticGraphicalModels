# %% markdown [markdown]
# # Car Bayesian Network
# Creating bayesian network to model use cases in [https://synergo.atlassian.net/wiki/spaces/CLNTMMC/pages/1812529153/RFP+-+Extra+use+cases+-+Appendix+A](https://synergo.atlassian.net/wiki/spaces/CLNTMMC/pages/1812529153/RFP+-+Extra+use+cases+-+Appendix+A).

# %% markdown [markdown]
# Doing path-setting:
# %% codecell
import os
import sys
from typing import *
from typing import Union, List, Any

import itertools

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


from src.utils.GraphvizUtil import *
from src.utils.NetworkUtil import *
from src.utils.DataUtil import *
from src.utils.GenericUtil import *

import pandas as pd
from pandas.core.frame import DataFrame


# %% markdown [markdown]
# ## Step 1: Creating / Loading Data
# %% codecell
import collections

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


pgmpyToGraph(model = carModel)

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
pgmpyTabularToDataFrame(carModel, queryVar = Time.var)
# %% codecell
pgmpyTabularToDataFrame(carModel, queryVar = ProcessType.var)
# %% codecell
pgmpyTabularToDataFrame(carModel, queryVar = ToolType.var)
# %% codecell
pgmpyTabularToDataFrame(carModel, queryVar = ExperienceLevel.var)
# %% codecell
pgmpyTabularToDataFrame(carModel, queryVar = WorkCapacity.var)
# %% codecell
pgmpyTabularToDataFrame(carModel, queryVar = InjuryType.var)

# %% codecell
pgmpyTabularToDataFrame(carModel, queryVar = AbsenteeismLevel.var)





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
pgmpyToGraph(carModel)
# %% markdown [markdown]
# #### Testing conditional independence:
# $$
# \color{DodgerBlue}{\text{WorkCapacity (observed)}: \;\;\;\;\;\;\;  \text{ExperienceLevel} \; \bot \; \text{AbsenteeismLevel} \; | \; \text{WorkCapacity}}
# $$

# Given that **WorkCapacity**'s state is observed, we can make the following equivalent statements:
# * there is NO active trail between **ExperienceLevel** and **AbsenteeismLevel**.
# * **ExperienceLevel** and **AbsenteeismLevel** are locally independent.
# * the probability of **ExperienceLevel** won't influence probability of **AbsenteeismLevel** (and vice versa).
#

# %% codecell
elim: VariableElimination = VariableElimination(model = carModel)

# %% markdown [markdown]
# **Testing Conditional Independence:** Using Active Trails Methods
# %% codecell
assert carModel.is_active_trail(start = ExperienceLevel.var, end = AbsenteeismLevel.var, observed = None)

assert carModel.is_active_trail(start = ExperienceLevel.var, end = AbsenteeismLevel.var, observed = [WorkCapacity.var]), "Check: still need to condition on extra variable for this not to be an active trail"

# Finding out which extra variable to condition on:
assert observedVars(carModel, startVar = ExperienceLevel.var, endVar = AbsenteeismLevel.var) == [{'Time', 'WorkCapacity'}], "Check: all list of extra variables to condition on to nullify active trail between Experience and Absenteeism"

# Check trail is nullified
assert not carModel.is_active_trail(start = ExperienceLevel.var, end = AbsenteeismLevel.var, observed = [WorkCapacity.var] + [Time.var]), "Check: active trail between Experience and Absenteeism is nullified with the extra variable observed"

# See, there is no active trail from ExperienceLevel to AbsenteeismLevel when observing WorkCapacity and time.
showActiveTrails(carModel, variables = [ExperienceLevel.var, AbsenteeismLevel.var], observed = [WorkCapacity.var, Time.var])

# %% markdown [markdown]
# **Testing Conditional Independence:** Using Probabilities
# %% codecell
OBS_STATE_WORKCAPACITY: State = 'Low'
OBS_STATE_TIME: int = 23

EWA: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = {WorkCapacity.var :OBS_STATE_WORKCAPACITY, Time.var : OBS_STATE_TIME})

EWA_1: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = {WorkCapacity.var : OBS_STATE_WORKCAPACITY, Time.var: OBS_STATE_TIME, ExperienceLevel.var : 'High'})

EWA_2: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = {WorkCapacity.var : OBS_STATE_WORKCAPACITY, Time.var : OBS_STATE_TIME, ExperienceLevel.var : 'Medium'})

EWA_3: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = {WorkCapacity.var : OBS_STATE_WORKCAPACITY, Time.var : OBS_STATE_TIME, ExperienceLevel.var : 'Low'})

print(EWA)
# %% markdown [markdown]
#
# The probabilities above are stated formulaically as follows:
#
# $$
# \begin{array}{ll}
# P(\text{AbsenteeismLevel} = \text{High} \; | \; \Big\{  \text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23  \Big\}) \\
# = P(\text{AbsenteeismLevel} = \text{High} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{Low})  \\
# = P(\text{AbsenteeismLevel} = \text{High} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{Medium}) \\
# = P(\text{AbsenteeismLevel} = \text{High} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{High}) \\
# = 0.4989
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{AbsenteeismLevel} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\}) \\
# = P(\text{AbsenteeismLevel} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{Low})  \\
# = P(\text{AbsenteeismLevel} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{Medium}) \\
# = P(\text{AbsenteeismLevel} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{High}) \\
# = 0.3994
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{AbsenteeismLevel} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\}) \\
# = P(\text{AbsenteeismLevel} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{Low})  \\
# = P(\text{AbsenteeismLevel} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{Medium}) \\
# = P(\text{AbsenteeismLevel} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{High}) \\
# = 0.1017
# \end{array}
# $$
#
# Since all the above stated probabilities are equal for each state of `AbsenteeismLevel` = `Low`, `Medium`, `High`, we can assert that the random variables `ExperienceLevel` and `AbsenteeismLevel` are independent of each other, when observing `WorkCapacity` state (and also observing the state of `Time` to adjust for backdoors). Arbitrarily choosing the states `backdoorStates` = `{WorkCapacity = Low, Time = 23}`, we can write:
# $$
# P(\text{AbsenteeismLevel} \; | \; \{\texttt{backdoorStates} \}) = P(\text{AbsenteeismLevel} \; | \; \{ \texttt{backdoorStates} \} \; \cap \; \text{ExperienceLevel})
# $$
# %% codecell
assert allEqual(EWA.values, EWA_1.values, EWA_2.values, EWA_3.values), "Check: the random variables Experience and Absenteeism are independent, when intermediary node WorkCapacity is observed (while accounting for backdoors)"






# %% markdown [markdown]
# #### Testing marginal dependence:
# $$
# \color{Green}{\text{WorkCapacity (unobserved)}: \;\;\;\;\;\;\;  \text{ExperienceLevel} \longrightarrow \text{WorkCapacity} \longrightarrow \text{AbsenteeismLevel}}
# $$
# Given that **WorkCapacity**'s state is unobserved, we can make the following equivalent statements:
# * there IS active trail between **ExperienceLevel** and **AbsenteeismLevel**.
# * **ExperienceLevel** and **AbsenteeismLevel** are dependent.
# * the probability of **ExperienceLevel** influences probability of **AbsenteeismLevel** (and vice versa).
#


# %% markdown [markdown]
# **Testing Conditional Independence:** Using Active Trails Methods
# %% codecell
assert carModel.is_active_trail(start = ExperienceLevel.var, end = AbsenteeismLevel.var, observed = None)

# See, there is active trail from ExperienceLevel to AbsenteeismLevel when not observing WorkCapacity variable
showActiveTrails(carModel, variables = [ExperienceLevel.var, AbsenteeismLevel.var], observed = None)

# %% markdown [markdown]
# **Testing Conditional Independence:** Using Probabilities

# %% codecell
OBS_STATE_WORKCAPACITY: State = 'Low'
OBS_STATE_TIME: int = 23

EA: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = {Time.var : OBS_STATE_TIME})
print(EA)
# %% codecell
EA_1: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'High', Time.var: OBS_STATE_TIME})
print(EA_1)
# %% codecell
EA_2: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'Medium', Time.var : OBS_STATE_TIME})
print(EA_2)
# %% codecell
EA_3: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'Low', Time.var : OBS_STATE_TIME})
print(EA_3)
# %% markdown [markdown]
#
# The probabilities above are stated formulaically as follows:
# $$
# \begin{array}{ll}
# P(\text{AbsenteeismLevel} = \text{High} \; | \; \Big\{ \text{Time} = 23  \Big\}) = 0.4965 \\
# \ne P(\text{AbsenteeismLevel} = \text{High} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{Low}) = 0.3885  \\
# \ne P(\text{AbsenteeismLevel} = \text{High} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{Medium}) = 0.3885 \\
# \ne P(\text{AbsenteeismLevel} = \text{High} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{High}) = 0.4973
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{AbsenteeismLevel} = \text{Low} \; | \; \Big\{ \text{Time} = 23  \Big\}) = 0.4965 \\
# \ne P(\text{AbsenteeismLevel} = \text{Low} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{Low}) = 0.3553 \\
# \ne P(\text{AbsenteeismLevel} = \text{Low} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{Medium}) = 0.3553 \\
# \ne P(\text{AbsenteeismLevel} = \text{Low} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{High}) = 0.3987
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{AbsenteeismLevel} = \text{Medium} \; | \; \Big\{ \text{Time} = 23  \Big\}) = 0.4965 \\
# \ne P(\text{AbsenteeismLevel} = \text{Medium} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{Low}) = 0.2561 \\
# \ne P(\text{AbsenteeismLevel} = \text{Medium} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{Medium}) = 0.2561 \\
# \ne P(\text{AbsenteeismLevel} = \text{Medium} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{ExperienceLevel} = \text{High}) = 0.1040
# \end{array}
# $$
#
# Since not all the above stated probabilities are equal for each state of `AbsenteeismLevel` = `Low`, `Medium`, `High`, we can assert that the random variables `ExperienceLevel` and `AbsenteeismLevel` are dependent of each other, when not observing `WorkCapacity` state (while  observing the state of `Time` to adjust for backdoors). Arbitrarily choosing the state `backdoorStates` = `{Time = 23}`, we can write:
# $$
# P(\text{AbsenteeismLevel} \; | \; \{\texttt{backdoorStates} \}) \ne P(\text{AbsenteeismLevel} \; | \; \{ \texttt{backdoorStates} \} \; \cap \; \text{ExperienceLevel})
# $$
# %% codecell

assert not allEqual(EA.values, EA_1.values, EA_2.values, EA_3.values), "Check: the random variables Experience and Absenteeism are dependent, when intermediary node WorkCapacity is NOT observed (while accounting for backdoors)"



# %% codecell
# DOing causal reasoning: (varying the time for each set and studying what effect there is between exertion-absenteeisn and experience-absenteeism and training-absenteeism)
carModel.is_active_trail(start = ExperienceLevel.var, end = AbsenteeismLevel.var, observed = [WorkCapacity.var] + [Time.var])
carModel.is_active_trail(start = ExertionLevel.var, end = AbsenteeismLevel.var, observed = [WorkCapacity.var] + [Time.var])
carModel.is_active_trail(start = TrainingLevel.var, end = AbsenteeismLevel.var, observed = [WorkCapacity.var] + [Time.var])

carModel.is_active_trail(start = ExperienceLevel.var, end = AbsenteeismLevel.var, observed = [Time.var])
carModel.is_active_trail(start = ExertionLevel.var, end = AbsenteeismLevel.var, observed = [Time.var])
carModel.is_active_trail(start = TrainingLevel.var, end = AbsenteeismLevel.var, observed = [Time.var])

carModel.active_trail_nodes(variables = [ExperienceLevel.var], observed = [Time.var])
showActiveTrails(carModel, variables = [ExperienceLevel.var], observed = [Time.var])




# %% markdown
# ### Causal Reasoning: Experience - Absenteeism Effect

# %% codecell
EARLY: int = 2
ONE_THIRD: int = 10
MID: int = 15
TWO_THIRD: int = 20
LATE: int = 30

# For early time, studying effects of varying experience level on absenteeism
EA_early_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'Low', Time.var : EARLY})
EA_early_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'Medium', Time.var : EARLY})
EA_early_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'High', Time.var : EARLY})
# %% codecell
print(EA_early_1)
# %% codecell
print(EA_early_2)
# %% codecell
print(EA_early_3)

# %% codecell
# For first-third time, studying effects of varying experience level on absenteeism
EA_onethird_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'Low', Time.var : ONE_THIRD})
EA_onethird_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'Medium', Time.var : ONE_THIRD})
EA_onethird_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'High', Time.var : ONE_THIRD})
# %% codecell
print(EA_onethird_1)
# %% codecell
print(EA_onethird_2)
# %% codecell
print(EA_onethird_3)

# %% codecell
# For two-third time, studying effects of varying experience level on absenteeism
EA_twothird_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'Low', Time.var : TWO_THIRD})
EA_twothird_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'Medium', Time.var : TWO_THIRD})
EA_twothird_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'High', Time.var : TWO_THIRD})
# %% codecell
print(EA_twothird_1)
# %% codecell
print(EA_twothird_2) # higher probability of absentee = High when Experience = Medium for Time nearly to the end
# %% codecell
print(EA_twothird_3)

# %% codecell
# For late time, studying effects of varying experience level on absenteeism
EA_late_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'Low', Time.var : LATE})
EA_late_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'Medium', Time.var : LATE})
EA_late_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExperienceLevel.var : 'High', Time.var : LATE})
# %% codecell
print(EA_late_1)
# %% codecell
print(EA_late_2) # higher probability of absentee = High when Experience = Medium for Time nearly to the end
# %% codecell
# HIgher probability of abseteen = High when Experience = High, for Time = Late (so there is an overriding factor other than Experience that influences Absenteeism), because I made High Experience yield Low Absenteeism.
print(EA_late_3)






# %% markdown
# ### Causal Reasoning: Exertion - Absenteeism Effect
# %% codecell

# For early time, studying effects of varying experience level on absenteeism
XA_early_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Low', Time.var : EARLY})
XA_early_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Medium', Time.var : EARLY})
XA_early_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'High', Time.var : EARLY})
# %% codecell
# TODO Low Exertion for Early Time gives High Absenteeism (?????)  Not correct
print(XA_early_1)
# %% codecell
print(XA_early_2)
# %% codecell
print(XA_early_3)

# %% codecell
# For first-third time, studying effects of varying experience level on absenteeism
XA_onethird_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Low', Time.var : ONE_THIRD})
XA_onethird_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Medium', Time.var : ONE_THIRD})
XA_onethird_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'High', Time.var : ONE_THIRD})
# %% codecell
print(XA_onethird_1)
# %% codecell
print(XA_onethird_2)
# %% codecell
print(XA_onethird_3)

# %% codecell
# For two-third time, studying effects of varying experience level on absenteeism
XA_twothird_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Low', Time.var : TWO_THIRD})
XA_twothird_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Medium', Time.var : TWO_THIRD})
XA_twothird_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'High', Time.var : TWO_THIRD})
# %% codecell
print(XA_twothird_1)
# %% codecell
print(XA_twothird_2) # higher probability of absentee = High when Exertion = Medium for Time nearly to the end
# %% codecell
print(XA_twothird_3)

# %% codecell
# For late time, studying effects of varying experience level on absenteeism
XA_late_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Low', Time.var : LATE})
XA_late_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Medium', Time.var : LATE})
XA_late_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'High', Time.var : LATE})
# %% codecell
print(XA_late_1)
# %% codecell
print(XA_late_2) # higher probability of absentee = High when Experience = Medium for Time nearly to the end
# %% codecell
# HIgher probability of absenteeism = High when Exertion = High, for Time = Late (so there is an overriding factor other than Exertion that influences Absenteeism), because I made High Exertion yield High Absenteeism.
print(XA_late_3)





# %% markdown
# ### Causal Reasoning: Training - Absenteeism
# %% codecell

# For early time, studying effects of varying experience level on absenteeism
TA_early_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Low', Time.var : EARLY})
TA_early_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Medium', Time.var : EARLY})
TA_early_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'High', Time.var : EARLY})
# %% codecell
# NOTE: at the beginning, Training and Absenteeism are oppositely correlated!
# Low training for Early time gives High Absenteeism
print(TA_early_1)
# %% codecell
print(TA_early_2)
# %% codecell
print(TA_early_3)

# %% codecell
# For first-third time, studying effects of varying experience level on absenteeism
TA_onethird_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Low', Time.var : ONE_THIRD})
TA_onethird_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Medium', Time.var : ONE_THIRD})
TA_onethird_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'High', Time.var : ONE_THIRD})
# %% codecell
# High Absentee results when Training is Low for Earlyish Time
print(TA_onethird_1)
# %% codecell
print(TA_onethird_2)
# %% codecell
print(TA_onethird_3)

# %% codecell
# For two-third time, studying effects of varying experience level on absenteeism
TA_twothird_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Low', Time.var : TWO_THIRD})
TA_twothird_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Medium', Time.var : TWO_THIRD})
TA_twothird_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'High', Time.var : TWO_THIRD})
# %% codecell
print(TA_twothird_1)
# %% codecell
print(TA_twothird_2) # higher probability of absentee = High when Training = Medium for Time nearly to the end, probably because workers are tired?
# %% codecell
print(TA_twothird_3)

# %% codecell
# For late time, studying effects of varying experience level on absenteeism
TA_late_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Low', Time.var : LATE})
TA_late_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Medium', Time.var : LATE})
TA_late_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'High', Time.var : LATE})
# %% codecell
print(TA_late_1)
# %% codecell
print(TA_late_2) # higher probability of absentee = High when Experience = Medium for Time nearly to the end
# %% codecell
# HIgher probability of absenteeism = High when Training = High, for Time = Late (so there is an overriding factor other than Experience that influences Absenteeism), because I made High Experience yield Low Absenteeism.
# THUS: # NOTE: at the end of time, Training and Absenteeism are positively correlated! (while at the beginning they were negatively correlated)
print(TA_late_3)






# %% markdown
# ### Causal Reasoning: Experience / Exertion / Training - Absenteeism
# %% codecell
# Function that calculate the distributins based on variables given to pass as observed; we calculate the variable elimination based on all possible combos of the given variable states
#TimeSmall


def eliminateGivenEvidence(model: BayesianModel,
                           query: RandomVariable,
                           evidence: List[RandomVariable] = None) -> DataFrame:

    varStatePairs: List[List[Tuple[Variable, State]]] = [list(itertools.product(*([ev.var], ev.states))) for ev in evidence]

    # Step 2: combine each pairs with the other
    observedTuples: List[Tuple[Variable, State]] = list(itertools.product(*varStatePairs))
    observed: List[Dict[Variable, State]] = list(map(lambda triple: dict(triple), observedTuples))

    elim = VariableElimination(model)
    dists: List[DiscreteFactor] = [elim.query(variables = [query.var], evidence = evPair) for evPair in observed]

    # Step 3: Create the data frame
    evStates: List[State] = list(map(lambda evVar: evVar.states, evidence))
    evStateCombos = list(itertools.product(*evStates))
    # The variable names of the given random variables
    evidence: List[Variable] = list(map(lambda evVar: evVar.var, evidence))

    queryProbs: List[List[Probability]] = np.asarray([dist.values for dist in dists]).T

    topColNames = [''] if evidence == None else pd.MultiIndex.from_tuples(evStateCombos, names=evidence)

    # Use the "ordered" state names instead of queryVar.states so that we get the actual order of the states as used in the Discrete Factor object
    ordStateNames = list(dists[0].state_names.values() )[0]
    df: DataFrame = DataFrame(data = queryProbs, index = ordStateNames, columns = topColNames)
    df.index.name = query.var

    return df.transpose()


TimeSmall = RandomVariable(var = "Time", states = [2, 30])

absentDf: DataFrame = eliminateGivenEvidence(carModel, query= AbsenteeismLevel,
                       evidence = [ExertionLevel, TrainingLevel, ExperienceLevel, TimeSmall])
absentDf

# %% codecell
# TODO: create "getallcausalchainswithinmodel" function , where we traverse each path and get all three-way causal chains
# TODO do the same for common parent and common effect models
model:BayesianModel = BayesianModel([
    ('X', 'F'),
    ('F', 'Y'),
    ('C', 'X'),
    ('A', 'C'),
    ('A', 'D'),
    ('D', 'X'),
    ('D', 'Y'), ('E', 'R'), ('F', 'J'),
    ('B', 'D'),
    ('B', 'E'), ('A', 'Y'), ('O','B'),
    ('E', 'Y'),
    ('X','E'), ('D','E'), ('B', 'X'), ('B','F'), ('E','F'), ('C', 'F'), ('C', 'E'), ('C','Y')
])
pgmpyToGraph(model)

# %% codecell
# STEP 1: get all causal chains
# STEP 2: get the nodes that go in the observed / evidence in order to  nullify active trails (the  middle node + the backdoors from getobservedvars function)

edges: List[Tuple[Variable, Variable]] = list(iter(model.edges()))


roots: List[Variable] = model.get_roots(); roots
leaves: List[Variable] = model.get_leaves(); leaves

# Create all possible causal chains from each node using the edges list (always going downward)

# Create a causal trail (longest possible until reaching the leaves

# METHOD 1: get longest possible trail from ROOT to LEAVES and only then do we chunk it into 3-node paths
startEdges = list(filter(lambda tup: tup[0] in roots, edges)); startEdges
interimNodes: List[Variable] = list(filter(lambda node: not (node in roots) and not (node in leaves), model.nodes())); interimNodes


# Returns dict {varfromvarlist : [children]}
def nodeChildPairs(model: BayesianModel, vars: List[Variable]) -> Dict[Variable, List[Variable]]:
    return [{node : list(model.successors(n = node))} for node in vars]

rootPairs: Dict[Variable, List[Variable]] = nodeChildPairs(model, roots); rootPairs
midPairs = [(node, *list(model.successors(n = node)) ) for node in interimNodes]; midPairs


# METHOD 2: for each edge, connect the tail and tip with matching ends
