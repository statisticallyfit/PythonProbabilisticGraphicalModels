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
Time = RandomVariable(var = "Time", states = list(range(1, 31))) # list(map(lambda day : str(day), range(1, 31))))

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
conditionalDist(carModel, query= Time.var)
# %% codecell
conditionalDist(carModel, query= ProcessType.var)
# %% codecell
conditionalDist(carModel, query= ToolType.var)
# %% codecell
conditionalDist(carModel, query= ExperienceLevel.var)
# %% codecell
conditionalDist(carModel, query= WorkCapacity.var)
# %% codecell
conditionalDist(carModel, query= InjuryType.var)

# %% codecell
conditionalDist(carModel, query= AbsenteeismLevel.var)





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
assert observedVars(carModel, start= ExperienceLevel.var, end= AbsenteeismLevel.var) == [{'Time', 'WorkCapacity'}], "Check: all list of extra variables to condition on to nullify active trail between Experience and Absenteeism"

# Check trail is nullified
assert not carModel.is_active_trail(start = ExperienceLevel.var, end = AbsenteeismLevel.var, observed = [WorkCapacity.var] + [Time.var]), "Check: active trail between Experience and Absenteeism is nullified with the extra variable observed"

# See, there is no active trail from ExperienceLevel to AbsenteeismLevel when observing WorkCapacity and time.
showActiveTrails(carModel, variables = [ExperienceLevel.var, AbsenteeismLevel.var], observed = [WorkCapacity.var, Time.var])

# %% markdown [markdown]
# **Testing Conditional Independence:** Using Probabilities
# %% codecell
OBS_STATE_WORKCAPACITY: State = 'Low'
OBS_STATE_TIME: int = 23

backdoorStates: Dict[VariableName, State] = {WorkCapacity.var : OBS_STATE_WORKCAPACITY, Time.var : OBS_STATE_TIME}


EWA: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = backdoorStates)

EWA_1: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'High'}))

EWA_2: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'Medium'}))

EWA_3: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'Low'}))

# %% codecell


# %% codecell

eliminateSlice(carModel, query = AbsenteeismLevel, evidence = {WorkCapacity.var :[ 'Low'], Time.var :[ 23], ExperienceLevel.var : ['High', 'Low', 'Medium']})

Time_Mid
WorkCapacity2 = RandomVariable(var = "WorkCapacity", states = ['Low'])
ExperienceLevel2 = RandomVariable(var = "ExperienceLevel", states = ['High'])

dfSingle = eliminate(carModel, query = AbsenteeismLevel, evidence = [WorkCapacity2, Time_Mid, ExperienceLevel2])
dfSingle

dfEWA = eliminate(carModel, query = AbsenteeismLevel, evidence = [WorkCapacity, Time_Big, ExperienceLevel])
#dfEWA.loc[('Low', 23)]


#dfEWA.xs(key = 'Low', level = 'WorkCapacity', axis=0)

dfEWA





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
#OBS_STATE_WORKCAPACITY: State = 'Low'
OBS_STATE_TIME: int = 23

backdoorStates: Dict[VariableName, State] = {Time.var : OBS_STATE_TIME}

EA: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = backdoorStates)
print(EA)
# %% codecell
EA_1: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'High'}))
print(EA_1)
# %% codecell
EA_2: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'Medium'}))
print(EA_2)
# %% codecell
EA_3: DiscreteFactor = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'Low'}))
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




# %% markdown
# ### Causal Reasoning: Experience - Absenteeism Effect

# %% codecell
EARLY: int = 2
ONE_THIRD: int = 10
MID: int = 15
TWO_THIRD: int = 20
LATE: int = 30



Time_EarlyLate = RandomVariable(var ="Time", states = [EARLY, ONE_THIRD, TWO_THIRD, LATE])

backdoorStates: Dict[VariableName, State] = {Time.var : EARLY}


# For early time, studying effects of varying experience level on absenteeism
EA_early_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'Low'}))
EA_early_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'Medium'}))
EA_early_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'High'}))
# %% codecell
print(EA_early_1)
# %% codecell
print(EA_early_2)
# %% codecell
print(EA_early_3)

# %% codecell
backdoorStates: Dict[VariableName, State] = {Time.var : ONE_THIRD}

# For first-third time, studying effects of varying experience level on absenteeism
EA_onethird_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'Low'}))
EA_onethird_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'Medium'}))
EA_onethird_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'High'}))
# %% codecell
print(EA_onethird_1)
# %% codecell
print(EA_onethird_2)
# %% codecell
print(EA_onethird_3)

# %% codecell
backdoorStates: Dict[VariableName, State] = {Time.var : TWO_THIRD}

# For two-third time, studying effects of varying experience level on absenteeism
EA_twothird_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'Low'}))
EA_twothird_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'Medium'}))
EA_twothird_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'High'}))
# %% codecell
print(EA_twothird_1)
# %% codecell
print(EA_twothird_2) # higher probability of absentee = High when Experience = Medium for Time nearly to the end
# %% codecell
print(EA_twothird_3)

# %% codecell
backdoorStates: Dict[VariableName, State] = {Time.var : LATE}

# For late time, studying effects of varying experience level on absenteeism
EA_late_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'Low'}))
EA_late_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'Medium'}))
EA_late_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = addEvidence(backdoorStates, {ExperienceLevel.var : 'High'}))
# %% codecell
print(EA_late_1)
# %% codecell
print(EA_late_2) # higher probability of absentee = High when Experience = Medium for Time nearly to the end
# %% codecell
# HIgher probability of abseteen = High when Experience = High, for Time = Late (so there is an overriding factor other than Experience that influences Absenteeism), because I made High Experience yield Low Absenteeism.
print(EA_late_3)


# %% markdown
# Summarizing disparate printing efforts above:
# %% markdown
# ### Causal Reasoning: Experience - Absenteeism

# %% codecell
experDf: DataFrame = eliminate(carModel, query = AbsenteeismLevel, evidence = [ExperienceLevel, Time_EarlyLate])

experDf
# %% markdown
# ### Causal Reasoning: Exertion - Absenteeism
# %% codecell
exertDf: DataFrame = eliminate(carModel, query = AbsenteeismLevel, evidence= [ExertionLevel, Time_EarlyLate])
exertDf
# %% markdown
# ### Causal Reasoning: Training - Absenteeism
# %% codecell
trainDf: DataFrame = eliminate(carModel, query = AbsenteeismLevel, evidence= [TrainingLevel, Time_EarlyLate])
trainDf

# %% markdown
# ### Causal Reasoning: Experience / Exertion / Training - Absenteeism
# %% codecell

absentDf: DataFrame = eliminate(carModel,
                                query= AbsenteeismLevel,
                                evidence = [ExertionLevel, TrainingLevel, ExperienceLevel, Time_EarlyLate])
absentDf



























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
# \color{DodgerBlue}{\text{WorkCapacity (unobserved)}: \;\;\;\;\;\;\;  \text{ExertionLevel} \; \bot \; \text{TrainingLevel}}
# $$

# Given that **WorkCapacity**'s state is NOT observed, we can make the following equivalent statements:
# * there is NO active trail between **ExertionLevel** and **TrainingLevel**.
# * **ExertionLevel** and **TrainingLevel** are locally independent.
# * the probability of **ExertionLevel** won't influence probability of **TrainingLevel** (and vice versa).
#


# %% markdown [markdown]
# **Testing Marginal Independence:** Using Active Trails Methods
# %% codecell
# When NOT observing the state of the middle node, there is NO active trail (but need to bserve the Time var state because this is a backdoor)
assert not carModel.is_active_trail(start = ExertionLevel.var, end = TrainingLevel.var, observed = [Time.var])

# When observing the state, there is IS an active trail (also must always account for the backdoor, Time)
assert carModel.is_active_trail(start = ExertionLevel.var, end = TrainingLevel.var, observed = [WorkCapacity.var, Time.var])

assert carModel.is_active_trail(start = ExertionLevel.var, end = TrainingLevel.var, observed = [WorkCapacity.var]), "Check: still need to condition on extra variable for this not to be an active trail"

# Finding out which extra variable to condition on: this is the backdoor
assert observedVars(carModel, start= ExertionLevel.var, end= TrainingLevel.var) == [{'Time'}], "Check: all list of extra variables (backdoors) to condition on to ACTIVATE active trail between Exertion and Training"



# See, there is no active trail from Exertion to Training when not observing WorkCapacity.
showActiveTrails(carModel, variables = [ExertionLevel.var, TrainingLevel.var], observed = [Time.var])
# %% codecell

# See, there IS active trail from Exertion to Training when observing WorkCapacity.
showActiveTrails(carModel, variables = [ExertionLevel.var, TrainingLevel.var], observed = [WorkCapacity.var, Time.var])

# %% markdown [markdown]
# **Testing Marginal Independence:** Using Probabilities
# %% codecell
# OBS_STATE_WORKCAPACITY: State = 'Low' # remember, not observing the state of the middle node.
OBS_STATE_TIME: int = 23

backdoorStates: Dict[VariableName, State] = {Time.var : OBS_STATE_TIME}

TE: DiscreteFactor = elim.query(variables = [ExertionLevel.var], evidence = backdoorStates)

TE_1: DiscreteFactor = elim.query(variables = [ExertionLevel.var], evidence = addEvidence(backdoorStates, {TrainingLevel.var : 'High'}))

TE_2: DiscreteFactor = elim.query(variables = [ExertionLevel.var], evidence = addEvidence(backdoorStates, {TrainingLevel.var : 'Medium'}))

TE_3: DiscreteFactor = elim.query(variables = [ExertionLevel.var], evidence = addEvidence(backdoorStates, {TrainingLevel.var : 'Low'}))
print(TE)

# %% markdown
# Summary of above eliminations, in one chart:
# %% codecell
dfTE = eliminate(carModel, query = ExertionLevel, evidence = [TrainingLevel, Time_Mid])
dfTE

# %% markdown [markdown]
#
# The probabilities above are stated formulaically as follows:
# $$
# \begin{array}{ll}
# P(\text{ExertionLevel} = \text{High} \; | \; \Big\{ \text{Time} = 23  \Big\}) \\
# = P(\text{ExertionLevel} = \text{High} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{Low})  \\
# = P(\text{ExertionLevel} = \text{High} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{Medium}) \\
# = P(\text{ExertionLevel} = \text{High} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{High}) \\
# = 0.9927
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{ExertionLevel} = \text{Low} \; | \; \Big\{ \text{Time} = 23 \Big\}) \\
# = P(\text{ExertionLevel} = \text{Low} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{Low})  \\
# = P(\text{ExertionLevel} = \text{Low} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{Medium}) \\
# = P(\text{ExertionLevel} = \text{Low} \; | \; \Big\{\text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{High}) \\
# = 0.0037
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{ExertionLevel} = \text{Medium} \; | \; \Big\{ \text{Time} = 23 \Big\}) \\
# = P(\text{ExertionLevel} = \text{Medium} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{Low})  \\
# = P(\text{ExertionLevel} = \text{Medium} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{Medium}) \\
# = P(\text{ExertionLevel} = \text{Medium} \; | \; \Big\{ \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{High}) \\
# = 0.0037
# \end{array}
# $$
#
# Since all the above stated probabilities are equal for each state of `ExertionLevel` = `Low`, `Medium`, `High`, we can assert that the random variables `TrainingLevel` and `ExertionLevel` are independent of each other, when NOT observing `WorkCapacity` state (but also observing the state of `Time` to adjust for backdoors). Arbitrarily choosing the `backdoorStates` = `{Time = 23}`, we can write:
# $$
# P(\text{ExertionLevel} \; | \; \{\texttt{backdoorStates} \}) = P(\text{ExertionLevel} \; | \; \{ \texttt{backdoorStates} \} \; \cap \; \text{TrainingLevel})
# $$
#
# %% codecell
assert allEqual(TE.values, TE_1.values, TE_2.values, TE_3.values), "Check: the random variables Exertion and Training are independent, when intermediary node WorkCapacity is NOT observed (while accounting for backdoors)"






# %% markdown [markdown]
# #### Testing conditional dependence:
# $$
# \color{Chartreuse}{\text{WorkCapacity (observed)}: \;\;\;\;\;\;\;  \text{ExertionLevel} \longrightarrow \text{WorkCapacity} \longrightarrow \text{TrainingLevel}}
# $$
# $$
# \color{LimeGreen}{\text{WorkCapacity (observed)}: \;\;\;\;\;\;\;  \text{ExertionLevel} \longrightarrow \text{WorkCapacity} \longrightarrow \text{TrainingLevel}}
# $$
# $$
# \color{Green}{\text{WorkCapacity (observed)}: \;\;\;\;\;\;\;  \text{ExertionLevel} \longrightarrow \text{WorkCapacity} \longrightarrow \text{TrainingLevel}}
# $$
# Given that **WorkCapacity**'s state is observed, we can make the following equivalent statements:
# * there IS active trail between **ExertionLevel** and **TrainingLevel**.
# * **ExertionLevel** and **TrainingLevel** are dependent.
# * the probability of **ExertionLevel** influences probability of **TrainingLevel** (and vice versa).
#


# %% markdown [markdown]
# **Testing Conditional Dependence:** Using Active Trails Methods
# %% codecell
assert carModel.is_active_trail(start = ExertionLevel.var, end = TrainingLevel.var, observed = [WorkCapacity.var, Time.var])

# See, there is active trail from ExperienceLevel to AbsenteeismLevel when not observing WorkCapacity variable
showActiveTrails(carModel, variables = [ExertionLevel.var, TrainingLevel.var], observed = [WorkCapacity.var, Time.var])

# %% markdown [markdown]
# **Testing Conditional Dependence:** Using Probabilities

# %% codecell
OBS_STATE_WORKCAPACITY: State = 'Low'
OBS_STATE_TIME: int = 23

backdoorStates: Dict[VariableName, State] = {Time.var: OBS_STATE_TIME, WorkCapacity.var : OBS_STATE_WORKCAPACITY}

TWE: DiscreteFactor = elim.query(variables = [ExertionLevel.var],
                                 evidence = backdoorStates)
print(TWE)
# %% codecell

TWE_1: DiscreteFactor = elim.query(variables = [ExertionLevel.var],
                                   evidence = addEvidence(backdoorStates, {TrainingLevel.var : 'High'}))
print(TWE_1)
# %% codecell
TWE_2: DiscreteFactor = elim.query(variables = [ExertionLevel.var],
                                   evidence = addEvidence(backdoorStates, {TrainingLevel.var : 'Medium'}))
print(TWE_2)
# %% codecell
TWE_3: DiscreteFactor = elim.query(variables = [ExertionLevel.var],
                                   evidence = addEvidence(backdoorStates, {TrainingLevel.var : 'Low'}))
print(TWE_3)
# %% markdown [markdown]
#
# $$
# \begin{array}{ll}
# P(\text{ExertionLevel} = \text{High} \; | \; \Big\{  \text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23  \Big\}) = 0.9975 \\
# \ne P(\text{ExertionLevel} = \text{High} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{Low})  = 0.9927 \\
# \ne P(\text{ExertionLevel} = \text{High} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{Medium}) = 0.9927 \\
# \ne P(\text{ExertionLevel} = \text{High} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{High})  = 0.9975
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{ExertionLevel} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\}) = 0.0012 \\
# \ne P(\text{ExertionLevel} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{Low})  = 0.0037 \\
# \ne P(\text{ExertionLevel} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{Medium}) = 0.0037 \\
# \ne P(\text{ExertionLevel} = \text{Low} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{High}) = 0.0012
# \end{array}
# $$
# $$
# \begin{array}{ll}
# P(\text{ExertionLevel} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\}) = 0.0012 \\
# \ne P(\text{ExertionLevel} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{Low}) = 0.0037 \\
# \ne P(\text{ExertionLevel} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{Medium}) = 0.0037 \\
# \ne P(\text{ExertionLevel} = \text{Medium} \; | \; \Big\{\text{WorkCapacity} = \text{Low} \; \cap \; \text{Time} = 23 \Big\} \; \cap \; \text{TrainingLevel} = \text{High}) = 0.0012
# \end{array}
# $$
#
# Since not all the above stated probabilities are equal for each state of `ExertionLevel` = `Low`, `Medium`, `High`, we can assert that the random variables `TrainingLevel` and `ExertionLevel` are dependent, when observing `WorkCapacity` state (and also observing the state of `Time` to adjust for backdoors). Arbitrarily choosing the states `backdoorStates` = `{WorkCapacity = Low, Time = 23}`, we can write:
# $$
# P(\text{ExertionLevel} \; | \; \{\texttt{backdoorStates} \}) \ne P(\text{ExertionLevel} \; | \; \{ \texttt{backdoorStates} \} \; \cap \; \text{TrainingLevel})
# $$
# %% codecell

assert not allEqual(TWE.values, TWE_1.values, TWE_2.values, TWE_3.values), "Check: the random variables Exertion and Training are dependent, when intermediary node WorkCapacity is observed (while accounting for backdoors)"



# %% markdown
# ### Common Effect Reasoning: Exertion --> WorkCapacity <-- Training
# %% codecell
Time_EarlyLate = RandomVariable(var ="Time", states = [2, 30])
Time_Mid = RandomVariable(var ="Time", states = [23])
# 1
# backdoor state here
observedVars(carModel, start= ExertionLevel.var, end= ExperienceLevel.var)
# %% codecell
assert carModel.is_active_trail(start = ExertionLevel.var, end = ExperienceLevel.var, observed = [Time.var, WorkCapacity.var]), "Check: that observing the backdoor state and middle node state CREATES an active trail between Exertion and Experience (common effect model)"
# %% markdown
# Observations:
#
# * When `WorkCapacity = Low` and `ExperienceLevel = High` and `Time = EARLY` then the query variable `ExertionLevel = High` with probability 0.9926. Intuitively: most likely for worker to be exerted early on in the month, even when experience is high and also when work capacity is low.
# * When `WorkCapacity = Low` and `ExperienceLevel = High` and `Time = LATE` then the query variable `ExertionLevel = High` with probability 0.9926. Intuitively: worker highly probable to be highly exerted when he has lots of experience, has low work capacity and time is late in the month.
# * ...
# %% codecell
# backdoor vars = workcapacity, Time
df1 = eliminate(carModel, query = ExertionLevel, evidence = [WorkCapacity, Time_EarlyLate, ExperienceLevel])
df1
# %% codecell
# 2
observedVars(carModel, start= ExertionLevel.var, end= TrainingLevel.var)
# %% codecell
assert carModel.is_active_trail(start = ExertionLevel.var, end = TrainingLevel.var, observed = [Time.var, WorkCapacity.var]), "Check: that observing the backdoor state and middle node state CREATES an active trail between Exertion and TrainingLevel (common effect model)"
# TODO observing none is also true but that cannot be true because there should be NO active trail when NOT observing the middle node WorkCpacity ???
#carModel.is_active_trail(start = ExertionLevel.var, end = TrainingLevel.var, observed = None)
# %% codecell
df2 = eliminate(carModel, query = ExertionLevel, evidence = [WorkCapacity, Time_EarlyLate, TrainingLevel])
df2
# %% codecell
# 4
# TODO left off here
observedVars(carModel, start= WorkCapacity.var, end= Time.var)
#backdoorAdjustSets(model = carModel, endVar = AbsenteeismLevel.var)
#inf = CausalInference(carModel)
#inf.get_all_backdoor_adjustment_sets(X = WorkCapacity.var, Y = Time.var)
carModel.is_active_trail(start = WorkCapacity.var, end = Time.var, observed = [AbsenteeismLevel.var, WorkCapacity.var, Time.var])
# %% codecell
df1 = eliminate(carModel, query = WorkCapacity, evidence = [AbsenteeismLevel, ExperienceLevel, Time_EarlyLate])
df1
# %% codecell
# 5
observedVars(carModel, start= Time.var, end= ExertionLevel.var)
# %% codecell
# 6
observedVars(carModel, start= Time.var, end= ExperienceLevel.var)
# %% codecell
# 7
observedVars(carModel, start= Time.var, end= TrainingLevel.var)
# %% codecell
# 9
observedVars(carModel, start= ProcessType.var, end= InjuryType.var)
# %% codecell
# 10
observedVars(carModel, start= ToolType.var, end= ProcessType.var)
# %% codecell
# 13
observedVars(carModel, start= WorkCapacity.var, end= InjuryType.var)
# %% codecell
# 14
observedVars(carModel, start= Time.var, end= ProcessType.var)
# %% codecell
