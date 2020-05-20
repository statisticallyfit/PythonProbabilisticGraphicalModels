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

import pandas as pd
from pandas.core.frame import DataFrame


# %% markdown
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


# %% markdown
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

# %% markdown
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
# ## Inference in Bayesian Car Model
#
# Now let us do inference in a  Bayesian model and predict values using this model over new data points for ML tasks.
#
# ### 1. Causal Reasoning in the Car Model
# For a causal model $A \rightarrow B \rightarrow C$, there are two cases:
#   * **Marginal Dependence:** ($B$ unknown): When $B$ is unknown / unobserved, there is an active trail between $A$ and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa).
#   * **Conditional Independence:** ($B$ fixed): When $B$ is fixed, there is NO active trail between $A$ and $C$, so they are independent, which means the probability of $A$ won't influence probability of $C$ (and vice versa).


# %% codecell
pgmpyToGraph(carModel)
# %% markdown [markdown]
# ### Case 1: Marginal Dependence (for Causal Model)
# For a causal model $A \rightarrow B \rightarrow C$, when the state of the middle node $B$ is unobserved, then an active trail is created between the nodes, namely the active trail is $A \rightarrow B \rightarrow C$. Information can now flow from node $A$ to node $C$ via node $B$. This implies there is a dependency between nodes $A$ and $C$, so the probability of $A$ taking on any of its states can influence the probability of $C$ taking on any of its states. This is called **marginal dependence** We can write this as: $P(A | C) \ne P(A)$
#
# $\color{red}{\text{TODO}}$ left off here trying to refactor the text (continue from sublime notes pg 35 Korb and pg 336 Bayesiabook)
# $$
# \color{Green}{ \text{ExertionLevel (unobserved): }\;\;\;\;\;\;\;\;\; \text{Time} \longrightarrow \text{Exertion} \longrightarrow \text{WorkCapacity}}
# $$
#
# Given that the state of `ExertionLevel` is unobserved, we can make the following equivalent statements:
# * there IS an active trail between `Time` and `WorkCapacity`.
# * the random variables `Time` and `WorkCapacity` are dependent.
# * the probability of `Time` can influence probability of `WorkCapacity` (and vice versa).
#
# Similarly, the same kinds of statements can be made for the other causal chain pathways in the graph:
#
# $$
# \color{Green}{ \text{InjuryType (unobserved): }\;\;\;\;\;\;\;\;\; \text{ProcessType} \longrightarrow \text{Injury} \longrightarrow \text{AbsenteeismLevel}}
# $$
# Given that the state of `InjuryType` is unobserved, we can make the following equivalent statements:
# * there IS an active trail between `ProcessType` and `AbsenteeismLevel`.
# * the random variables `ProcessType` and `AbsenteeismLevel` are dependent.
# * the probability of `ProcessType` can influence probability of `AbsenteeismLevel` (and vice versa).
#

# %% codecell
vals = [{'A'}, {'B', 'C'}, {'D', 'E', 'A'}, {'D', 'R', 'C', 'B'}]
combos = list(itertools.combinations(vals, r = 2)); combos
# 1. create combination tuples
# 2. check if the one is the superset of the other, if so, then unionize them, else they go separate
# 3. remove duplicates
for first, sec in combos:
    if first.issubset(sec) or first.issuperset(sec):
        unions.append(first.union(sec))
    else:
        unions.append(first)
        unions.append(sec)

list(map(lambda lst : set(lst), np.unique(list(map(lambda sett : list(sett), unions)))))
combos[1][0].issubset(combos[1][1])
combos[1][0]
combos[1][1]
# %% codecell
elim: VariableElimination = VariableElimination(model = carModel)


# %% markdown [markdown]
# **Verify:** Using Probabilities (example of $B \rightarrow A \rightarrow J$ trail)
# * **NOTE:** Causal Reasoning For Causal Model:
# %% markdown [markdown]
# The probability below is:
# $$
# P(\text{JohnCalls} = \text{True}) =  0.0521
# $$
# %% codecell
BJ: DiscreteFactor = elim.query(variables = [JohnCalls.var], evidence = None)
print(BJ)
# %% markdown [markdown]
# Below we see that when there is evidence of `Burglary` and no `Alarm` was observed, there is a higher probability of `JohnCalls`, compared to when no `Burglary` was observed and no `Alarm` was observed (BJ). Specifically,
# $$
# P(\text{JohnCalls} = \text{True} \; | \; \text{Burglary} = \text{True}) = 0.8490
# $$
# while above:
# $$
# P(\text{JohnCalls} = \text{True}) =  0.0521
# $$
# %% codecell
BJ_1 = elim.query(variables = [JohnCalls.var], evidence = {Burglary.var: 'True'})
print(BJ_1)
# %% markdown [markdown]
# Below we see that when there is no `Burglary` and no `Alarm` was observed, there is a lower probability of `JohnCalls`, compared to when `Burglary` did occur and no `Alarm` was observed (BJ_1).
# $$
# P(\text{JohnCalls} = \text{True} \; | \; \text{Burglary} = \text{False}) = 0.0513
# $$
# %% codecell
BJ_2 = elim.query(variables = [JohnCalls.var], evidence = {Burglary.var:'False'})
print(BJ_2)
# %% markdown [markdown]
# Through the above steps, we observed some probabilities conditional on some states. To present them all in one place, here are the probabilities that `JohnCalls = True`, where the last two are conditional on `Burglary.states = ['True', 'False']`.
# $$
# \begin{array}{ll}
# P(\text{JohnCalls} = \text{True}) &= 0.0521 \\
# P(\text{JohnCalls} = \text{True} \; | \; \text{Burglary} = \text{True}) &= 0.8490 \\
# P(\text{JohnCalls} = \text{True} \; | \; \text{Burglary} = \text{False}) &= 0.0513
# \end{array}
# $$
# From probability theory, we know that two random variables $A$ and $B$ are independent if and only if $P(A) = P(A \; | \; B)$ (by definition this statement holds for all the states that the random variables $A$ and $B$ can take on).
#
# Therefore, the fact that the above probabilities are not the same implies the random variables `JohnCalls` and `Burglary` are dependent (not independent). This is expressed in probability notation as follows:
# $$
# P(\text{JohnCalls}) \ne P(\text{JohnCalls} \; | \; \text{Burglary})
# $$
# Using pgmpy, we can access the previously calculated probabilites using the `.values` accessor to assert the probabilities aren't equal, which asserts the random variables `JohnCalls` and `Burglary` are dependent:
# %% codecell
assert (BJ.values != BJ_1.values).all() and (BJ.values != BJ_2.values).all(), "Check: variables Burglary and JohnCalls are dependent, given that Alarm's state is unobserved "


# %% markdown [markdown]
# **Verify:** Using Probabilities (example of $E \rightarrow A \rightarrow M$ trail)
# * **NOTE:** Causal Reasoning For Causal Model:
# %% markdown [markdown]
# The probability below is:
# $$
# P(\text{MaryCalls} = \text{True}) = 0.0117
# $$
# %% codecell
EM: DiscreteFactor = elim.query(variables = [MaryCalls.var], evidence = None)
print(EM)
# %% markdown [markdown]
# Below we see that when `Earthquake` occurs and no `Alarm` was observed, there is a higher probability of `MaryCalls`, compared to when neither `Alarm` nor `Earthquake` were observed:
# $$
# P(\text{MaryCalls} = \text{True} \; | \; \text{Earthquake} = \text{True}) = 0.2106
# $$
# %% codecell
EM_1 = elim.query(variables = [MaryCalls.var], evidence = {Earthquake.var:'True'})
print(EM_1)
# %% markdown [markdown]
# Below we see that when `Earthquake` does not occur and no `Alarm` was observed, there is a lower probability of `MaryCalls`, compared to when `Earthquake` occurs and no `Alarm` was observed:
# $$
# P(\text{MaryCalls} = \text{True} \; | \; \text{Earthquake} = \text{False}) = 0.0113
# $$
# Incidentally, this is the same probability as when no `Earthquake` and no `Alarm` was observed.
# %% codecell
EM_2 = elim.query(variables = [MaryCalls.var], evidence = {Earthquake.var:'False'})
print(EM_2)

# %% markdown [markdown]
# Through the above steps, we observed some probabilities conditional on some states. To present them all in one place, here are the probabilities that `MaryCalls = True`, where the last two are conditional on `Earthquake.states = ['True', 'False']`.
# $$
# \begin{array}{ll}
# P(\text{MaryCalls} = \text{True}) &= 0.0117 \\
# P(\text{MaryCalls} = \text{True} \; | \; \text{Earthquake} = \text{True}) &= 0.2106 \\
# P(\text{MaryCalls} = \text{True} \; | \; \text{Earthquake} = \text{False}) &= 0.0113
# \end{array}
# $$
# From probability theory, we know that two random variables $A$ and $B$ are independent if and only if $P(A) = P(A \; | \; B)$ (by definition this statement holds for all the states that the random variables $A$ and $B$ can take on).
#
# Therefore, the fact that the above probabilities are not the same implies the random variables `MaryCalls` and `Earthquake` are dependent (not independent). This is expressed in probability notation as follows:
# $$
# P(\text{MaryCalls}) \ne P(\text{MaryCalls} \; | \; \text{Earthquake})
# $$
# Using pgmpy, we can access the previously calculated probabilites using the `.values` accessor to assert the probabilities aren't equal, which asserts that the random variables `MaryCalls` and `Earthquake` are dependent:
# %% codecell
assert (EM.values != EM_1.values).all() and (EM.values != EM_2.values).all(), "Check: random variables Earthquake and MaryCalls are independent, given that Alarm state is unobserved "




# %% markdown [markdown]
# ### Case 2: Conditional Independence (for Causal Model)
# For a causal model $A \rightarrow B \rightarrow C$, when the state of the middle node $B$ is unobserved, then an active trail is created between the nodes, namely the active trail is $A \rightarrow B \rightarrow C$. Information can now flow from node $A$ to node $C$ via node $B$. This implies there is a dependency between nodes $A$ and $C$, so the probability of $A$ taking on any of its states can influence the probability of $C$ taking on any of its states. This is called **marginal dependence** We can write this as: $P(A | C) \ne P(A)$ or even as $P(C | A) \ne P(C)$.
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (observed): }\;\;\;\;\;\;\;\; \text{Burglary} \; \bot \; \text{MaryCalls} \; | \; \text{Alarm}}
# $$
# Given that `Alarm`'s state is fixed / observed, we can make the following equivalent statements:
# * there is NO active trail between `Burglary` and `MaryCalls`.
# * `Burglary` and `MaryCalls` are locally independent.
# * the probability of `Burglary` won't influence probability of `MaryCalls` (and vice versa).
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (observed): }\;\;\;\;\;\;\;\; \text{Burglary} \; \bot \; \text{JohnCalls} \; | \; \text{Alarm}}
# $$
# Given that `Alarm`'s state is fixed / observed, we can make the following equivalent statements:
# * there is NO active trail between `Burglary` and `JohnCalls`.
# * `Burglary` and `JohnCalls` are locally independent.
# * the probability of `Burglary` won't influence probability of `JohnCalls` (and vice versa).
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (observed): }\;\;\;\;\;\;\;\; \text{Earthquake} \; \bot \; \text{MaryCalls} \; | \; \text{Alarm}}
# $$
# Given that `Alarm`'s state is fixed / observed, we can make the following equivalent statements:
# * there is NO active trail between `Earthquake` and `MaryCalls`.
# * `Earthquake` and `MaryCalls` are locally independent.
# * the probability of `Earthquake` won't influence probability of `MaryCalls` (and vice versa).
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (observed): }\;\;\;\;\;\;\;\; \text{Earthquake} \; \bot \; \text{JohnCalls} \; | \; \text{Alarm}}
# $$
# Given that `Alarm`'s state is fixed / observed, we can make the following equivalent statements:
# * there is NO active trail between `Earthquake` and `JohnCalls`.
# * `Earthquake` and `JohnCalls` are locally independent.
# * the probability of `Earthquake` won't influence probability of `JohnCalls` (and vice versa).
# %% markdown [markdown]
# **Verify:** Using Active Trails
# %% codecell
assert not carModel.is_active_trail(start = Burglary.var, end = MaryCalls.var, observed = Alarm.var)
assert not carModel.is_active_trail(start = Burglary.var, end = JohnCalls.var, observed = Alarm.var)
assert not carModel.is_active_trail(start = Earthquake.var, end = MaryCalls.var, observed = Alarm.var)
assert not carModel.is_active_trail(start = Earthquake.var, end = JohnCalls.var, observed = Alarm.var)

showActiveTrails(model = carModel, variables = [Burglary.var, MaryCalls.var], observed = Alarm.var)

# %% markdown [markdown]
# **Verify:** Using Independencies (just the $(B \; \bot \; M \; | \; A)$ independence)
# %% codecell
indepBurglary: IndependenceAssertion = Independencies([Burglary.var, MaryCalls.var, [Alarm.var]]).get_assertions()[0]; indepBurglary

indepMary: IndependenceAssertion = Independencies([MaryCalls.var, Burglary.var, [Alarm.var]]).get_assertions()[0]; indepMary

# Using the fact that closure returns independencies that are IMPLIED by the current independencies:
assert (str(indepMary) == '(MaryCalls _|_ Burglary | Alarm)' and
        indepMary in carModel.local_independencies(MaryCalls.var).closure().get_assertions()),  \
        "Check 1: Burglary and MaryCalls are independent once conditional on Alarm"

assert (str(indepBurglary) == '(Burglary _|_ MaryCalls | Alarm)' and
        indepBurglary in carModel.local_independencies(MaryCalls.var).closure().get_assertions()), \
        "Check 2: Burglary and MaryCalls are independent once conditional on Alarm"

carModel.local_independencies(MaryCalls.var).closure()

# %% codecell
# See: MaryCalls and Burglary are conditionally independent on Alarm:
indepSynonymTable(model = alarmModel_brief, queryNode = 'M')



# %% markdown [markdown]
# **Verify:** Using Probabilities Method (just the $(E \; \bot \; J \; | \; A)$ independence)

# %% markdown [markdown]
# Doing the case where `Alarm = True`:
# %% codecell

# Case 1: Alarm = True
EAJ: DiscreteFactor = elim.query(variables = [JohnCalls.var], evidence = {Alarm.var: 'True'})
EAJ_1 = elim.query(variables = [JohnCalls.var], evidence = {Alarm.var: 'True', Earthquake.var:'True'})
EAJ_2 = elim.query(variables = [JohnCalls.var], evidence = {Alarm.var: 'True', Earthquake.var:'False'})

print(EAJ)

# %% markdown [markdown]
# Through the above steps, we observed some probabilities conditional on some states:
# $$
# \begin{array}{ll}
# P(\text{JohnCalls} = \text{True} \; | \; \text{Alarm} = \text{True})
# &= P(\text{JohnCalls} = \text{True} \; | \; \text{Alarm} = \text{True} \; \cap \; \text{Earthquake} = \text{True})  \\
# &= P(\text{JohnCalls} = \text{True} \; | \; \text{Alarm} = \text{True} \; \cap \; \text{Earthquake} = \text{False}) \\
# &= 0.90
# \end{array}
# $$
# From probability theory, we know that two random variables $A$ and $B$ are independent if and only if $P(A) = P(A \; | \; B)$ (by definition this statement holds for all the states that the random variables $A$ and $B$ can take on).
#
# Therefore, the fact that the above probabilities ARE the same implies the random variables `JohnCalls` and `Earthquake` are independent when having observed `Alarm = True`. This is expressed in probability notation as follows:
# $$
# P(\text{JohnCalls} \; | \; \text{Alarm = True}) = P(\text{JohnCalls} \; | \; \text{Earthquake} \cap \text{Alarm = True})
# $$

# Using pgmpy, we can access the previously calculated probabilites using the `.values` accessor to assert the probabilities ARE equal, which asserts that the random variables `JohnCalls` and `Earthquake` are independent (given `Alarm = True`):

# %% codecell

assert (EAJ.values == EAJ_1.values).all() and (EAJ.values == EAJ_2.values).all(), "Check: random variables Earthquake and JohnCalls are independent when Alarm state is observed (e.g. Alarm = True)"



# %% markdown [markdown]
# Doing the case where `Alarm = True`:
# %% codecell
# Case 2: Alarm = False
EAJ: DiscreteFactor = elim.query(variables = [JohnCalls.var], evidence = {Alarm.var: 'False'})
EAJ_1 = elim.query(variables = [JohnCalls.var], evidence = {Alarm.var: 'False', Earthquake.var:'True'})
EAJ_2 = elim.query(variables = [JohnCalls.var], evidence = {Alarm.var: 'False', Earthquake.var:'False'})

print(EAJ)

# %% markdown [markdown]
# Through the above steps, we observed some probabilities conditional on some states:
# $$
# \begin{array}{ll}
# P(\text{JohnCalls} = \text{True} \; | \; \text{Alarm} = \text{False})
# &= P(\text{JohnCalls} = \text{True} \; | \; \text{Alarm} = \text{False} \; \cap \; \text{Earthquake} = \text{True})  \\
# &= P(\text{JohnCalls} = \text{True} \; | \; \text{Alarm} = \text{False} \; \cap \; \text{Earthquake} = \text{False}) \\
# &= 0.05
# \end{array}
# $$
# From probability theory, we know that two random variables $A$ and $B$ are independent if and only if $P(A) = P(A \; | \; B)$ (by definition this statement holds for all the states that the random variables $A$ and $B$ can take on).
#
# Therefore, the fact that the above probabilities ARE the same implies the random variables `JohnCalls` and `Earthquake` are independent when having observed `Alarm = False`. This is expressed in probability notation as follows:
# $$
# P(\text{JohnCalls} \; | \; \text{Alarm = False}) = P(\text{JohnCalls} \; | \; \text{Earthquake} \cap \text{Alarm = False})
# $$

# Using pgmpy, we can access the previously calculated probabilites using the `.values` accessor to assert the probabilities ARE equal, which asserts that the random variables `JohnCalls` and `Earthquake` are independent (given `Alarm = False`)
# %% codecell
assert (EAJ.values == EAJ_1.values).all() and (EAJ.values == EAJ_2.values).all(), "Check: Earthquake and JohnCalls are independent, given that Alarm state is observed (e.g. Alarm = False)"



# %% markdown [markdown]
# * Comment: comparing the probabilities obtained from the two cases `Alarm = True` and `Alarm = False`, we see that the probability of John calling when there is an `Alarm` is higher ($P(\text{JohnCalls} = \text{True} \; | \; \text{Alarm} = \text{True}) = 0.90$) than when there is no `Alarm` ringing ($P(\text{JohnCalls} = \text{True} \; | \; \text{Alarm} = \text{False}) = 0.05$).








# %% markdown [markdown]
# ### 2. Evidential Reasoning in the Car Model
# For an evidential model $A \leftarrow B \leftarrow C$, there are two cases:
#   * **Marginal Dependence:** ($B$ unobserved): When $B$ is unobserved, there is an active trail between $A$  and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa).
#   * **Conditional Independence:** ($B$ observed): When $B$ is fixed, there is NO active trail between $A$ and $C$, so they are independent. The probability of $A$ won't influence probability of $C$ (and vice versa) when $B$'s state is observed.

# %% codecell
pgmpyToGraph(carModel)
# %% markdown [markdown]
# ### Case 1: Marginal Dependence (for Evidential Model)
#
# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{Burglary} \longleftarrow \text{Alarm} \longleftarrow
# \text{MaryCalls}}
# $$
# Given that the state of `Alarm` is unobserved, we can make the following equivalent statements:
# * there IS an active trail between `Burglary` and `MaryCalls`.
# * the random variables `Burglary` and `MaryCalls` are dependent.
# * the probability of `Burglary` can influence probability of `MaryCalls` (and vice versa).
#
#
# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{Burglary} \longleftarrow \text{Alarm} \longleftarrow \text{JohnCalls}}
# $$
# Given that the state of `Alarm` is unobserved, we can make the following equivalent statements:
# * there IS an active trail between `Burglary` and `JohnCalls`.
# * the random variables `Burglary` and `JohnCalls` are dependent.
# * the probability of `Burglary` can influence probability of `JohnCalls` (and vice versa).
#
# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{Earthquake} \longleftarrow \text{Alarm} \longleftarrow \text{MaryCalls}}
# $$
# Given that the state of `Alarm` is unobserved, we can make the following equivalent statements:
# * there IS an active trail between `Earthquake` and `MaryCalls`.
# * the random variables `Earthquake` and `MaryCalls` are dependent.
# * the probability of `Earthquake` can influence probability of `MaryCalls` (and vice versa).
#
# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{Earthquake} \longleftarrow \text{Alarm} \longleftarrow \text{JohnCalls}}
# $$
# Given that the state of `Alarm` is unobserved, we can make the following equivalent statements:
# * there IS an active trail between `Earthquake` and `JohnCalls`.
# * the random variables `Earthquake` and `JohnCalls` are dependent.
# * the probability of `Earthquake` can influence probability of `JohnCalls` (and vice versa).


# %% markdown [markdown]
# **Verify:** Using Active Trails
# %% codecell
assert carModel.is_active_trail(start = MaryCalls.var, end = Burglary.var, observed = None)
assert carModel.is_active_trail(start = MaryCalls.var, end = Earthquake.var, observed = None)
assert carModel.is_active_trail(start = JohnCalls.var, end = Burglary.var, observed = None)
assert carModel.is_active_trail(start = JohnCalls.var, end = Earthquake.var, observed = None)

showActiveTrails(model = carModel, variables = [MaryCalls.var, Burglary.var])


# %% markdown [markdown]
# **Verify:** Using Probabilities (example of $B \leftarrow A \leftarrow J$ trail)
# * **NOTE:** Evidential Reasoning For Evidential Model:
# %% codecell
JB: DiscreteFactor = elim.query(variables = [Burglary.var], evidence = None)
print(JB)

# %% codecell
JB_1 = elim.query(variables = [Burglary.var], evidence = {JohnCalls.var:'True'})
print(JB_1)
# %% markdown [markdown]
# Below we see that when `JohnCalls` does not occur and no `Alarm` was observed, there is a lower probability of `Burglary`, compared to when neither `Alarm` nor `JohnCalls` were observed:
# $$
# P(\text{Burglary} = \text{True} \; | \; \text{JohnCalls} = \text{False}) = 0.9937
# $$
# %% codecell
JB_2 = elim.query(variables = [Burglary.var], evidence = {JohnCalls.var:'False'})
print(JB_2)
# %% codecell
assert (JB.values != JB_1.values).all() and (JB.values != JB_2.values).all(), "Check there is dependency between Burglary and JohnCalls, when Alarm state is unobserved "


# %% markdown [markdown]
# **Verify:** Using Probabilities (example of $E \leftarrow A \leftarrow J$ trail)
# * **NOTE:** Evidential Reasoning For Evidential Model:
# %% codecell
JE: DiscreteFactor = elim.query(variables = [Earthquake.var], evidence = None)
print(JE)

# %% codecell
JE_1 = elim.query(variables = [Earthquake.var], evidence = {JohnCalls.var:'True'})
print(JE_1)
# %% markdown [markdown]
# Below we see that when `JohnCalls` does not occur and no `Alarm` was not observed, there is a lower probability of `Earthquake`, compared to when John did call and `Alarm` was not observed:
# $$
# P(\text{Earthquake} = \text{True} \; | \; \text{JohnCalls} = \text{False}) = 0.0019
# $$
# %% codecell
JE_2 = elim.query(variables = [Earthquake.var], evidence = {JohnCalls.var:'False'})
print(JE_2)
# %% codecell
assert (JE.values != JE_1.values).all() and (JE.values != JE_2.values).all(), "Check there is dependency between " \
                                                                              "JohnCalls and Earthquake, when Alarm " \
                                                                              "state is unobserved "




# %% markdown [markdown]
# ### Case 2: Conditional Independence (for Evidential Model)
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (observed): }\;\;\;\;\;\;\;\; \text{Burglary} \; \bot \; \text{MaryCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Burglary` and `MaryCalls`. In other words, `Burglary` and `MaryCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Burglary` won't influence probability of `MaryCalls` (and vice versa) when `Alarm`'s state is observed.
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (observed): }\;\;\;\;\;\;\;\; \text{Burglary} \; \bot \; \text{JohnCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Burglary` and `JohnCalls`. In other words, `Burglary` and `JohnCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Burglary` won't influence probability of `JohnCalls` (and vice versa) when `Alarm`'s state is observed.
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (observed): }\;\;\;\;\;\;\;\; \text{Earthquake} \; \bot \; \text{MaryCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Earthquake` and `MaryCalls`. In other words, `Earthquake` and `MaryCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Earthquake` won't influence probability of `MaryCalls` (and vice versa) when `Alarm`'s state is observed.
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (observed): }\;\;\;\;\;\;\;\; \text{Earthquake} \; \bot \; \text{JohnCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Earthquake` and `JohnCalls`. In other words, `Earthquake` and `JohnCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Earthquake` won't influence probability of `JohnCalls` (and vice versa) when `Alarm`'s state is observed.
# %% markdown [markdown]
# **Verify:** Using Active Trails
# %% codecell
assert not carModel.is_active_trail(start = MaryCalls.var, end = Burglary.var, observed = Alarm.var)
assert not carModel.is_active_trail(start = MaryCalls.var, end = Earthquake.var, observed = Alarm.var)
assert not carModel.is_active_trail(start = JohnCalls.var, end = Burglary.var, observed = Alarm.var)
assert not carModel.is_active_trail(start = JohnCalls.var, end = Earthquake.var, observed = Alarm.var)

showActiveTrails(model = carModel, variables = [JohnCalls.var, Earthquake.var], observed = Alarm.var)

# %% markdown [markdown]
# **Verify:** Using Independencies (just the $(B \; \bot \; M \; | \; A)$ independence)
# %% codecell
indepBurglary: IndependenceAssertion = Independencies([Burglary.var, MaryCalls.var, [Alarm.var]]).get_assertions()[0]; indepBurglary

indepMary: IndependenceAssertion = Independencies([MaryCalls.var, Burglary.var, [Alarm.var]]).get_assertions()[0]; indepMary

# Using the fact that closure returns independencies that are IMPLIED by the current independencies:
assert (str(indepMary) == '(MaryCalls _|_ Burglary | Alarm)' and
        indepMary in carModel.local_independencies(MaryCalls.var).closure().get_assertions()),  \
        "Check 1: Burglary and MaryCalls are independent once conditional on Alarm"

assert (str(indepBurglary) == '(Burglary _|_ MaryCalls | Alarm)' and
        indepBurglary in carModel.local_independencies(MaryCalls.var).closure().get_assertions()), \
        "Check 2: Burglary and MaryCalls are independent once conditional on Alarm"

carModel.local_independencies(MaryCalls.var).closure()

# %% codecell
# See: MaryCalls and Burglary are conditionally independent on Alarm:
indepSynonymTable(model = alarmModel_brief, queryNode = 'M')



# %% markdown [markdown]
# **Verify:** Using Probabilities Method (just the $(E \; \bot \; J \; | \; A)$ independence)
# %% markdown [markdown]
# The probability below is:
# $$
# \begin{array}{ll}
# P(\text{Earthquake} = \text{True} \; | \; \text{Alarm} = \text{True})
# &= P(\text{Earthquake} = \text{True} \; | \; \text{Alarm} = \text{True} \; \cap \; \text{JohnCalls} = \text{True})  \\
# &= P(\text{Earthquake} = \text{True} \; | \; \text{Alarm} = \text{True} \; \cap \; \text{JohnCalls} = \text{False}) \\
# &= 0.02
# \end{array}
# $$
# %% codecell

# Case 1: Alarm = True
JAE: DiscreteFactor = elim.query(variables = [Earthquake.var], evidence = {Alarm.var: 'True'})
JAE_1 = elim.query(variables = [Earthquake.var], evidence = {Alarm.var: 'True', JohnCalls.var:'True'})
JAE_2 = elim.query(variables = [Earthquake.var], evidence = {Alarm.var: 'True', JohnCalls.var:'False'})

assert (JAE.values == JAE_1.values).all() and (JAE.values == JAE_2.values).all(), "Check: there is independence between Earthquake and JohnCalls when Alarm state is observed (Alarm = True)"

print(JAE)
# %% markdown [markdown]
# The probability below is:
# $$
# \begin{array}{ll}
# P(\text{Earthquake} = \text{True} \; | \; \text{Alarm} = \text{False})
# &= P(\text{Earthquake} = \text{True} \; | \; \text{Alarm} = \text{False} \; \cap \; \text{JohnCalls} = \text{True})  \\
# &= P(\text{Earthquake} = \text{True} \; | \; \text{Alarm} = \text{False} \; \cap \; \text{JohnCalls} = \text{False}) \\
# &= 0.0017
# \end{array}
# $$
# %% codecell
# Case 2: Alarm = False
JAE: DiscreteFactor = elim.query(variables = [Earthquake.var], evidence = {Alarm.var: 'False'})
JAE_1 = elim.query(variables = [Earthquake.var], evidence = {Alarm.var: 'False', JohnCalls.var:'True'})
JAE_2 = elim.query(variables = [Earthquake.var], evidence = {Alarm.var: 'False', JohnCalls.var:'False'})

assert (JAE.values == JAE_1.values).all() and (JAE.values == JAE_2.values).all(), "Check: there is independence between Earthquake and JohnCalls when Alarm state is observed (Alarm = False)"

print(JAE)







# %% markdown [markdown]
# ### 3. Inter-Causal (?) Reasoning in the Car Model
# For a common cause model $A \leftarrow B \rightarrow C$, there are two cases:
#   * **Marginal Dependence:** ($B$ unknown): When $B$ is unknown / unobserved, there is an active trail between $A$ and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa) when information about $B$'s state is unknown.
#   * **Conditional Independence:** ($B$ fixed): When $B$ is fixed, there is NO active trail between $A$ and $C$, so they are independent. The probability of $A$ won't influence probability of $C$ (and vice versa) when $B$'s state is observed.

# %% codecell
pgmpyToGraph(carModel)
# %% markdown [markdown]
# ### Case 1: Marginal Dependence (for Evidential Model)
#
# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{JohnCalls} \longleftarrow \text{Alarm} \longleftarrow \text{MaryCalls}}
# $$
#
# When the parent node `Alarm` is unknown / unobserved, there IS an active trail between `JohnCalls` and `MaryCalls`. In other words, there is a dependence between `JohnCalls` and `MaryCalls` when `Alarm` is unobserved. This means the probability of `JohnCalls` can influence probability of `MaryCalls` (and vice versa) when information about `Alarm`'s state is unknown.

# %% markdown [markdown]
# **Verify:** Using Active Trails
# %% codecell
assert carModel.is_active_trail(start = JohnCalls.var, end = MaryCalls.var, observed = None)

showActiveTrails(model = carModel, variables = [JohnCalls.var, MaryCalls.var])


# %% markdown [markdown]
# **Verify:** Using Probabilities
# * **NOTE:** Inter-Causal Reasoning For Common Cause Model:
# %% codecell
JM: DiscreteFactor = elim.query(variables = [MaryCalls.var], evidence = None)
print(JM)
# %% markdown [markdown]
# Below we see that when `JohnCalls` and no `Alarm` was observed, there is a higher probability of `MaryCalls`, compared to when no `JohnCalls` nor `Alarm` were observed:
# $$
# P(\text{MaryCalls} = \text{True} \; | \; \text{JohnCalls} = \text{True}) = 0.6975
# $$
# %% codecell
JM_1 = elim.query(variables = [MaryCalls.var], evidence = {JohnCalls.var:'True'})
print(JM_1)
# %% markdown [markdown]
# Below we see that when `JohnCalls` does not occur and no `Alarm` was observed, there is a lower probability of `MaryCalls`, compared to when `JohnCalls` and `Alarm` was not observed:
# $$
# P(\text{MaryCalls} = \text{True} \; | \; \text{JohnCalls} = \text{False}) = 0.4369
# $$
# %% codecell
JM_2 = elim.query(variables = [MaryCalls.var], evidence = {JohnCalls.var:'False'})
print(JM_2)
# %% codecell
assert (JM.values != JM_1.values).all() and (JM.values != JM_2.values).all(), "Check: Marginal Dependence: there is dependency between MaryCalls and JohnCalls, when Alarm state is unobserved "


# %% markdown [markdown]
# ### Case 2: Conditional Independence (for Common Cause Model)
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (observed): }\;\;\;\;\;\;\;\; \text{JohnCalls} \; \bot \; \text{MaryCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `JohnCalls` and `MaryCalls`. In other words, `JohnCalls` and `MaryCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `JohnCalls` won't influence probability of `MaryCalls` (and vice versa) when `Alarm`'s state is observed.
#
# %% markdown [markdown]
# **Verify:** Using Active Trails
# %% codecell
assert not carModel.is_active_trail(start = JohnCalls.var, end = MaryCalls.var, observed = Alarm.var)

showActiveTrails(model = carModel, variables = [JohnCalls.var, MaryCalls.var], observed = Alarm.var)

# %% markdown [markdown]
# **Verify:** Using Independencies
# %% codecell
indepJohn: IndependenceAssertion = Independencies([JohnCalls.var, MaryCalls.var, [Alarm.var]]).get_assertions()[0]; indepJohn

indepMary: IndependenceAssertion = Independencies([MaryCalls.var, JohnCalls.var, [Alarm.var]]).get_assertions()[0]; indepMary


# Using the fact that closure returns independencies that are IMPLIED by the current independencies:
assert (str(indepMary) == '(MaryCalls _|_ JohnCalls | Alarm)' and
        indepMary in carModel.local_independencies(MaryCalls.var).closure().get_assertions()),  \
        "Check 1: MaryCalls and JohnCalls are independent once conditional on Alarm"

carModel.local_independencies(MaryCalls.var).closure()
# %% codecell
assert (str(indepJohn) == '(JohnCalls _|_ MaryCalls | Alarm)' and
        indepJohn in carModel.local_independencies(JohnCalls.var).closure().get_assertions()), \
        "Check 2: JohnCalls and MaryCalls are independent once conditional on Alarm"

carModel.local_independencies(MaryCalls.var).closure()

# %% codecell
# See: MaryCalls and JohnCalls are conditionally independent on Alarm:
indepSynonymTable(model = alarmModel_brief, queryNode = 'M')
# %% codecell
indepSynonymTable(model = alarmModel_brief, queryNode = 'J')


# %% markdown [markdown]
# **Verify:** Using Probabilities Method

# %% markdown [markdown]
# The probability below is:
# $$
# \begin{array}{ll}
# P(\text{MaryCalls} = \text{True} \; | \; \text{Alarm} = \text{True})
# &= P(\text{MaryCalls} = \text{True} \; | \; \text{Alarm} = \text{True} \; \cap \; \text{JohnCalls} = \text{True})  \\
# &= P(\text{MaryCalls} = \text{True} \; | \; \text{Alarm} = \text{True} \; \cap \; \text{JohnCalls} = \text{False}) \\
# &= 0.7
# \end{array}
# $$
# %% codecell

# Case 1: Alarm = True
JAM: DiscreteFactor = elim.query(variables = [MaryCalls.var], evidence = {Alarm.var: 'True'})
JAM_1 = elim.query(variables = [MaryCalls.var], evidence = {Alarm.var: 'True', JohnCalls.var:'True'})
JAM_2 = elim.query(variables = [MaryCalls.var], evidence = {Alarm.var: 'True', JohnCalls.var:'False'})

assert (JAM.values == JAM_1.values).all() and (JAM.values == JAM_2.values).all(), "Check: there is independence between MaryCalls and JohnCalls when Alarm state is observed (Alarm = True)"

print(JAM)
# %% markdown [markdown]
# The probability below is:
# $$
# \begin{array}{ll}
# P(\text{MaryCalls} = \text{True} \; | \; \text{Alarm} = \text{False})
# &= P(\text{MaryCalls} = \text{True} \; | \; \text{Alarm} = \text{False} \; \cap \; \text{JohnCalls} = \text{True})  \\
# &= P(\text{MaryCalls} = \text{True} \; | \; \text{Alarm} = \text{False} \; \cap \; \text{JohnCalls} = \text{False}) \\
# &= 0.7
# \end{array}
# $$
# %% codecell

# Case 2: Alarm = False
JAM: DiscreteFactor = elim.query(variables = [MaryCalls.var], evidence = {Alarm.var: 'False'})
JAM_1 = elim.query(variables = [MaryCalls.var], evidence = {Alarm.var: 'False', JohnCalls.var:'True'})
JAM_2 = elim.query(variables = [MaryCalls.var], evidence = {Alarm.var: 'False', JohnCalls.var:'False'})

assert (JAM.values == JAM_1.values).all() and (JAM.values == JAM_2.values).all(), "Check: there is independence between MaryCalls and JohnCalls when Alarm state is observed (Alarm = False)"

print(JAM)

# %% codecell
# Symmetry:
MAJ: DiscreteFactor = elim.query(variables = [JohnCalls.var], evidence = {Alarm.var: 'False'})
MAJ_1 = elim.query(variables = [JohnCalls.var], evidence = {Alarm.var: 'False', MaryCalls.var:'True'})
MAJ_2 = elim.query(variables = [JohnCalls.var], evidence = {Alarm.var: 'False', MaryCalls.var:'False'})

assert (MAJ.values == MAJ_1.values).all() and (MAJ.values == MAJ_2.values).all(), "Check: there is independence between MaryCalls and JohnCalls when Alarm state is observed (Alarm = False)"

print(MAJ)




# %% markdown [markdown]
# ### 4. Inter-Causal Reasoning in the Car Model
# For a common evidence model $A \rightarrow B \leftarrow C$, there are two cases:
#   * **Marginal Independence:** ($B$ unknown): When $B$ is unknown / unobserved, there is NO active trail between $A$ and $C$; they are independent. The probability of $A$ won't influence probability of $C$ (and vice versa) when $B$'s state is unknown.
#   * **Conditional Dependence:** ($B$ fixed): When $B$ is fixed, there IS an active trail between $A$ and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa) when information about $B$ is observed / fixed.


# %% codecell
pgmpyToGraph(carModel)
# %% markdown [markdown]
# ### Case 1: Marginal Independence (for Common Evidence Model)
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (observed): }\;\;\;\;\;\;\;\; \text{Burglary} \; \bot \; \text{Earthquake} \; | \; \text{Alarm}}
# $$
#
# When the effect node `Alarm` is unknown / unobserved, there is NO an active trail between `Burglary` and `Earthquake`. In other words, there is a local marginal independence between `Burglary` and `Earthquake` when `Alarm` is unobserved. This means the probability of `Burglary` won't influence the probability of `Earthquake` (and vice versa) when `Alarm`'s state is unknown.
#
# %% markdown [markdown]
# **Verify:** Using Active Trails
# %% codecell
assert not carModel.is_active_trail(start = Burglary.var, end = Earthquake.var, observed = None)

showActiveTrails(model = carModel, variables = [Burglary.var, Earthquake.var])

# %% markdown [markdown]
# **Verify:** Using Independencies
# %% codecell
indepBurgEarth = Independencies([Burglary.var, Earthquake.var])

assert indepBurgEarth == carModel.local_independencies(Burglary.var), 'Check 1: Burglary and Earthquake are marginally independent'

assert indepBurgEarth == carModel.local_independencies(Earthquake.var), 'Check 2: Burglary and Earthquake are marginally independent'


# See: MaryCalls and Burglary are marginally independent :
print(indepSynonymTable(model = carModel, queryNode = Burglary.var))
print(indepSynonymTable(model = carModel, queryNode = Earthquake.var))


# %% markdown [markdown]
# **Verify:** Using Probabilities Method
# %% markdown [markdown]
# The probability below is:
# $$
# \begin{array}{ll}
# P(\text{Earthquake} = \text{True})
# &= P(\text{Earthquake} = \text{True} \; | \; \text{Burglary} = \text{True})  \\
# &= P(\text{Earthquake} = \text{True} \; | \; \text{Burglary} = \text{False}) \\
# &= 0.7
# \end{array}
# $$
# %% codecell

BE: DiscreteFactor = elim.query(variables = [Earthquake.var], evidence = None)
BE_1 = elim.query(variables = [Earthquake.var], evidence = {Burglary.var:'True'})
BE_2 = elim.query(variables = [Earthquake.var], evidence = {Burglary.var: 'False'})

# Using np.allclose instead of exact equals sign (there must be some numerical inconsistency ... otherwise they wouldn't be different at all! BAE.values[0] = 0.0019999999 while BAE_1.values[0] = 0.002)
assert np.allclose(BE.values, BE_1.values) and np.allclose(BE.values, BE_2.values), "Check: there is marginal independence between Earthquake and Burglary when Alarm state is NOT observed"

print(BE)



# %% markdown [markdown]
# ### Case 2: Conditional Dependence (for Common Evidence Model)

# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{Burglary} \longleftarrow \text{Alarm} \longrightarrow \text{Earthquake}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there IS an active trail between `Burglary` and `Earthquake`. In other words, `Burglary` and `Earthquake` are dependent when `Alarm`'s state is observed. This means the probability of `Burglary` can influence probability of `Earthquake` (and vice versa) when `Alarm`'s state is observed.

# %% markdown [markdown]
# **Verify:** Using Active Trails
# %% codecell
assert carModel.is_active_trail(start = Burglary.var, end = Earthquake.var, observed = Alarm.var)

showActiveTrails(model = carModel, variables = [Burglary.var, Earthquake.var], observed = Alarm.var)


# %% markdown [markdown]
# **Verify:** Using Probabilities
# * **NOTE:** Inter-Causal Reasoning For Common Evidence Model:
# %% codecell

# Case 1: Alarm = True
BAE: DiscreteFactor = elim.query(variables = [Earthquake.var], evidence = {Alarm.var: 'True'})
print(BAE)
# %% codecell
BAE_1: DiscreteFactor = elim.query(variables = [Earthquake.var], evidence = {Burglary.var:'True', Alarm.var: 'True'})
print(BAE_1)
# %% markdown [markdown]
# Below we see that when there was no`Burglary` (cause) and `Alarm` rang, there is a higher probability of `Earthquake` (other cause) compared to when there was a `Burglary` and `Alarm` rang:
# $$
# P(\text{Earthquake} = \text{True} \; | \; \text{Burglary} = \text{False} \; \cap \; \text{Alarm} = \text{True}) = 0.3676
# $$
# * NOTE: This is like in page 41 of Korb book (inverse of "explaining away")
# %% codecell
BAE_2: DiscreteFactor = elim.query(variables = [Earthquake.var], evidence = {Burglary.var:'False', Alarm.var: 'True'})
print(BAE_2)

# %% codecell
assert (BAE_2.values != BAE.values).all(), 'Check: there is dependency between Earthquake and Burglary when Alarm state is observed (True)'

# %% markdown [markdown]
# The probability below is:
# $$
# P(\text{Earthquake} = \text{True} \; | \;\text{Alarm} = \text{False}) = 0.0017
# $$
# %% codecell
# Case 2: Alarm = False
BAE: DiscreteFactor = elim.query(variables = [Earthquake.var], evidence = {Alarm.var: 'False'})
print(BAE)
# %% markdown [markdown]
# The probability below is:
# $$
# P(\text{Earthquake} = \text{True} \; | \; \text{Burglary} = \text{True} \; \cap \; \text{Alarm} = \text{False}) = 0.0017
# $$
# %% codecell
BAE_1: DiscreteFactor = elim.query(variables = [Earthquake.var], evidence = {Burglary.var:'True', Alarm.var: 'False'})
print(BAE_1)
# %% markdown [markdown]
# Below we see that when there was no `Burglary` (cause) and `Alarm` did not ring, there is a lower probability of `Earthquake` (other cause) compared to when there was a `Burglary` and `Alarm` didn't ring:
# $$
# P(\text{Earthquake} = \text{True} \; | \; \text{Burglary} = \text{False} \; \cap \; \text{Alarm} = \text{False}) = 0.0014
# $$
# %% codecell
BAE_2: DiscreteFactor = elim.query(variables = [Earthquake.var], evidence = {Burglary.var:'False', Alarm.var: 'False'})
print(BAE_2)
# %% codecell
assert (BAE_2.values != BAE.values).all(), 'Check: there is dependency between Earthquake and Burglary when Alarm state is observed (False)'
