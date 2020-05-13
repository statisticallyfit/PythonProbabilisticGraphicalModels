# %% markdown [markdown]
# [Source for tutorial](https://github.com/pgmpy/pgmpy/blob/dev/examples/Alarm.ipynb)
#
#
# # Alarm Bayesian Network
# Creating the Alarm Bayesian network using pgmpy and doing some simple queries (mentioned in Bayesian Artificial Intelligence, Section 2.5.1: )

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

dataPath: str = curPath + "data/"

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
# %% markdown [markdown]
# ## Problem Statement: 2.5.1 Earthquake Alarm
# **Example statement:** You have a new burglar alarm installed. It reliably detects burglary, but also responds to minor earthquakes. Two neighbors, John and Mary, promise to call the police when they hear the alarm. John always calls when he hears the alarm, but sometimes confuses the alarm with the phone ringing and calls then also. On the other hand, Mary likes loud music and sometimes doesn't hear the alarm. Given evidence about who has and hasn't called, you'd like to estimate the probability of a burglary alarm (from Pearl (1988)).
#
# %% codecell

import collections

# Creating some names for the random variables (nodes) in the graph, to clarify meaning.

# Create named tuple class with names "Names" and "Objects"
RandomVariable = collections.namedtuple("RandomVariable", ["var", "states"])

Burglary = RandomVariable(var = "Burglary", states = ['True', 'False'])
Earthquake = RandomVariable(var = "Earthquake", states = ['True', 'False'])
Alarm = RandomVariable(var = "Alarm", states = ['True', 'False'])
JohnCalls = RandomVariable(var = "JohnCalls", states = ['True', 'False'])
MaryCalls = RandomVariable(var = "MaryCalls", states = ['True', 'False'])

#Burglary: Dict[Variable, List[State]] = {Burglary.var: ['True', 'False']}
#Earthquake: Dict[Variable, List[State]] = {Earthquake.var: ['True', 'False']}
#Alarm: Dict[Variable, List[State]] = {Alarm.var: ['True', 'False']}
#JohnCalls: Dict[Variable, List[State]] = {JohnCalls.var: ['True', 'False']}
#MaryCalls: Dict[Variable, List[State]] = {MaryCalls.var: ['True', 'False']}


# Defining the network structure:
#alarmModel: BayesianModel = BayesianModel([(Burglary.variable, Alarm.variable),
#                                           (Earthquake.variable, Alarm.variable),
#                                           (Alarm.variable, JohnCalls.variable),
#                                           (Alarm.variable, MaryCalls.variable)])

alarmModel: BayesianModel = BayesianModel([(Burglary.var, Alarm.var),
                                          (Earthquake.var, Alarm.var),
                                          (Alarm.var, JohnCalls.var),
                                          (Alarm.var, MaryCalls.var)])

# Defining parameters using CPT

cpdBurglary: TabularCPD = TabularCPD(variable = Burglary.var, variable_card = len(Burglary.states),
                                     values = [[0.001, 0.999]],
                                     state_names = {Burglary.var : Burglary.states})

print(cpdBurglary)

# %% codecell
cpdEarthquake: TabularCPD = TabularCPD(variable = Earthquake.var,
                                       variable_card = len(Earthquake.states),
                                       values = [[0.002, 0.998]],
                                       state_names = {Earthquake.var : Earthquake.states})

print(cpdEarthquake)

# %% codecell
cpdAlarm: TabularCPD = TabularCPD(variable = Alarm.var,
                                  variable_card = len(Alarm.states),
                                  values = [[0.95, 0.94, 0.29, 0.001],
                                            [0.05, 0.06, 0.71, 0.999]],
                                  evidence = [Burglary.var, Earthquake.var],
                                  evidence_card = [len(Burglary.states), len(Earthquake.states)],
                                  state_names = {Alarm.var: Alarm.states,
                                                 Burglary.var : Burglary.states,
                                                 Earthquake.var : Earthquake.states})
print(cpdAlarm)


#def toDict(tup: RandomVariable):
#    return {tup.var : tup.states}

# %% codecell
cpdJohnCalls: TabularCPD = TabularCPD(variable = JohnCalls.var, variable_card = len(JohnCalls.states),
                                      values = [[0.90, 0.05],
                                                [0.10, 0.95]],
                                      evidence = [Alarm.var], evidence_card = [len(Alarm.states)],
                                      state_names = {JohnCalls.var : JohnCalls.states,
                                                     Alarm.var : Alarm.states})
print(cpdJohnCalls)

# %% codecell
cpdMaryCalls: TabularCPD = TabularCPD(variable = MaryCalls.var, variable_card = len(MaryCalls.states),
                                      values = [[0.70, 0.01],
                                                [0.30, 0.99]],
                                      evidence = [Alarm.var], evidence_card = [len(Alarm.states)],
                                      state_names = {MaryCalls.var: MaryCalls.states,
                                                     Alarm.var : Alarm.states})
print(cpdMaryCalls)

# %% codecell
alarmModel.add_cpds(cpdBurglary, cpdEarthquake, cpdAlarm, cpdJohnCalls, cpdMaryCalls)

assert alarmModel.check_model()


# %% markdown [markdown]
# Making a brief-name version for viewing clarity, in tables:
# %% codecell
alarmModel_brief: BayesianModel = BayesianModel([(Burglary.var[0], Alarm.var[0]),
                                          (Earthquake.var[0], Alarm.var[0]),
                                          (Alarm.var[0], JohnCalls.var[0]),
                                          (Alarm.var[0], MaryCalls.var[0])])

# Defining parameters using CPT

cpdBurglary: TabularCPD = TabularCPD(variable = Burglary.var[0], variable_card = len(Burglary.states),
                                     values = [[0.001, 0.999]],
                                     state_names = {Burglary.var[0] : Burglary.states})


cpdEarthquake: TabularCPD = TabularCPD(variable = Earthquake.var[0],
                                       variable_card = len(Earthquake.states),
                                       values = [[0.002, 0.998]],
                                       state_names = {Earthquake.var[0] : Earthquake.states})


cpdAlarm: TabularCPD = TabularCPD(variable = Alarm.var[0],
                                  variable_card = len(Alarm.states),
                                  values = [[0.95, 0.94, 0.29, 0.001],
                                            [0.05, 0.06, 0.71, 0.999]],
                                  evidence = [Burglary.var[0], Earthquake.var[0]],
                                  evidence_card = [len(Burglary.states), len(Earthquake.states)],
                                  state_names = {Alarm.var[0]: Alarm.states,
                                                 Burglary.var[0] : Burglary.states,
                                                 Earthquake.var[0] : Earthquake.states})

cpdJohnCalls: TabularCPD = TabularCPD(variable = JohnCalls.var[0], variable_card = len(JohnCalls.states),
                                      values = [[0.90, 0.05],
                                                [0.10, 0.95]],
                                      evidence = [Alarm.var[0]], evidence_card = [len(Alarm.states)],
                                      state_names = {JohnCalls.var[0] : JohnCalls.states,
                                                     Alarm.var[0] : Alarm.states})

cpdMaryCalls: TabularCPD = TabularCPD(variable = MaryCalls.var[0], variable_card = len(MaryCalls.states),
                                      values = [[0.70, 0.01],
                                                [0.30, 0.99]],
                                      evidence = [Alarm.var[0]], evidence_card = [len(Alarm.states)],
                                      state_names = {MaryCalls.var[0]: MaryCalls.states,
                                                     Alarm.var[0] : Alarm.states})

alarmModel_brief.add_cpds(cpdBurglary, cpdEarthquake, cpdAlarm, cpdJohnCalls, cpdMaryCalls)

assert alarmModel_brief.check_model()


# %% codecell
pgmpyToGraphCPD(model = alarmModel, shorten = False)


# %% markdown [markdown]
# ## 1/ Independencies of the Alarm Model
# %% codecell
alarmModel.local_independencies(Burglary.var)
# %% codecell
alarmModel.local_independencies(Earthquake.var)
# %% codecell
alarmModel.local_independencies(Alarm.var)
# %% codecell
print(alarmModel.local_independencies(MaryCalls.var))

indepSynonymTable(model = alarmModel, queryNode = MaryCalls.var)
# %% codecell
print(alarmModel.local_independencies(JohnCalls.var))

indepSynonymTable(model = alarmModel, queryNode = JohnCalls.var)
# %% codecell
alarmModel.get_independencies()



# %% codecell
# TODO say direct dependency assumptions (from Korb book)

pgmpyToGraph(alarmModel)
# %% markdown [markdown]
# ### Study: Independence Maps (I-Maps)
# * **Markov Assumption:** Bayesian networks require the assumption of **Markov Property**: that there are no direct dependencies in the system being modeled, which are not already explicitly shown via arcs. (In the earthquake example, this translates to saying there is no way for an `Earthquake` to influence `MaryCalls` except by way of the `Alarm`.  There is no **hidden backdoor** from  `Earthquake` to `MaryCalls`).
# * **I-maps:** Bayesian networks which have this **Markov property** are called **Independence-maps** or **I-maps**, since every independence suggested by the lack of an arc is actual a valid, real independence in the system.
#
# Source: Korb book, Bayesian Artificial Intelligence (section 2.2.4)
# %% markdown [markdown]
# ### Example 1: I-map
# Testing meaning of an **I-map** using a simple student example
# %% codecell
Difficulty: RandomVariable = RandomVariable(var = 'diff', states = ['Easy', 'Hard'])
Grade: RandomVariable = RandomVariable(var = 'grade', states = ['Good', 'Medium', 'Bad'])
Intelligence: RandomVariable = RandomVariable(var = 'intel', states = ['Dumb', 'Average', 'Intelligent'])


gradeModel = BayesianModel([(Difficulty.var, Grade.var), (Intelligence.var, Grade.var)])

diff_cpd = TabularCPD(variable = Difficulty.var, variable_card = len(Difficulty.states),
                      values = [[0.2], [0.8]])

intel_cpd = TabularCPD(variable = Intelligence.var, variable_card = len(Intelligence.states),
                       values = [[0.5], [0.3], [0.2]])

grade_cpd = TabularCPD(variable = Grade.var, variable_card = len(Grade.states),
                       values = [[0.1,0.1,0.1,0.1,0.1,0.1],
                                    [0.1,0.1,0.1,0.1,0.1,0.1],
                                    [0.8,0.8,0.8,0.8,0.8,0.8]],
                       evidence=[Difficulty.var, Intelligence.var],
                       evidence_card=[len(Difficulty.states), len(Intelligence.states)])

gradeModel.add_cpds(diff_cpd, intel_cpd, grade_cpd)


pgmpyToGraphCPD(gradeModel)

# %% markdown [markdown]
# Showing two ways of creating the `JointProbabilityDistribution`    : (1) by feeding in values manually, or (2) by using `reduce` over the `TabularCPD`s or `DiscreteFactor`s.
# %% codecell
# Method 1: Creating joint distribution manually, by feeding in the calculated values:
jpdValues = [0.01, 0.01, 0.08, 0.006, 0.006, 0.048, 0.004, 0.004, 0.032,
           0.04, 0.04, 0.32, 0.024, 0.024, 0.192, 0.016, 0.016, 0.128]

JPD = JointProbabilityDistribution(variables = [Difficulty.var , Intelligence.var, Grade.var],
                                   cardinality = [len(Difficulty.states), len(Intelligence.states), len(Grade.states)], values = jpdValues)

print(JPD)

# %% markdown [markdown]
# Showing if small student model is I-map:
# %% codecell
from src.utils.NetworkUtil import *


# Method 2: creating the JPD by multiplying over the TabularCPDs (as per formula in page 16 of pgmpy book, Ankur Ankan)
gradeJPDFactor: DiscreteFactor = DiscreteFactor(variables = JPD.variables, cardinality =  JPD.cardinality, values = JPD.values)
gradeJPD: JointProbabilityDistribution = jointDistribution(gradeModel)


assert gradeModel.is_imap(JPD = JPD), "Check: using JPD to verify the graph is an independence-map: means no hidden backdoors between nodes and no way for variables to influence others except by one path"

assert gradeJPD == gradeJPDFactor, "Check: joint distribution is the same as multiplying the cpds"

# %% markdown [markdown]
# Grade model's `JointProbabilityDistribution` over all variables:
# %% codecell
print(gradeJPD)

# %% markdown [markdown]
# Checking if alarm model is I-map:
# %% codecell
alarmJPD: JointProbabilityDistribution = jointDistribution(alarmModel_brief)

assert not alarmModel_brief.is_imap(JPD = alarmJPD)





# %% markdown [markdown]
# ## 3/ Joint Distribution Represented by the Bayesian Network
# Computing the Joint Distribution from the Bayesian Network, `model`:
#
# From the **chain rule of probability (also called and rule):**
# $$
# P(A, B) = P(B) \cdot P(A \; | \; B)
# $$
# Now in this case for the `alarmModel`:
# $$
# \begin{array}{ll}
# P(J, M, A, E, B)
# &= P(J \; | \; M, A, E, B) \cdot P(J, M, A, E, B) \\
# &= P(J \; | \; M, A, E, B) \cdot {\color{cyan} (} P(M \; | \; A,E,B) \cdot P(A,E,B) {\color{cyan} )} \\
# &=  P(J \; | \; M, A, E, B) \cdot  P(M \; | \; A,E,B) \cdot {\color{cyan} (}P(A \; | \; E,B) \cdot P(E, B){\color{cyan} )} \\
# &= P(J \; | \; M, A, E, B) \cdot  P(M \; | \; A,E,B) \cdot P(A \; | \; E,B) \cdot {\color{cyan} (}P(E \; | \; B) \cdot P(B){\color{cyan} )} \\
# \end{array}
# $$
# %% codecell
probChainRule(['J','M','A','E','B'])
# %% markdown [markdown]
# Alarm model's `JointProbabilityDistribution` over all variables
# %% codecell
print(alarmJPD)




# %% markdown [markdown]
# ## 4/ Inference in Bayesian Alarm Model
#
# Now let us do inference in a  Bayesian model and predict values using this model over new data points for ML tasks.
#
# ### 1. Causal Reasoning in the Alarm Model
# For a causal model $A \rightarrow B \rightarrow C$, there are two cases:
#   * **Marginal Dependence:** ($B$ unknown): When $B$ is unknown / unobserved, there is an active trail between $A$ and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa).
#   * **Conditional Independence:** ($B$ fixed): When $B$ is fixed, there is NO active trail between $A$ and $C$, so they are independent, which means the probability of $A$ won't influence probability of $C$ (and vice versa).


# %% codecell
pgmpyToGraph(alarmModel)
# %% markdown [markdown]
# ### Case 1: Marginal Dependence (for Causal Model)
# For a causal model $A \rightarrow B \rightarrow C$, when the state of the middle node $B$ is unobserved, then an active trail is created between the nodes, namely the active trail is $A \rightarrow B \rightarrow C$. Information can now flow from node $A$ to node $C$ via node $B$. This implies there is a dependency between nodes $A$ and $C$, so the probability of $A$ taking on any of its states can influence the probability of $C$ taking on any of its states. This is called **marginal dependence** We can write this as: $P(A | C) \ne P(A)$
#
# $\color{red}{\text{TODO}}$ left off here trying to refactor the text (continue from sublime notes pg 35 Korb and pg 336 Bayesiabook)
# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{Burglary} \longrightarrow \text{Alarm} \longrightarrow \text{MaryCalls}}
# $$
#
# Given that the state of `Alarm` is unobserved, we can state the following equivalent statements:
# * there IS an active trail between `Burglary` and `MaryCalls`.
# * the random variables `Burglary` and `MaryCalls` are dependent.
# * the probability of `Burglary` can influence probability of `MaryCalls` (and vice versa).
#
# Similarly, the same kinds of statements can be made for the other causal chain pathways in the graph:
#
# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{Burglary} \longrightarrow \text{Alarm} \longrightarrow \text{JohnCalls}}
# $$
# Given that the state of `Alarm` is unobserved, we can state the following equivalent statements:
# * there IS an active trail between `Burglary` and `JohnCalls`.
# * the random variables `Burglary` and `JohnCalls` are dependent.
# * the probability of `Burglary` can influence probability of `JohnCalls` (and vice versa).
#
# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{Earthquake} \longrightarrow \text{Alarm} \longrightarrow \text{MaryCalls}}
# $$
# Given that the state of `Alarm` is unobserved, we can state the following equivalent statements:
# * there IS an active trail between `Earthquake` and `MaryCalls`.
# * the random variables `Earthquake` and `MaryCalls` are dependent.
# * the probability of `Earthquake` can influence probability of `MaryCalls` (and vice versa).
#
# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{Earthquake} \longrightarrow \text{Alarm} \longrightarrow \text{JohnCalls}}
# $$
# Given that the state of `Alarm` is unobserved, we can state the following equivalent statements:
# * there IS an active trail between `Earthquake` and `JohnCalls`.
# * the random variables `Earthquake` and `JohnCalls` are dependent.
# * the probability of `Earthquake` can influence probability of `JohnCalls` (and vice versa).
# %% markdown [markdown]
# **Verify:** Using Active Trails
# %% codecell
assert alarmModel.is_active_trail(start = Burglary.var, end = MaryCalls.var, observed = None)
assert alarmModel.is_active_trail(start = Burglary.var, end = JohnCalls.var, observed = None)
assert alarmModel.is_active_trail(start = Earthquake.var, end = MaryCalls.var, observed = None)
assert alarmModel.is_active_trail(start = Earthquake.var, end = JohnCalls.var, observed = None)

showActiveTrails(model = alarmModel, variables = [Burglary.var, MaryCalls.var])

# %% codecell
elim: VariableElimination = VariableElimination(model = alarmModel)

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
# Through the above steps, we observed some probabilities conditional on some states. To present them all in one place, here are the probabilities that `JohnCalls = True, where the last two are conditional on `Burglary.states = ['True', 'False']`.
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
# Through the above steps, we observed some probabilities conditional on some states. To present them all in one place, here are the probabilities that `MaryCalls = True, where the last two are conditional on `Earthquake.states = ['True', 'False']`.
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
# Given that `Alarm`'s state is fixed / observed, we can state the following equivalent statements:
# * there is NO active trail between `Burglary` and `MaryCalls`.
# * `Burglary` and `MaryCalls` are locally independent.
# * the probability of `Burglary` won't influence probability of `MaryCalls` (and vice versa).
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (observed): }\;\;\;\;\;\;\;\; \text{Burglary} \; \bot \; \text{JohnCalls} \; | \; \text{Alarm}}
# $$
# Given that `Alarm`'s state is fixed / observed, we can state the following equivalent statements:
# * there is NO active trail between `Burglary` and `JohnCalls`.
# * `Burglary` and `JohnCalls` are locally independent.
# * the probability of `Burglary` won't influence probability of `JohnCalls` (and vice versa).
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (observed): }\;\;\;\;\;\;\;\; \text{Earthquake} \; \bot \; \text{MaryCalls} \; | \; \text{Alarm}}
# $$
# Given that `Alarm`'s state is fixed / observed, we can state the following equivalent statements:
# * there is NO active trail between `Earthquake` and `MaryCalls`.
# * `Earthquake` and `MaryCalls` are locally independent.
# * the probability of `Earthquake` won't influence probability of `MaryCalls` (and vice versa).
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (observed): }\;\;\;\;\;\;\;\; \text{Earthquake} \; \bot \; \text{JohnCalls} \; | \; \text{Alarm}}
# $$
# Given that `Alarm`'s state is fixed / observed, we can state the following equivalent statements:
# * there is NO active trail between `Earthquake` and `JohnCalls`.
# * `Earthquake` and `JohnCalls` are locally independent.
# * the probability of `Earthquake` won't influence probability of `JohnCalls` (and vice versa).
# %% markdown [markdown]
# **Verify:** Using Active Trails
# %% codecell
assert not alarmModel.is_active_trail(start = Burglary.var, end = MaryCalls.var, observed = Alarm.var)
assert not alarmModel.is_active_trail(start = Burglary.var, end = JohnCalls.var, observed = Alarm.var)
assert not alarmModel.is_active_trail(start = Earthquake.var, end = MaryCalls.var, observed = Alarm.var)
assert not alarmModel.is_active_trail(start = Earthquake.var, end = JohnCalls.var, observed = Alarm.var)

showActiveTrails(model = alarmModel, variables = [Burglary.var, MaryCalls.var], observed = Alarm.var)

# %% markdown [markdown]
# **Verify:** Using Independencies (just the $(B \; \bot \; M \; | \; A)$ independence)
# %% codecell
indepBurglary: IndependenceAssertion = Independencies([Burglary.var, MaryCalls.var, [Alarm.var]]).get_assertions()[0]; indepBurglary

indepMary: IndependenceAssertion = Independencies([MaryCalls.var, Burglary.var, [Alarm.var]]).get_assertions()[0]; indepMary

# Using the fact that closure returns independencies that are IMPLIED by the current independencies:
assert (str(indepMary) == '(MaryCalls _|_ Burglary | Alarm)' and
        indepMary in alarmModel.local_independencies(MaryCalls.var).closure().get_assertions()),  \
        "Check 1: Burglary and MaryCalls are independent once conditional on Alarm"

assert (str(indepBurglary) == '(Burglary _|_ MaryCalls | Alarm)' and
        indepBurglary in alarmModel.local_independencies(MaryCalls.var).closure().get_assertions()), \
        "Check 2: Burglary and MaryCalls are independent once conditional on Alarm"

alarmModel.local_independencies(MaryCalls.var).closure()

# %% codecell
# See: MaryCalls and Burglary are conditionally independent on Alarm:
indepSynonymTable(model = alarmModel_brief, queryNode = 'M')



# %% markdown [markdown]
# **Verify:** Using Probabilities Method (just the $(E \; \bot \; J \; | \; A)$ independence)
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
# Case 2: Alarm = False
EAJ: DiscreteFactor = elim.query(variables = [JohnCalls.var], evidence = {Alarm.var: 'False'})
EAJ_1 = elim.query(variables = [JohnCalls.var], evidence = {Alarm.var: 'False', Earthquake.var:'True'})
EAJ_2 = elim.query(variables = [JohnCalls.var], evidence = {Alarm.var: 'False', Earthquake.var:'False'})

assert (EAJ.values == EAJ_1.values).all() and (EAJ.values == EAJ_2.values).all(), "Check: Earthquake and JohnCalls are independent, given that Alarm state is observed (e.g. Alarm = False)"

print(EAJ)

# %% markdown [markdown]
# Comment: above we see that the probability of John calling when there is an `Alarm` is higher ($P(\text{JohnCalls} = \text{True} \; | \; \text{Alarm} = \text{True}) = 0.90$) than when there is no `Alarm` ringing ($P(\text{JohnCalls} = \text{True} \; | \; \text{Alarm} = \text{False}) = 0.05$).








# %% markdown [markdown]
# ### 2. Evidential Reasoning in the Alarm Model
# For an evidential model $A \leftarrow B \leftarrow C$, there are two cases:
#   * **Marginal Dependence:** ($B$ unknown): When $B$ is unknown / unobserved, there is an active trail between $A$
#   and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa) when information about $B$'s state is unknown.
#   * **Conditional Independence:** ($B$ fixed): When $B$ is fixed, there is NO active trail between $A$ and $C$,
#   so they are independent. The probability of $A$ won't influence probability of $C$ (and vice versa) when $B$'s state is observed.

# %% codecell
pgmpyToGraph(alarmModel)
# %% markdown [markdown]
# ### Case 1: Marginal Dependence (for Evidential Model)
#
# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{Burglary} \longleftarrow \text{Alarm} \longleftarrow
# \text{MaryCalls}}
# $$
#
# When the middle node `Alarm` is unknown / unobserved, there IS an active trail between `Burglary` and `MaryCalls`. In other words, there is a dependence between `Burglary` and `MaryCalls` when `Alarm` is unobserved. This means the probability of `Burglary` can influence probability of `MaryCalls` (and vice versa) when information about `Alarm`'s state is unknown.
#
#
# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{Burglary} \longleftarrow \text{Alarm} \longleftarrow \text{JohnCalls}}
# $$
# When `Alarm`'s state is uknown, there is an active trail or dependency between `Burglary` and `JohnCalls`, so the probability of `Burglary` can influence the probability of `JohnCalls` (and vice versa) when `Alarm`'s state is unknown.
#
# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{Earthquake} \longleftarrow \text{Alarm} \longleftarrow \text{MaryCalls}}
# $$
# When `Alarm`'s state is uknown, there is an active trail or dependency between `Earthquake` and `MaryCalls`, so the probability of `Earthquake` can influence the probability of `MaryCalls` (and vice versa) when `Alarm`'s state is unknown.
#
# $$
# \color{Green}{ \text{Alarm (unobserved): }\;\;\;\;\;\;\;\;\; \text{Earthquake} \longleftarrow \text{Alarm} \longleftarrow \text{JohnCalls}}
# $$
# When `Alarm`'s state is uknown, there is an active trail or dependency between `Earthquake` and `JohnCalls`, so the probability of `Earthquake` can influence the probability of `JohnCalls` (and vice versa) when `Alarm`'s state is unknown.
# %% markdown [markdown]
# **Verify:** Using Active Trails
# %% codecell
assert alarmModel.is_active_trail(start = MaryCalls.var, end = Burglary.var,  observed = None)
assert alarmModel.is_active_trail(start = MaryCalls.var, end = Earthquake.var, observed = None)
assert alarmModel.is_active_trail(start = JohnCalls.var, end = Burglary.var, observed = None)
assert alarmModel.is_active_trail(start = JohnCalls.var, end = Earthquake.var, observed = None)

showActiveTrails(model = alarmModel, variables = [MaryCalls.var, Burglary.var])


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
assert not alarmModel.is_active_trail(start = MaryCalls.var, end = Burglary.var, observed = Alarm.var)
assert not alarmModel.is_active_trail(start = MaryCalls.var, end = Earthquake.var, observed = Alarm.var)
assert not alarmModel.is_active_trail(start = JohnCalls.var, end = Burglary.var, observed = Alarm.var)
assert not alarmModel.is_active_trail(start = JohnCalls.var, end = Earthquake.var, observed = Alarm.var)

showActiveTrails(model = alarmModel, variables = [JohnCalls.var, Earthquake.var], observed = Alarm.var)

# %% markdown [markdown]
# **Verify:** Using Independencies (just the $(B \; \bot \; M \; | \; A)$ independence)
# %% codecell
indepBurglary: IndependenceAssertion = Independencies([Burglary.var, MaryCalls.var, [Alarm.var]]).get_assertions()[0]; indepBurglary

indepMary: IndependenceAssertion = Independencies([MaryCalls.var, Burglary.var, [Alarm.var]]).get_assertions()[0]; indepMary

# Using the fact that closure returns independencies that are IMPLIED by the current independencies:
assert (str(indepMary) == '(MaryCalls _|_ Burglary | Alarm)' and
        indepMary in alarmModel.local_independencies(MaryCalls.var).closure().get_assertions()),  \
        "Check 1: Burglary and MaryCalls are independent once conditional on Alarm"

assert (str(indepBurglary) == '(Burglary _|_ MaryCalls | Alarm)' and
        indepBurglary in alarmModel.local_independencies(MaryCalls.var).closure().get_assertions()), \
        "Check 2: Burglary and MaryCalls are independent once conditional on Alarm"

alarmModel.local_independencies(MaryCalls.var).closure()

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
# ### 3. Inter-Causal (?) Reasoning in the Alarm Model
# For a common cause model $A \leftarrow B \rightarrow C$, there are two cases:
#   * **Marginal Dependence:** ($B$ unknown): When $B$ is unknown / unobserved, there is an active trail between $A$ and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa) when information about $B$'s state is unknown.
#   * **Conditional Independence:** ($B$ fixed): When $B$ is fixed, there is NO active trail between $A$ and $C$, so they are independent. The probability of $A$ won't influence probability of $C$ (and vice versa) when $B$'s state is observed.

# %% codecell
pgmpyToGraph(alarmModel)
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
assert alarmModel.is_active_trail(start = JohnCalls.var, end = MaryCalls.var,  observed = None)

showActiveTrails(model = alarmModel, variables = [JohnCalls.var, MaryCalls.var])


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
assert not alarmModel.is_active_trail(start = JohnCalls.var, end = MaryCalls.var, observed = Alarm.var)

showActiveTrails(model = alarmModel, variables = [JohnCalls.var, MaryCalls.var], observed = Alarm.var)

# %% markdown [markdown]
# **Verify:** Using Independencies
# %% codecell
indepJohn: IndependenceAssertion = Independencies([JohnCalls.var, MaryCalls.var, [Alarm.var]]).get_assertions()[0]; indepJohn

indepMary: IndependenceAssertion = Independencies([MaryCalls.var, JohnCalls.var, [Alarm.var]]).get_assertions()[0]; indepMary


# Using the fact that closure returns independencies that are IMPLIED by the current independencies:
assert (str(indepMary) == '(MaryCalls _|_ JohnCalls | Alarm)' and
        indepMary in alarmModel.local_independencies(MaryCalls.var).closure().get_assertions()),  \
        "Check 1: MaryCalls and JohnCalls are independent once conditional on Alarm"

alarmModel.local_independencies(MaryCalls.var).closure()
# %% codecell
assert (str(indepJohn) == '(JohnCalls _|_ MaryCalls | Alarm)' and
        indepJohn in alarmModel.local_independencies(JohnCalls.var).closure().get_assertions()), \
        "Check 2: JohnCalls and MaryCalls are independent once conditional on Alarm"

alarmModel.local_independencies(MaryCalls.var).closure()

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
# ### 4. Inter-Causal Reasoning in the Alarm Model
# For a common evidence model $A \rightarrow B \leftarrow C$, there are two cases:
#   * **Marginal Independence:** ($B$ unknown): When $B$ is unknown / unobserved, there is NO active trail between $A$ and $C$; they are independent. The probability of $A$ won't influence probability of $C$ (and vice versa) when $B$'s state is unknown.
#   * **Conditional Dependence:** ($B$ fixed): When $B$ is fixed, there IS an active trail between $A$ and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa) when information about $B$ is observed / fixed.


# %% codecell
pgmpyToGraph(alarmModel)
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
assert not alarmModel.is_active_trail(start = Burglary.var, end = Earthquake.var, observed = None)

showActiveTrails(model = alarmModel, variables = [Burglary.var, Earthquake.var])

# %% markdown [markdown]
# **Verify:** Using Independencies
# %% codecell
indepBurgEarth = Independencies([Burglary.var, Earthquake.var])

assert indepBurgEarth == alarmModel.local_independencies(Burglary.var), 'Check 1: Burglary and Earthquake are marginally independent'

assert indepBurgEarth == alarmModel.local_independencies(Earthquake.var), 'Check 2: Burglary and Earthquake are marginally independent'


# See: MaryCalls and Burglary are marginally independent :
print(indepSynonymTable(model = alarmModel, queryNode = Burglary.var))
print(indepSynonymTable(model = alarmModel, queryNode = Earthquake.var))


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
assert alarmModel.is_active_trail(start = Burglary.var, end = Earthquake.var,  observed = Alarm.var)

showActiveTrails(model = alarmModel, variables = [Burglary.var, Earthquake.var], observed = Alarm.var)


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
