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
# ### 1/ Causal Reasoning in the Car Model
# For a causal model $A \rightarrow B \rightarrow C$, there are two cases:
#   * **Marginal Dependence:** ($B$ unknown): When $B$ is unknown / unobserved, there is an active trail between $A$ and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa).
#   * **Conditional Independence:** ($B$ fixed): When $B$ is fixed, there is NO active trail between $A$ and $C$, so they are independent, which means the probability of $A$ won't influence probability of $C$ (and vice versa).


# %% codecell
pgmpyToGraph(carModel)
# %% markdown [markdown]
# #### Testing conditional independence:
# $$
# \color{DodgerBlue}{\text{ExertionLevel (observed)}: \;\;\;\;\;\;\;  \text{Time} \; \bot \; \text{WorkCapacity} \; | \; \text{ExertionLevel}}
# $$

# Given that **ExertionLevel**'s state is unobserved, we can make the following equivalent statements:
# * there is NO active trail between **Time** and **WorkCapacity**.
# * **Time** and **WorkCapacity** are locally independent.
# * the probability of **Time** won't influence probability of **WorkCapacity** (and vice versa).
#

# %% codecell
elim: VariableElimination = VariableElimination(model = carModel)

# %% markdown
# **Testing Conditional Independence:** Using Active Trails
#
# Note here that we need to set another node `Time` as the `observed` variable, alongside the customary middle one `ExertionLevel`, because of the nature of the graph's dependencies (key concept = backdoor adjustment sets). Thus we set `Time` as `observed` also.
# %% codecell
observedVars(carModel, startVar = Time.var, endVar = WorkCapacity.var)
# %% codecell
assert not carModel.is_active_trail(start = Time.var, end = WorkCapacity.var, observed = [ExertionLevel.var] + [Time.var])

# See, there is no active trail from time to work capacity when observing exertion and time.
showActiveTrails(carModel, variables = [Time.var, WorkCapacity.var], observed = [ExertionLevel.var, Time.var])

# %% markdown
# **Testing Conditional Independence:** Using Independencies
# %% codecell
indep: IndependenceAssertion = Independencies([Time.var, WorkCapacity.var, [ExertionLevel.var]]).get_assertions()[0]; indep

carModel.local_independencies(Time.var)
# TODO this is false, why? how to now check conditional independence using independencies method???
# NOTE also that there is never work capacity independent of time, which is what I wanted to show...
indep in carModel.local_independencies(WorkCapacity.var).closure().get_assertions()

carModel.local_independencies(WorkCapacity.var).closure().get_assertions()

# %% markdown
# **Testing Conditional Independence:** Using Probabilities
# The probability below is: (letting $n_i$ be some number such that $1 \leq n_i \leq 30$)
# $$
# \begin{array}{ll}
# P(\text{WorkCapacity} = \text{Low} \; | \; \text{ExertionLevel} = \text{Low})
# &= P(\text{WorkCapacity} = \text{Low} \; | \; \text{ExertionLevel} = \text{Low} \; \cap \; \text{Time} = n_i)  \\
# &= P(\text{WorkCapacity} = \text{Low} \; | \; \text{ExertionLevel} = \text{Low} \; \cap \; \text{Time} = n_j) \\
# &= P(\text{WorkCapacity} = \text{Low} \; | \; \text{ExertionLevel} = \text{Low} \; \cap \; \text{Time} = n_k) \\
# &= 0.02
# \end{array}
# $$
# %% codecell
# (T _|_ W | E)
TEW: DiscreteFactor = elim.query(variables = [WorkCapacity.var], evidence = {ExertionLevel.var : 'Low'})
TEW_1: DiscreteFactor = elim.query(variables = [WorkCapacity.var], evidence = {ExertionLevel.var : 'Low', Time.var : 1})
TEW_2: DiscreteFactor = elim.query(variables = [WorkCapacity.var], evidence = {ExertionLevel.var : 'Low', Time.var : 10})
TEW_3: DiscreteFactor = elim.query(variables = [WorkCapacity.var], evidence = {ExertionLevel.var : 'Low', Time.var : 26})

print(TEW)
print(TEW_1)
print(TEW_2)
print(TEW_3)
# %% codecell
