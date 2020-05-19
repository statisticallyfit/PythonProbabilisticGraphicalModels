
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
usecaseData = cleanData(usecaseData)

# Now convert the Time to int:
usecaseData[Time.var] = usecaseData[Time.var].astype(int)

# Quick check that no spaces still remain in the values (like 'High  ' and 'High')
assert len(np.unique(usecaseData[ExertionLevel.var])) == 3

usecaseData
# %% codecell
# Concatenate the two data frames to create the final full data set
# whiteNoiseData['Time'] = whiteNoiseData['Time'].astype(int)

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
# %% markdown
# $\color{green}{\text{SUCCESS:}}$
#
# 1. as **time** increases, it is more likely that **exertion-level** rises also.
# %% codecell
bayesNet.cpds[ExertionLevel.var]
# %% markdown
# $\color{green}{\text{SUCCESS:}}$
#
# 1. as **time** increases, it is more likely that **experience-level** rises also.
# %% codecell
bayesNet.cpds[ExperienceLevel.var]
# %% markdown
# $\color{green}{\text{SUCCESS:}}$
#
# 1. as **time** increases, it is more likely that **training-level** rises also.
# %% codecell
bayesNet.cpds[TrainingLevel.var]


# %% markdown
# $\color{red}{\text{TODO: why isn't work capacity reflecting HIGH --> LOW??}}$
#
# * (a) $\color{red}{\text{X}}$ As **time** increases, the **exertion-level** rises and **experience-level** rises and **training-level** rises which in turn might raise **work-capacity**.
# * (b)  $\color{red}{\text{X}}$  As **time** increases more, the **exertion-level**, **experience-level**, **training-level** may all rise but at a specific point in time, the **exertion-level** may be high enough to lower **work-capacity** more than in Scenario 1, despite the higher levels of **experience-level** and **training-level**.
# %% codecell
bayesNet.cpds[WorkCapacity.var]
# %% markdown
# $\color{green}{\text{SUCCESS: }}$
#
# * (a) Given **process-type** = `Engine-Mount` and **tool-type** = `Forklift` the most likely **injury-type** = `Contact-Contusion` or even `Fall-Gtm` rather than things like `Chemical-Burn`. $\color{red}{\text{Actually, Chemical-Burn turns out high probablility ...? todo}}$
# * (b) Given **process-type** = `Engine-Wiring` and **tool-type** = `Power-Gun` the most likely **injury-type** = `Electrical-Shock` than things like `Contact-Contusion`.
# * (e) Given **process-type** = `Oil-Fill` and **tool-type** = `Oil`, the most likely **injury-type** = either `Chemical-Burn` or `Electrical-Shock`. (NOTE: focused on `Chemical-Burn` in data set so that is why the other option does not have high probability)
# %% codecell
bayesNet.cpds[InjuryType.var]

# %% markdown
# * $\color{blue}{\text{DEBUG}}:$ case (a): the **injury-type** = `Chemical-Burn` came out with probability = $0.496711$ (so basically the highest probability in the CPD) when **tool-type** = `Forklift` and **process-type** = `Engine-Mount` because there was higher frequency of **injury-type** = `Chemical-Burn` in the data, when conditional on these variable states. Just see the snapshot of the data below for the `Forklift` section:

# %% codecell

usecaseData[(usecaseData.ProcessType == 'Engine-Mount') &
            (usecaseData.Time == 1) &
            (usecaseData.ToolType == 'Forklift')]
# %% markdown
# * $\color{blue}{\text{DEBUG}}:$ case (b): there was even probability (both high compared to rest) for values `Electrical-Shock` and `Fall-Gtm` when **process-type** = `Engine-Wiring` and **tool-type** = `Power-Gun`, which reflects what was in the data (but not what was specified in the use case)
# %% codecell

usecaseData[(usecaseData.ProcessType == 'Engine-Wiring') &
            (usecaseData.Time == 1) &
            (usecaseData.ToolType == 'Power-Gun')]

# %% markdown
# * **work-capacity** = `High` when **absenteeism-level** = `Low` with probability $0.750$
# * $\color{red}{\text{TODO}}:$ why is it true that there are equally likely probabilities everywhere else?
# %% codecell
bayesNet.cpds[AbsenteeismLevel.var]



# %% markdown
# ## Step 4: Inference (querying marginals)
# %% codecell
from causalnex.inference import InferenceEngine


eng = InferenceEngine(bn = bayesNet)

# querying the baseline marginals as learned from the data
marginalDist: Dict[Variable, Dict[State, Probability]] = eng.query()
marginalDist

# %% markdown
# Checking marginal distribution of **work-capacity**:
# %% codecell
eng.query()[WorkCapacity.var]
# %% markdown
# Biasing so that lower work capacity probability gets higher:
# %% codecell
# NOTE: in the data, in TIME + 30, when exertion, training, experience are all HIGH, the work-capacity = LOW
eng.query({Time.var : 30, ExertionLevel.var : 'High', TrainingLevel.var : 'High', ExperienceLevel.var : 'High'})[WorkCapacity.var]
# %% codecell
# Different than data: at time = 30, in data all these exertion, experience, training are High, so testing what happens to workcapacity when they are set to Medium:
eng.query({Time.var : 30, ExertionLevel.var : 'Medium', TrainingLevel.var : 'Medium', ExperienceLevel.var : 'Medium'})[WorkCapacity.var]
# %% codecell
# Different than data: at time = 30, in data all these exertion, experience, training are High, so testing what happens to workcapacity when they are set to Low:
eng.query({Time.var : 30, ExertionLevel.var : 'Low', TrainingLevel.var : 'Low', ExperienceLevel.var : 'Low'})[WorkCapacity.var]
# %% codecell
# NOTE: in the data set, exertion, training, experience are all MEDIUM during Time = 5 so this is why the inference also assigns high probability to Medium
eng.query({Time.var : 5, ExertionLevel.var : 'Medium', TrainingLevel.var : 'Medium', ExperienceLevel.var : 'Medium'})[WorkCapacity.var]
# %% codecell
# NOTE: in the data, the work capacity  = Medium when all other vars here are medium, during time = 26 so that is why there is such high probability of Medium for work capacity, assuming these states.
eng.query({Time.var : 26, ExertionLevel.var : 'Medium', TrainingLevel.var : 'Medium', ExperienceLevel.var : 'Medium'})[WorkCapacity.var]
# %% codecell
# NOTE: in the data set, exertion, training, experience are all LOW during Time = 2 so this is why the inference also assigns high probability to LOW

eng.query({Time.var : 2, ExertionLevel.var : 'Low', TrainingLevel.var : 'Low', ExperienceLevel.var : 'Low'})[WorkCapacity.var]

# %% markdown
# ## Step 5: Reasoning via Active Trails
# ### 1/ Reasoning via Active Trails along Causal Chains in the Car Model
# %% codecell
structToGraph(carModel)

# %% markdown
# #### Testing conditional independence:
# $$
# \color{DodgerBlue}{\text{ExertionLevel (observed)}: \;\;\;\;\;\;\;  \text{Time} \; \bot \; \text{WorkCapacity} \; | \; \text{ExertionLevel}}
# $$
# Given that **ExertionLevel**'s state is unobserved, we can make the following equivalent statements:
# * there IS active trail between **Time** and **WorkCapacity**.
# * **Time** and **WorkCapacity** are dependent.
# * the probability of **Time** can influence probability of **WorkCapacity** (and vice versa).
# %% codecell
eng.query({Time.var : 30, ExertionLevel.var : 'High'})[WorkCapacity.var]
# %% codecell
eng.query({Time.var : 5, ExertionLevel.var : 'High'})[WorkCapacity.var]
# %% codecell
eng.query({Time.var : 23, ExertionLevel.var : 'High'})[WorkCapacity.var]
# %% codecell
eng.query({Time.var : 11, ExertionLevel.var : 'High'})[WorkCapacity.var]


# %% markdown
# $\color{green}{\text{SUCCESS}}$ the probabilities ARE different, signifying an active trail between Time and WorkCapacity.

# %% markdown
# #### Testing the causal chain:
# $$
# \color{SeaGreen}{\text{ExertionLevel (unobserved)}: \;\;\;\;\;\;\; \text{Time} \rightarrow \text{ExertionLevel} \rightarrow \text{WorkCapacity}}
# $$
# Given that **ExertionLevel**'s state is unobserved, we can make the following equivalent statements:
# * there is NO active trail between **Time** and **WorkCapacity**.
# * **Time** and **WorkCapacity** are locally independent.
# * the probability of **Time** won't influence probability of **WorkCapacity** (and vice versa).

# %% codecell
eng.query({Time.var : 30})[WorkCapacity.var]
# %% codecell
eng.query({Time.var : 5})[WorkCapacity.var]
# %% codecell
eng.query({Time.var : 23})[WorkCapacity.var]
# %% codecell
eng.query({Time.var : 11})[WorkCapacity.var]
# %% markdown
# $\color{red}{\text{TODO Problem!: }}$ Not supposed to be dependent probabilities (different distributions) when Exertion is not observed, so why are the probabilities different?






# %% markdown
# ### 4/ Reasoning via Active Trails along Common Effect Structures in the Car Model
# %% codecell
structToGraph(carModel)

# %% markdown
# #### Testing marginal independence:
# $$
# \color{DodgerBlue}{\text{AbsenteeismLevel (unobserved)}: \;\;\;\;\;\;\;  \text{InjuryType} \; \bot \; \text{ProcessType} \; | \; \text{AbsenteeismLevel}}
# $$
# Given that **AbsenteeismLevel**'s state is unobserved, we can make the following equivalent statements:
# * there is NO active trail between **InjuryType** and **ProcessType**.
# * **InjuryType** and **ProcessType** are locally independent.
# * the probability of **InjuryType** cannot influence probability of **ProcessType** (and vice versa).
# %% codecell
eng.query({InjuryType.var : 'Contact-Contusion'})[ProcessType.var]
# %% codecell
eng.query({InjuryType.var : 'Fall-Gtm'})[ProcessType.var]
# %% codecell
eng.query({InjuryType.var : 'Electrical-Shock'})[ProcessType.var]
# %% codecell
eng.query({InjuryType.var : 'Chemical-Burn'})[ProcessType.var]

# %% markdown
# $\color{red}{\text{TODO Problem!: }}$ Not supposed to be dependent probabilities (different distributions)



# %% markdown
# #### Testing the inter-causal chain:
# $$
# \color{SeaGreen}{\text{AbsenteeismLevel (observed)}: \;\;\;\;\;\;\; \text{InjuryType} \Longrightarrow \text{AbsenteeismLevel} \Longleftarrow \text{ProcessType}}
# $$
# Given that **AbsenteeismLevel**'s state is observed, we can make the following equivalent statements:
# * there IS active trail between **InjuryType** and **ProcessType**.
# * **InjuryType** and **ProcessType** are dependent.
# * the probability of **InjuryType** can influence probability of **ProcessType** (and vice versa).

# %% codecell
# Arbitrarily setting absentee = Medium for all of them (to compare):
eng.query({InjuryType.var : 'Contact-Contusion', AbsenteeismLevel.var : 'Medium'})[ProcessType.var]
# %% codecell
eng.query({InjuryType.var : 'Fall-Gtm', AbsenteeismLevel.var : 'Medium'})[ProcessType.var]
# %% codecell
eng.query({InjuryType.var : 'Electrical-Shock', AbsenteeismLevel.var : 'Medium'})[ProcessType.var]
# %% codecell
eng.query({InjuryType.var : 'Chemical-Burn', AbsenteeismLevel.var : 'Medium'})[ProcessType.var]
