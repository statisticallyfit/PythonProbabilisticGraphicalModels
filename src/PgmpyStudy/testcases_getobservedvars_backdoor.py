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

# %% codecell
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import JointProbabilityDistribution
from pgmpy.factors.discrete.DiscreteFactor import DiscreteFactor
from pgmpy.independencies import Independencies
from pgmpy.independencies.Independencies import IndependenceAssertion


import operator
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

mod:BayesianModel = BayesianModel([
    ('A', 'B'), ('A', 'X'), ('C', 'B'), ('C', 'Y'), ('B', 'X'), ('X', 'Y')
])

mod2:BayesianModel = BayesianModel([
    ('X', 'F'),
    ('F', 'Y'),
    ('C', 'X'),
    ('A', 'C'),
    ('A', 'D'),
    ('D', 'X'),
    ('D', 'Y'),
    ('B', 'D'),
    ('B', 'E'),
    ('E', 'Y')
])

infmod2 = CausalInference(mod2)


infmod2.get_all_backdoor_adjustment_sets("B", "Y")
infmod2.get_all_frontdoor_adjustment_sets("B", "Y")
infmod2.get_all_frontdoor_adjustment_sets("Y", "B")
infmod2.get_all_backdoor_adjustment_sets("Y", "B")
mod2.is_active_trail(start = "B", end = "Y", observed = None)
mod2.is_active_trail(start = "B", end = "Y", observed = ['C', 'D', 'E'])
mod2.is_active_trail(start = "B", end = "Y", observed = ['A', 'D', 'E'])
mod2.is_active_trail(start = "B", end = "Y", observed = ['D', 'E', 'F'])
mod2.is_active_trail(start = "B", end = "Y", observed = ['D', 'E', 'X'])


# %% codecell
# ---------
X = RandomVariable(var = "X", states = [])
Y = RandomVariable(var = "Y", states = [])

observedVars(mod2, X, Y)

starts = backdoorsTo(mod2, X, notation = None); starts
ends = backdoorsTo(mod2, Y, notation = None); ends

starts[Y.var] # Y ---> X
ends[X.var] # X ---> Y

res = starts[Y.var] + ends[X.var]; res
# %% codecell
mergeSubsets(res)

# %% codecell
mergeSubsets([{Process.var}, {Tool.var}])


# %% codecell
# Going to fix the mergeSubsets function
#res = res + [{'Z','X','M'}, {'B', 'H'}]


def mergeSubsets(varSetList: List[Set[Name]]) -> List[Set[Name]]:

    # Step 1: Create combinations of tuples of the two dicts:
    combos: List[Tuple[Set[Name], Set[Name]]] = list(itertools.combinations(varSetList, r = 2))

    # Step 2: Merge subsets
    mergedPairs: List[List[Set[Name]]] = list(map(lambda tup : [tup[0].union(tup[1])] if isOverlap(tup[0], tup[1]) else
    list(tup), combos))

    # If the item in the list was merged then its length is necessarily 1 so get the length-1 lists and flatten lists
    merged: List[Set[Name]] = list(itertools.chain(*filter(lambda lst: len(lst) == 1, mergedPairs)))


    # Step 3: Get items that weren't merged:

    # Convert inner types from SET ---> TUPLE because need to do symmetric difference of inner types later and sets aren't hashable while tuples are hashable.
    mergedSetOfTuples: Set[Tuple[Name]] = set(map(lambda theSet: tuple(sorted(theSet)), merged))
    resSetOfTuples: Set[Tuple[Name]] = set(map(lambda theSet: tuple(sorted(theSet)), varSetList))

    # Getting the items not merged
    diffs: Set[Tuple[Name]] = mergedSetOfTuples.symmetric_difference(resSetOfTuples)

    # If for any tuple in the DIFF set we have that the tuple is overlapping any tuple in the merged set, then we know this diff-set tuple has been merged and its union result is already in the merged set, so mark True. Else mark False so that we can keep that one and return it. Or filter.
    diffMergedPairs: List[Tuple[Tuple[Name], Tuple[Name]]] = list(itertools.product(diffs, mergedSetOfTuples))


    def markState(diffMergeTuple: Tuple[Tuple, Tuple]) -> Tuple[Name, bool]:
        diff, merged = diffMergeTuple
        diffSet, mergedSet = set(diff), set(merged)

        if isOverlap(diffSet, mergedSet):
            return (diff, True)
        else:
            return (diff, False)


    # Marking the state with true or false
    boolPairs: List[Tuple[Tuple[Name], bool]] = list(map(lambda diffMergeTup: markState(diffMergeTup), diffMergedPairs))

    # Chunking the bool pairs by first element
    chunks: List[List[Tuple[Tuple[Name], bool]]] = [list(group) for key, group in itertools.groupby(boolPairs, operator.itemgetter(0))]


    # If the bool value is False, for all the items in the current inner list (chunk) then return the first item of the tuple in that list (the first item of the tuple is the same for all tuples in the chunk)

    def anySecondTrue(listOfTuples: List[Tuple[Tuple[Name], bool]]) -> bool:
        ''' If any tuple in the given list has True in its second location, then return True else False'''
        return any(map(lambda tup: tup[1],  listOfTuples))


    unmerged: List[Tuple[Name]] = list(map(lambda lstOfTups:  lstOfTups[0][0]  if not anySecondTrue(lstOfTups) else
    None, chunks))

    # Removing the None value, just keep the tuples
    unmerged: List[Tuple[Name]] = list(filter(lambda thing: thing != None, unmerged))
    # Changing the type to match the merged list type
    unmerged: List[Set[Name]] = list(map(lambda tup: set(sorted(tup)), unmerged))


    # Step 4: final result is the concatenation: set that were merged included in same list as sets that could
    # not be merged (that didn't have overlaps with any other set in the original varSetList)
    return merged + unmerged



mergeSubsets(res)




# %% codecell

mod2.is_active_trail(start = "X", end = "Y", observed = None)

mod2.is_active_trail(start = "X", end = "Y", observed = ['C', 'D', 'F'])
mod2.is_active_trail(start = "X", end = "Y", observed = ['A', 'D', 'F'])
mod2.is_active_trail(start = "X", end = "Y", observed = ['D', 'E', 'F'])
mod2.is_active_trail(start = "X", end = "Y", observed = ['B', 'D', 'F'])

mod2.is_active_trail(start = "X", end = "Y", observed = ['C', 'D'])
mod2.is_active_trail(start = "X", end = "Y", observed = ['A', 'D'])
mod2.is_active_trail(start = "X", end = "Y", observed = ['D', 'E'])
mod2.is_active_trail(start = "X", end = "Y", observed = ['B', 'D'])

# -----
inf = CausalInference(carModel)
inf.get_all_backdoor_adjustment_sets(Tool.var, Absenteeism.var)
inf.get_all_backdoor_adjustment_sets(Absenteeism.var, Tool.var)
inf.get_all_frontdoor_adjustment_sets(Tool.var, Absenteeism.var)
inf.get_all_frontdoor_adjustment_sets(Absenteeism.var, Tool.var)


observedVars(carModel, Tool.var, Absenteeism.var)


# TODO left off here
carModel.is_active_trail(start = Tool.var, end = Absenteeism.var, observed = None)
carModel.is_active_trail(start = Tool.var, end = Absenteeism.var, observed = [Injury.var] + [Process.var])
# With startvar as observed, the active trail is nullified
#carModel.is_active_trail(start = Tool.var, end = Absenteeism.var, observed = [Injury.var, Tool.var])
# With endvar in observed, the active trail is nullified
#carModel.is_active_trail(start = Tool.var, end = Absenteeism.var, observed = [Injury.var, Absenteeism.var])
# Verify how an exact set from backdoor adjustments actually worked here to nullify the active trail:
carModel.is_active_trail(start = Tool.var, end = Absenteeism.var, observed = [Injury.var, Process.var])
# BAD: a direct set from backdoor did not nullify the trail (because we don't include the intermediary required node Injury)
carModel.is_active_trail(start = Tool.var, end = Absenteeism.var, observed = [Process.var])

# -------
# TODO Example of how backdoor adjustment sets don't give the EXTRA node required in order to nullify the active trail. Here the intermediary node is Tooltype (need to put as observed anyway) but the queries for backdoor adjustment sets never yield that extra node Process, which when put in the list with Tool, nullifies the active trail between Process and Injury
# HYPOTHESIS: I think when this is the case (or to do anyway), we need to provide the startvar and endvar in the backdoor adjustment sets just in case ??? is this just circumstance??
inf.get_all_backdoor_adjustment_sets(Process.var, Injury.var)
inf.get_all_backdoor_adjustment_sets(Injury.var, Process.var)
inf.get_all_frontdoor_adjustment_sets(Process.var, Injury.var)
inf.get_all_frontdoor_adjustment_sets(Injury.var, Process.var)


observedVars(carModel, Process.var, Injury.var)


carModel.is_active_trail(start = Process.var, end = Injury.var, observed = None)
carModel.is_active_trail(start = Process.var, end = Injury.var, observed = [Tool.var, Process.var])
# With endvar as observed (Injury) the active trail is properly nullified
carModel.is_active_trail(start = Process.var, end = Injury.var, observed = [Tool.var, Injury.var])
