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
drawGraph(model)

# %% codecell
# STEP 1: get all causal chains
# STEP 2: get the nodes that go in the observed / evidence in order to  nullify active trails (the  middle node + the backdoors from getobservedvars function)

edges: List[Tuple[VariableName, VariableName]] = list(iter(model.edges()))


roots: List[VariableName] = model.get_roots(); roots
leaves: List[VariableName] = model.get_leaves(); leaves

# Create all possible causal chains from each node using the edges list (always going downward)

# Create a causal trail (longest possible until reaching the leaves

# METHOD 1: get longest possible trail from ROOT to LEAVES and only then do we chunk it into 3-node paths
startEdges = list(filter(lambda tup: tup[0] in roots, edges)); startEdges
interimNodes: List[VariableName] = list(filter(lambda node: not (node in roots) and not (node in leaves), model.nodes())); interimNodes


# Returns dict {varfromvarlist : [children]}
def nodeChildPairs(model: BayesianModel, vars: List[VariableName]) -> Dict[VariableName, List[VariableName]]:
    return [{node : list(model.successors(n = node))} for node in vars]

rootPairs: Dict[VariableName, List[VariableName]] = nodeChildPairs(model, roots); rootPairs
midPairs = [(node, *list(model.successors(n = node)) ) for node in interimNodes]; midPairs


# METHOD 2: for each edge, connect the tail and tip with matching ends
