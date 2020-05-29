import daft
from typing import *


from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import JointProbabilityDistribution
from pgmpy.factors.discrete.DiscreteFactor import DiscreteFactor
from pgmpy.independencies import Independencies
from pgmpy.independencies.Independencies import IndependenceAssertion
from pgmpy.inference.CausalInference import CausalInference

from operator import mul
from functools import reduce

import itertools

import numpy as np

import collections


import pandas as pd
from pandas.core.frame import DataFrame

# Type alias for clarity
from src.utils.TypeAliases import *







# ----------------------------------------------------

def convertDaftToPgmpy(pgm: daft.PGM) -> BayesianModel:
    """Takes a Daft PGM object and converts it to a pgmpy BayesianModel"""
    edges = [(edge.node1.name, edge.node2.name) for edge in pgm._edges]
    model = BayesianModel(edges)
    return model



# ----------------------------------------------------

def localIndependencySynonyms(model: BayesianModel,
                              query: RandomVariable,
                              useNotation = False) -> List[Name]:
    '''
    Generates all possible equivalent independencies, given a query node and separator nodes.

    For example, for the independency (G _|_ S, L | I, D), all possible equivalent independencies are made by permuting the letters S, L and I, D in their positions. An resulting equivalent independency would then be (G _|_ L, S | I, D) or (G _|_ L, S | D, I)  etc.

    Arguments:
        query: the node from which local independencies are to be calculated.
        condNodes: either List[str] or List[List[str]].
            ---> When it is List[str], it contains a list of nodes that are only after the conditional | sign. For instance, for (D _|_ G,S,L,I), the otherNodes = ['D','S','L','I'].
            ---> when it is List[List[str]], otherNodes contains usually two elements, the list of nodes BEFORE and AFTER the conditional | sign. For instance, for (G _|_ L, S | I, D), otherNodes = [ ['L','S'], ['I','D'] ], where the nodes before the conditional sign are L,S and the nodes after the conditional sign are I, D.

    Returns:
        List of generated string independency combinations.
    '''
    # First check that the query node has local independencies!
    # TODO check how to match up with the otherNodes argument
    if model.local_independencies(query.var) == Independencies():
        return


    locIndeps = model.local_independencies(query.var)
    _, condExpr = str(locIndeps).split('_|_')

    condNodes: List[List[Name]] = []

    if "|" in condExpr:
        beforeCond, afterCond = condExpr.split("|")
        # Removing the paranthesis after the last letter:
        afterCond = afterCond[0 : len(afterCond) - 1]

        beforeCondList: List[Name] = list(map(lambda letter: letter.strip(), beforeCond.split(",")))
        afterCondList: List[Name] = list(map(lambda letter: letter.strip(), afterCond.split(",")))
        condNodes: List[List[Name]] = [beforeCondList] + [afterCondList]

    else: # just have an expr like "leters" that are only before cond
        beforeCond = condExpr[0 : len(condExpr) - 1]
        beforeCondList: List[Name] = list(map(lambda letter: letter.strip(), beforeCond.split(",")))
        condNodes: List[List[Name]] = [beforeCondList]

    otherComboStrList = []

    for letterSet in condNodes:
        # NOTE: could use comma here instead of the '∩' (and) symbol
        if useNotation: # use 'set and' symbol and brackets (set notation, clearer than simple notation)
            comboStrs: List[str] = list(map(
                lambda letterCombo : "{" + ' ∩ '.join(letterCombo) + "}" if len(letterCombo) > 1 else ' ∩ '.join(letterCombo),
                itertools.permutations(letterSet)))
        else: # use commas and no brackets (simple notation)
            comboStrs: List[str] = list(map(lambda letterCombo : ', '.join(letterCombo),
                                            itertools.permutations(letterSet)))

        # Add this particular combination of letters (variables) to the list.
        otherComboStrList.append(comboStrs)


    # Do product of the after-before variable string combinations.
    # (For instance, given the list [['S,L', 'L,S'], ['D,I', 'I,D']], this operation returns the product list: [('S,L', 'D,I'), ('S,L', 'I,D'), ('L,S', 'D,I'), ('L,S', 'I,D')]
    condComboStr: List[Tuple[Name]] = list(itertools.product(*otherComboStrList))

    # Joining the individual strings in the tuples (above) with conditional sign '|'
    condComboStr: List[str] = list(map(lambda condPair : ' | '.join(condPair), condComboStr))

    independencyCombos: List[str] = list(map(lambda letterComboStr : f"({query.var} _|_ {letterComboStr})", condComboStr))

    return independencyCombos




def indepSynonymTable(model: BayesianModel, query: RandomVariable):

    # fancy independencies
    xs: List[str] = localIndependencySynonyms(model = model, query= query.var, useNotation = True)
    # regular notation independencies
    ys: List[str] = localIndependencySynonyms(model = model, query= query.var)

    # Skip if no result (if not independencies)
    if xs is None and ys is None:
        return

    # Create table spacing logic
    numBetweenSpace: int = 5
    numDots: int = 5


    dots: str = ''.ljust(numDots, '.') # making as many dots as numDots
    betweenSpace: str = ''.ljust(numBetweenSpace, ' ')

    fancyNotationTitle: str = 'Fancy Notation'.ljust(len(xs[0]) , ' ')
    regularNotationTitle: str = "Regular Notation".ljust(len(ys[0]), ' ')

    numTotalRowSpace: int = max(len(xs[0]), len(fancyNotationTitle.strip())) + \
                            2 * numBetweenSpace + numDots + \
                            max(len(ys[0]), len(regularNotationTitle.strip()))

    title: str = "INDEPENDENCIES TABLE".center(numTotalRowSpace, ' ')

    separatorLine: str = ''.ljust(numTotalRowSpace, '-')

    zs: List[str] = list(map(lambda tuple : f"{tuple[0]}{betweenSpace + dots + betweenSpace}{tuple[1]}", zip(xs, ys)))

    # TODO had to add extra space --- why? (below before dots to make dots in title line up with dots in rows)
    table: str = title + "\n" + \
                 fancyNotationTitle + betweenSpace +  dots + betweenSpace + regularNotationTitle + "\n" + \
                 separatorLine + "\n" + \
                 "\n".join(zs)

    print(table)



# ------------------------------------------------------------------------------------------------------------

# TODO given two nodes A, B with conditional dependencies, say A | D, E and B | D,F,H then how do we compute their
#  joint probability distribution?

# The efforts of the goal below are these two commented functions:
# TODO IDEA: P(A, B) = SUM (other vars not A, B) of P(A, B, C, D, E, F ...)
# and that is the so-called joint distribution over two nodes or one node or whatever (is in fact called the marginal
# distribution)
# TODO Same as saying variableElimObj.query(A, B)
# And provide evidence if only need state ???

"""


def jointProbNode_manual(model: BayesianModel, queryNode: Variable) -> JointProbabilityDistribution:
    queryCPD: List[List[Probability]] = model.get_cpds(queryNode).get_values().T.tolist()

    evVars: List[Variable] = list(model.get_cpds(queryNode).state_names.keys())[1:]

    if evVars == []:
        return model.get_cpds(queryNode).to_factor()

    # 1 create combos of values between the evidence vars
    evCPDLists: List[List[Probability]] = [(model.get_cpds(ev).get_values().T.tolist()) for ev in evVars]
    # Make flatter so combinations can be made properly (below)
    evCPDFlatter: List[Probability] = list(itertools.chain(*evCPDLists))
    # passing the flattened list
    evValueCombos = list(itertools.product(*evCPDFlatter))

    # 2. do product of the combos of those evidence var values
    evProds = list(map(lambda evCombo : reduce(mul, evCombo), evValueCombos))

    # 3. zip the products above with the list of values of the CPD of the queryNode
    pairProdAndQueryCPD: List[Tuple[float, List[float]]] = list(zip(evProds, queryCPD))
    # 4. do product on that zip
    jpd: List[Probability] = list(itertools.chain(*[ [evProd * prob for prob in probs] for evProd, probs in pairProdAndQueryCPD]))

    return JointProbabilityDistribution(variables = [queryNode] + evVars,
                          cardinality = model.get_cpds(queryNode).cardinality,
                          values = jpd / sum(jpd)


def jointProbNode(model: BayesianModel, queryNode: Variable) -> JointProbabilityDistribution:
    '''Returns joint prob (discrete factor) for queryNode. Not a probability distribution since sum of outputted probabilities may not be 1, so cannot put in JointProbabilityDistribution object'''

    # Get the conditional variables
    evVars: List[Variable] = list(model.get_cpds(queryNode).state_names.keys())[1:]
    evCPDs: List[DiscreteFactor] = [model.get_cpds(evVar).to_factor() for evVar in evVars]
    queryCPD: DiscreteFactor = model.get_cpds(queryNode).to_factor()
    # There is no reason the cpds must be converted to DiscreteFactors ; can access variables, values, cardinality the same way, but this is how the mini-example in API docs does it. (imap() implementation)

    #factors: List[DiscreteFactor] = [cpd.to_factor() for cpd in model.get_cpds(queryNode)]
    # If there are no evidence variables, then the query node is not conditional on anything, so just return its cpd
    jointProbFactor: DiscreteFactor = reduce(mul, [queryCPD] + evCPDs) if evCPDs != [] else queryCPD

    #Normalizing numbers so they sum to 1, so that we can return as distribution.
    jointProbFactor: DiscreteFactor = jointProbFactor.normalize(inplace = False)


    return JointProbabilityDistribution(variables = jointProbFactor.variables,
                                        cardinality = jointProbFactor.cardinality,
                                        values = jointProbFactor.values)


# --------

# Test cases
#print(jointProbNode_manual(alarmModel_brief, 'J'))

#print(jointProbNode(alarmModel_brief, 'J'))


# %% codecell
print(jointProbNode(alarmModel, 'Alarm'))
# -------

# Test cases 2 (grade model from Alarm.py) ----- works well when the tables are independent!

joint_diffAndIntel: TabularCPD = reduce(mul, [gradeModel.get_cpds('diff'), gradeModel.get_cpds('intel')])
print(joint_diffAndIntel)

"""

# ------------------------------------------------------------------------------

def jointDistribution(model: BayesianModel) -> JointProbabilityDistribution:
    ''' Returns joint prob distribution over entire network'''

    # There is no reason the cpds must be converted to DiscreteFactors ; can access variables, values, cardinality the same way, but this is how the mini-example in API docs does it. (imap() implementation)
    factors: List[DiscreteFactor] = [cpd.to_factor() for cpd in model.get_cpds()]
    jointProbFactor: DiscreteFactor = reduce(mul, factors)

    # TODO need to assert that probabilities sum to 1? Always true? or to normalize here?

    return JointProbabilityDistribution(variables = jointProbFactor.variables,
                                        cardinality = jointProbFactor.cardinality,
                                        values = jointProbFactor.values)

# ------------------------------------------------------------------------------

# TODO function that simplifies expr like P(L | S, G, I) into P(L | S) when L _|_ G, I, for instance
# TODO RULE (from Ankur ankan, page 17 probability chain rule for bayesian networks)

# See 2_BayesianNetworks file after the probChainRule example

# -------------------------------------------------------------------------------------

def probChainRule(condAcc: List[Name], acc: Name = '') -> str:
    '''
    Recursively applies the probability chain rule when given a list like [A, B, C] interprets this to be P(A, B,
    C) and decomposes it into 'P(A | B, C) * P(B | C) * P(C)'

    '''
    if len(condAcc) == 1:
        #print(acc + "P(" + condAcc[0] + ")")
        return acc + "P(" + condAcc[0] + ")"
    else:
        firstVar = condAcc[0]
        otherVars = condAcc[1:]
        curAcc = f'P({firstVar} | {", ".join(otherVars)}) * '
        return probChainRule(condAcc = otherVars, acc = acc + curAcc)


# ------------------------------------------------------------------------------------------


def activeTrails(model: BayesianModel,
                 variables: List[RandomVariable],
                 observed: List[RandomVariable] = None,
                 skipSelfTrail: bool = True) -> List[Trail]:

    '''Creates trails by threading the way through the dictionary returned by the pgmpy function `active_trail_nodes`'''
    varNames: List[Name] = list(map(lambda randomVar: randomVar.var, variables))
    obsNames: List[Name] = None if observed is None else list(map(lambda obsVar : obsVar.var, observed))

    trails: Dict[Name, Set[Name]] = model.active_trail_nodes(variables = varNames, observed = obsNames)

    trailTupleList: List[List[Tuple[Name, Name]]] = [[(startVar, endVar) for endVar in endVarList]
                                                     for (startVar, endVarList) in trails.items()]


    trailTuples: List[Tuple[Name, Name]] = list(itertools.chain(*trailTupleList))

    if skipSelfTrail: # then remove the ones with same start and end
        trailTuples = list(filter(lambda tup : tup[0] != tup[1], trailTuples))

    explicitTrails: List[Trail] = list(map(lambda tup : f"{tup[0]} --> {tup[1]}", trailTuples))

    return explicitTrails





def showActiveTrails(model: BayesianModel,
                     variables: List[RandomVariable],
                     observed: List[RandomVariable] = None):

    trails: List[Trail] = activeTrails(model, variables, observed)
    print('\n'.join(trails))




# ----------------------------------------------------------------------------------------------------






ARROW = "ARROW"
PAIR = "PAIR"


# Gets all backdoor adjustment sets between the query var and all other nodes in the graph.
def backdoorAdjustSets(model: BayesianModel,
                       node: RandomVariable,
                       notation: str = ARROW) -> Dict[Name, List[Set[Name]]]:

    inference: CausalInference = CausalInference(model)

    # Getting all the predecessors to get a more complete list of possible adjustment sets
    # TODO: does this give the entire possible list of adjustment sets with each predecessor node, from the bottom
    #  queryVar?
    #predecessorVars = model.predecessors(queryVar) #model.get_parents(queryVar)
    allVars = model.nodes()

    # Getting the variables that will be used as evidence / observed to influence active trails (in other words,
    # the variables that must be set as observed in the query of variable elimination)
    varAndObservedPairs = set([ (startVar, inference.get_all_backdoor_adjustment_sets(X = startVar, Y = node.var))
                                for startVar in allVars])
    # remove null forzen sets
    #pairsOfPredObservedVars = list(filter(lambda pair: pair[1] != frozenset(), pairsOfPredObservedVars))


    # Attaching ev var to the frozen set adjustment sets (changing the frozenset datatype to be set on the inside and
    # list on the outside)
    backdoorChoices: List[Tuple[Name, Set[Name]]] = list(itertools.chain(
        *[[ (startVar, set(innerFroz)) for innerFroz in outerFroz] if outerFroz != frozenset() else [(startVar, None)]
          for startVar, outerFroz in varAndObservedPairs])
    )

    # Creating a dict to accumulate adjustment sets of the same keys (concatenating)
    backdoorDict: Dict[Name, List[Set[Name]]] = {}

    for startVar, adjustSets in backdoorChoices:

        if startVar in backdoorDict.keys():
            backdoorDict[startVar] = backdoorDict[startVar] + [adjustSets]
        else:
            backdoorDict[startVar] = [adjustSets]


    if notation == ARROW: #use arrows
        # Now creating the arrow between startvar and endvar (to make the path clear)
        backdoorTrailDict: Dict[Trail, List[Set[Name]]] = {}

        for startVar, adjustLists in backdoorDict.items():
            backdoorTrailDict[f"{startVar} --> {node.var}"] = adjustLists


        return backdoorTrailDict
    elif notation == PAIR:
        Pair = collections.namedtuple("Pair", ["From", "To", "AdjustSets" ])

        lists = []
        for startVar, adjustLists in backdoorDict.items():
            lists.append(Pair(From = startVar, To = node.var, AdjustSets = adjustLists))

        return lists
    else: # do some notation (if notation == None)
        return backdoorDict


# ----------------------------------------------------------------------------------------------------------------------


# Uses backdoor adjustment sets to find potential observed variables that can nullify active trail nodes (for first
# three models) and create active trail nodes (for common ev model)


def observedVars(model: BayesianModel, start: RandomVariable, end: RandomVariable) -> List[Set[RandomVariable]]:


    startBackdoors: Dict[Name, List[Set[Name]]] = backdoorAdjustSets(model, node = start, notation = None)
    endBackdoors: Dict[Name, List[Set[Name]]] = backdoorAdjustSets(model, node = end, notation =None)

    shortenedResult: List[Set[Name]] = startBackdoors[end.var] + endBackdoors[start.var]

    shortenedResult = list(filter(lambda elem : elem != None, shortenedResult))

    # If the list is empty then we must include the start and end vars so we can potentially nullify the active trail: thinking of test case processtype -> injurytype when the above shortenedresult == [] and where the startvar and endvar included as observed vars will nullify the active trail (no others are found)
    shortenedResult = shortenedResult if shortenedResult != [] else [{start.var}, {end.var}]

    return mergeSubsets(shortenedResult) # merge teh subsets within this list




# Helper functions below for observedVars function:

def mergeSubsets(varSetList: List[Set[Name]]) -> List[List[Name]]:

    # Step 1: create combination tuples
    combos = list(itertools.combinations(varSetList, r = 2))


    # STEP 2a) gathered the same key values under the same key
    gather = dict()
    #counter = 0
    #('A','B') in {('A','B'):[1,2,3]}
    for i in range(0, len(combos)-1):
        curKey, curValue = combos[i]
        nextKey, nextValue = combos[i+1]

        curKey: Tuple[Name] = tuple(curKey) # so that it becomes hashable to allow search in the dict

        # Replacing the value or adding to it at the current key, leaving next key for next time if different.
        valueAdd: List[Set[Name]] = [curValue, nextValue] if curKey == nextKey else [curValue]
        gather[curKey] = valueAdd if curKey not in gather.keys() else gather[curKey] + valueAdd

    # Now do the last value:
    curKey, curValue = combos[len(combos)-1]
    curKey = tuple(curKey) # make hashable
    gather[curKey] = [curValue]

    # STEP 2b) For each key : list pair in the dict, ...flag if have merged with the value: IF NOT MERGED ANY: gather just the key ('A') ELSE IF have merged at least once, then gather just the merged results
    merged = []

    for sourceTuple, sets in gather.items():

        sourceSet: Set[Name] = set(sourceTuple)

        for valueSet in sets:

            if isOverlap(sourceSet, valueSet):

                merged.append(tuple( sourceSet.union(valueSet) ))


        if not haveMergedInPast(sourceSet, merged):
            merged.append(sourceTuple) # adding as tuple to be able to remove duplicate easily later

    # Clean up types in the merged result so it is list of sets
    return list(map(lambda tup: set(tup), set(merged)))



# if the result of the filter is empty then we have not merged that in the past, else if it is not empty then that
# means we merged the sourceset in the past.
def haveMergedInPast(sourceSet: Set[Name], merged: List) -> bool:
    zipSourceMerged: List[Tuple[Set[Name], Set[Name]]] = list(zip([sourceSet] * len(merged), merged))

    pastMerges: List[Tuple[Set[Name], Set[Name]]] = list(filter(lambda tup : isOverlap(tup[0], tup[1]),
                                                                zipSourceMerged))
    return len(pastMerges) != 0


# Checks if either set is the subset of the other
def isOverlap(set1: Set[Name], set2: Set[Name]) -> bool:
    return set1.issubset(set2) or set1.issuperset(set2)


# use case for above function
'''
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

vals = getPotentialObservedVars(mod2, "X", "Y"); vals
assert mergeSubsets(vals) == [{'D', 'E', 'F'}, {'C', 'D', 'F'}, {'A', 'D', 'F'}, {'B', 'D', 'F'}]

'''




# ----------------------------------------------------------------------------------------------------------------------


# Function that calculate the distributins based on variables given to pass as observed; we calculate the variable elimination based on all possible combos of the given variable states



# TODO: check if the order in which values are displayed in the data frame are actually the order in which the
#  elimination occurs (so check that the probabilities on a certain row actually CORRESPOND to the states of the
#  random variables)

def eliminate(model: BayesianModel,
              query: RandomVariable,
              evidence: List[RandomVariable] = None) -> DataFrame:
    '''Does Variable Elimination for all the combinations of states of the given evidence variables, using the query
    as the query variable'''

    elim = VariableElimination(model)

    if evidence is None:
        marginalDist: DiscreteFactor = elim.query(variables = [query.var], evidence = None, show_progress = False)
        queryProbs: List[Probability] = marginalDist.values
        topColNames = ['']
        ordStateNames: List[State] = list(marginalDist.state_names.values())[0]
        df: DataFrame = DataFrame(data = queryProbs, index = ordStateNames, columns = topColNames)
        df.index.name = query.var
        return df.transpose()


    # The variable names of the given random variables
    evidenceNames: List[Name] = list(map(lambda node : node.var, evidence))
    # The variable states of the given random variables
    evidenceStates: List[List[State]] = list(map(lambda node : node.states, evidence))


    # Step 1: connect each varname with each of its possible states:  [(var1, state1), (var1, state2)...]
    varStatePairs: List[List[Tuple[Name, State]]] = [list(itertools.product(*([ev.var], ev.states))) for ev in evidence]

    # Step 2: combine each pairs of a variable with those of other variables: [(var1, state1), (var2, state1)] ...
    observedTuples: List[Tuple[Name, State]] = list(itertools.product(*varStatePairs))
    # Convert each tuple to a dict
    observed: List[Dict[Name, State]] = list(map(lambda triple: dict(triple), observedTuples))

    # Step 3: Key step, eliminating using the evidence combo
    condDists: List[DiscreteFactor] = [elim.query(variables = [query.var],
                                              evidence = evDict,
                                              show_progress = False) for evDict in observed]

    # Step 4: Create the data frame
    evStateCombos = list(itertools.product(*evidenceStates))

    queryProbs: List[List[Probability]] = np.asarray([dist.values for dist in condDists]).T

    #topColNames = [''] if evidence == None else pd.MultiIndex.from_tuples(evStateCombos, names=evidence)
    topColNames = pd.MultiIndex.from_tuples(evStateCombos, names=evidenceNames)

    # Use the "ordered" state names instead of queryVar.states so that we get the actual order of the states as used in the Discrete Factor object
    ordStateNames = list(condDists[0].state_names.values() )[0]
    df: DataFrame = DataFrame(data = queryProbs, index = ordStateNames, columns = topColNames)
    df.index.name = query.var

    return df.transpose()





def eliminateSlice(model: BayesianModel,
                   query: RandomVariable,
                   evidence: Dict[Name, List[State]] = None) -> DataFrame:
    '''
    Applies variable elimination to all possible combinations of the var-states in the passed evidence, for the query variable.
    Arguments:
        model
        query: the variable on which variable elimination is done (to calculate the marginal or conditional CPD)
        evidence: if None, we calculate marginal dist of query, else if evidence is given, the resulting dist we
        calculate of query is called the conditional dist.
        Is a dict of variable string names mapped to list of states we are interested in (may not be ALL states,
        just a snapshot of states from other existing random variables)
    Returns:
        pandas dataframe of the marginal / conditional distribution for query.

    '''
    # Step 1: make the evidence list of name - states into random variables with singular states
    # Each singular state becomes a single-element list in the random variable object
    # NOTE: the nameStatePair[1] MUST be a list, cannot be a single non-iterable value like string or int.
    evidenceVars: List[RandomVariable] = list(map(lambda nameStatePair : RandomVariable(var = nameStatePair[0],
                                                                                        states = nameStatePair[1]),
                                                  evidence.items()))
    # Step 2: pass into the eliminate method
    return eliminate(model, query = query, evidence = evidenceVars)




