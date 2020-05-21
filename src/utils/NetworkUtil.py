import daft
from typing import *


from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.independencies.Independencies import Independencies
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import JointProbabilityDistribution
from pgmpy.factors.discrete.DiscreteFactor import DiscreteFactor
from pgmpy.inference.CausalInference import CausalInference

from operator import mul
from functools import reduce

import itertools

import collections


# Type alias for clarity

Variable = str
Probability = float
Trail = str

# ----------------------------------------------------

def convertDaftToPgmpy(pgm: daft.PGM) -> BayesianModel:
    """Takes a Daft PGM object and converts it to a pgmpy BayesianModel"""
    edges = [(edge.node1.name, edge.node2.name) for edge in pgm._edges]
    model = BayesianModel(edges)
    return model



# ----------------------------------------------------

def localIndependencySynonyms(model: BayesianModel,
                              queryNode: Variable,
                              useNotation = False) -> List[Variable]:
    '''
    Generates all possible equivalent independencies, given a query node and separator nodes.

    For example, for the independency (G _|_ S, L | I, D), all possible equivalent independencies are made by permuting the letters S, L and I, D in their positions. An resulting equivalent independency would then be (G _|_ L, S | I, D) or (G _|_ L, S | D, I)  etc.

    Arguments:
        queryNode: the node from which local independencies are to be calculated.
        condNodes: either List[str] or List[List[str]].
            ---> When it is List[str], it contains a list of nodes that are only after the conditional | sign. For instance, for (D _|_ G,S,L,I), the otherNodes = ['D','S','L','I'].
            ---> when it is List[List[str]], otherNodes contains usually two elements, the list of nodes BEFORE and AFTER the conditional | sign. For instance, for (G _|_ L, S | I, D), otherNodes = [ ['L','S'], ['I','D'] ], where the nodes before the conditional sign are L,S and the nodes after the conditional sign are I, D.

    Returns:
        List of generated string independency combinations.
    '''
    # First check that the query node has local independencies!
    # TODO check how to match up with the otherNodes argument
    if model.local_independencies(queryNode) == Independencies():
        return


    locIndeps = model.local_independencies(queryNode)
    _, condExpr = str(locIndeps).split('_|_')

    condNodes: List[List[Variable]] = []

    if "|" in condExpr:
        beforeCond, afterCond = condExpr.split("|")
        # Removing the paranthesis after the last letter:
        afterCond = afterCond[0 : len(afterCond) - 1]

        beforeCondList: List[Variable] = list(map(lambda letter: letter.strip(), beforeCond.split(",")))
        afterCondList: List[Variable] = list(map(lambda letter: letter.strip(), afterCond.split(",")))
        condNodes: List[List[Variable]] = [beforeCondList] + [afterCondList]

    else: # just have an expr like "leters" that are only before cond
        beforeCond = condExpr[0 : len(condExpr) - 1]
        beforeCondList: List[Variable] = list(map(lambda letter: letter.strip(), beforeCond.split(",")))
        condNodes: List[List[Variable]] = [beforeCondList]

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
    condComboStr: List[Tuple[Variable]] = list(itertools.product(*otherComboStrList))

    # Joining the individual strings in the tuples (above) with conditional sign '|'
    condComboStr: List[str] = list(map(lambda condPair : ' | '.join(condPair), condComboStr))

    independencyCombos: List[str] = list(map(lambda letterComboStr : f"({queryNode} _|_ {letterComboStr})", condComboStr))

    return independencyCombos




def indepSynonymTable(model: BayesianModel, queryNode: Variable):

    # fancy independencies
    xs: List[str] = localIndependencySynonyms(model = model, queryNode = queryNode, useNotation = True)
    # regular notation independencies
    ys: List[str] = localIndependencySynonyms(model = model, queryNode = queryNode)

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

def probChainRule(condAcc: List[Variable], acc: Variable = '') -> str:
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
                 variables: List[Variable],
                 observed: List[Variable] = None) -> List[Trail]:
    '''Creates trails by threading the way through the dictionary returned by the pgmpy function `active_trail_nodes`'''

    trails: Dict[Variable, Set[Variable]] = model.active_trail_nodes(variables = variables, observed = observed)

    trailTupleList: List[List[Tuple[Variable]]] = [[(startVar, endVar) for endVar in endVarList]
                                                   for (startVar, endVarList) in trails.items()]

    trailTuples: List[Tuple[Variable]] = list(itertools.chain(*trailTupleList))

    explicitTrails: List[Trail] = list(map(lambda tup : f"{tup[0]} --> {tup[1]}", trailTuples))

    return explicitTrails



def showActiveTrails(model: BayesianModel,
                     variables: List[Variable],
                     observed: List[Variable] = None):

    trails: List[Trail] = activeTrails(model, variables, observed)
    print('\n'.join(trails))




# ----------------------------------------------------------------------------------------------------






ARROW = "ARROW"
PAIR = "PAIR"


# Gets all backdoor adjustment sets between the query var and all other nodes in the graph.
def backdoorAdjustSets(model: BayesianModel, endVar: Variable,
                       notation: str = ARROW) -> Dict[Variable, List[Set[Variable]]]:

    inference: CausalInference = CausalInference(model)

    # Getting all the predecessors to get a more complete list of possible adjustment sets
    # TODO: does this give the entire possible list of adjustment sets with each predecessor node, from the bottom
    #  queryVar?
    #predecessorVars = model.predecessors(queryVar) #model.get_parents(queryVar)
    allVars = model.nodes()

    # Getting the variables that will be used as evidence / observed to influence active trails (in other words,
    # the variables that must be set as observed in the query of variable elimination)
    varAndObservedPairs = set([ (startVar, inference.get_all_backdoor_adjustment_sets(X = startVar, Y = endVar))
                                for startVar in allVars])
    # remove null forzen sets
    #pairsOfPredObservedVars = list(filter(lambda pair: pair[1] != frozenset(), pairsOfPredObservedVars))


    # Attaching ev var to the frozen set adjustment sets (changing the frozenset datatype to be set on the inside and
    # list on the outside)
    backdoorChoices: List[Tuple[Variable, Set[Variable]]] = list(itertools.chain(
        *[[ (startVar, set(innerF)) for innerF in outerF] if outerF != frozenset() else [(startVar, None)]
          for startVar, outerF in varAndObservedPairs])
    )

    # Creating a dict to accumulate adjustment sets of the same keys (concatenating)
    backdoorDict: Dict[Variable, List[Set[Variable]]] = {}

    for startVar, adjustSets in backdoorChoices:

        if startVar in backdoorDict.keys():
            backdoorDict[startVar] = backdoorDict[startVar] + [adjustSets]
        else:
            backdoorDict[startVar] = [adjustSets]


    if notation == ARROW: #use arrows
        # Now creating the arrow between startvar and endvar (to make the path clear)
        backdoorTrailDict: Dict[Trail, List[Set[Variable]]] = {}

        for startVar, adjustLists in backdoorDict.items():
            backdoorTrailDict[f"{startVar} --> {endVar}"] = adjustLists


        return backdoorTrailDict
    elif notation == PAIR:
        Pair = collections.namedtuple("Pair", ["From", "To", "AdjustSets" ])

        lists = []
        for startVar, adjustLists in backdoorDict.items():
            lists.append(Pair(From = startVar, To = endVar, AdjustSets = adjustLists))

        return lists
    else: # do some notation (if notation == None)
        return backdoorDict




# Uses backdoor adjustment sets to find potential observed variables that can nullify active trail nodes (for first
# three models) and create active trail nodes (for common ev model)


def observedVars(model: BayesianModel, startVar: Variable, endVar: Variable) -> List[Set[Variable]]:


    startBackdoors: Dict[Variable, List[Set[Variable]]] = backdoorAdjustSets(model, endVar = startVar, notation = None)
    endBackdoors: Dict[Variable, List[Set[Variable]]] = backdoorAdjustSets(model, endVar = endVar, notation = None)

    shortenedResult: List[Set[Variable]] = startBackdoors[endVar] + endBackdoors[startVar]

    shortenedResult = list(filter(lambda elem : elem != None, shortenedResult))

    # If the list is empty then we must include the start and end vars so we can potentially nullify the active trail: thinking of test case processtype -> injurytype when the above shortenedresult == [] and where the startvar and endvar included as observed vars will nullify the active trail (no others are found)
    shortenedResult = shortenedResult if shortenedResult != [] else [{startVar}, {endVar}]

    return mergeSubsets(shortenedResult) # merge teh subsets within this list



def mergeSubsets(varSetList: List[Set[Variable]]) -> List[List[Variable]]:

    # Step 1: create combination tuples
    combos = list(itertools.combinations(varSetList, r = 2)); combos


    # STEP 2a) gathered the same key values under the same key
    gather = dict()
    #counter = 0
    #('A','B') in {('A','B'):[1,2,3]}
    for i in range(0, len(combos)-1):
        curKey, curValue = combos[i]
        nextKey, nextValue = combos[i+1]

        curKey: Tuple[Variable] = tuple(curKey) # so that it becomes hashable to allow search in the dict

        # Replacing the value or adding to it at the current key, leaving next key for next time if different.
        valueAdd: List[Set[Variable]] = [curValue, nextValue] if curKey == nextKey else [curValue]
        gather[curKey] = valueAdd if curKey not in gather.keys() else gather[curKey] + valueAdd

    # Now do the last value:
    curKey, curValue = combos[len(combos)-1]
    curKey = tuple(curKey) # make hashable
    gather[curKey] = [curValue]

    # STEP 2b) For each key : list pair in the dict, ...flag if have merged with the value: IF NOT MERGED ANY: gather just the key ('A') ELSE IF have merged at least once, then gather just the merged results
    merged = []

    for sourceTuple, sets in gather.items():

        sourceSet: Set[Variable] = set(sourceTuple)

        for valueSet in sets:

            if isOverlap(sourceSet, valueSet):

                merged.append(tuple( sourceSet.union(valueSet) ))


        if not haveMergedInPast(sourceSet, merged):
            merged.append(sourceTuple) # adding as tuple to be able to remove duplicate easily later

    # Clean up types in the merged result so it is list of sets
    return list(map(lambda tup: set(tup), set(merged)))



# if the result of the filter is empty then we have not merged that in the past, else if it is not empty then that
# means we merged the sourceset in the past.
def haveMergedInPast(sourceSet: Set[Variable], merged: List) -> bool:
    zipSourceMerged: List[Tuple[Set[Variable], Set[Variable]]] = list(zip([sourceSet] * len(merged), merged))

    pastMerges: List[Tuple[Set[Variable], Set[Variable]]] = list(filter(lambda tup : isOverlap(tup[0], tup[1]),
                                                                        zipSourceMerged))
    return len(pastMerges) != 0


# Checks if either set is the subset of the other
def isOverlap(set1: Set[Variable], set2: Set[Variable]) -> bool:
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