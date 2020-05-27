from typing import *


import itertools
import numpy as np


VariableName = str
State = str




def tupleDoubleIntersect(doubleTuple: Tuple[Tuple[VariableName, Set[VariableName]], Tuple[VariableName, Set[VariableName]]]):
    (keyA, setA), (keyB, setB) = doubleTuple

    if keyA == keyB:
        return [(keyA, setA.intersection(setB))]
    elif  keyA < keyB:
        return [(keyA, setA), (keyB, setB)]
    else: #keyA > keyB:
        return [(keyB, setB), (keyA, setA)]




def intersectDictValues(xs: Dict[VariableName, Set[VariableName]],
                        ys: Dict[VariableName, Set[VariableName]]) -> Dict[VariableName, Set[VariableName]]:

    # First sort the dictionaries by KEY so that we can satisfy the assumption of categorize tuple function, that there are no repetition of keys in either dicts:
    xsSorted = dict(sorted(xs.items()))
    ysSorted = dict(sorted(ys.items()))

    # merge the tuple values using intersection, else keep the set if either key is above or below the other
    listOfTupleList = map(lambda aTupBTup : tupleDoubleIntersect(aTupBTup), zip(xsSorted.items(), ysSorted.items()))

    return dict(itertools.chain(*listOfTupleList)) # using `chain` to flatten the list of lists






def tupleDoubleConcat(doubleTuple: Tuple[Tuple[VariableName, Set[VariableName]], Tuple[VariableName, Set[VariableName]]]):
    (keyA, setA), (keyB, setB) = doubleTuple

    # ASSUME: the case keyA == keyB is ignored in the addDicts function

    if keyA == keyB:
        raise Exception('You ignored assumption: Do not give dicts with duplicate keys within them!')

    elif  keyA < keyB:
        return [(keyA, setA), (keyB, setB)]
    else: #keyA > keyB:
        return [(keyB, setB), (keyA, setA)]


'''
# TODO fix the fact that the longer list is ignored in the ZIP operation!!!

# Simple addition (concat) of dicts, assuming no keys are the same
def addDicts(d1: Dict[Variable, Set[Variable]],
             d2: Dict[Variable, Set[Variable]]) -> Dict[Variable, Set[Variable]]:

    # First sort the dictionaries by KEY so that we can satisfy the assumption of categorize tuple function, that there are no repetition of keys in either dicts:
    xsSorted = dict(sorted(d1.items()))
    ysSorted = dict(sorted(d2.items()))

    # merge the tuple values using intersection, else keep the set if either key is above or below the other
    listOfTupleList = map(lambda aTupBTup : tupleDoubleConcat(aTupBTup), zip(xsSorted.items(), ysSorted.items()))

    return dict(itertools.chain(*listOfTupleList)) # using `chain` to flatten the list of lists
'''


# Adds two dicts, assuming no keys are the same (so no var names are the same
# Used for quickly concatenating the backdoor states dict to the extra testing dict (like backdoor states of work
#  capacity and time dict added to the training level dict in the Car demo, for part 4/ intercausaul reasoning)

# TODO explore how RandomVariable (named tuple) can be converted to DICT in Python 3.8

def addEvidence(d1: Dict[VariableName, State], d2: Dict[VariableName, State]):
    return dict(list(d1.items()) + list(d2.items()))


# TODO multiway dict add, must fix the tuple concat function -- how?

# Simple addition (concat) of dicts, assuming no keys are the same
'''
def addDicts(*dicts: Dict[Variable, Set[Variable]]) -> Dict[Variable, Set[Variable]]:

    # First sort the dictionaries by KEY so that we can satisfy the assumption of categorize tuple function, that there are no repetition of keys in either dicts:
    dictsSorted: List[Dict] = map(lambda theDict: sorted(theDict.items()), dicts)

    # merge the tuple values using intersection, else keep the set if either key is above or below the other
    listOfTupleList = map(lambda aTupBTup : tupleDoubleConcat(aTupBTup), zip(*dictsSorted))

    return dict(itertools.chain(*listOfTupleList)) # using `chain` to flatten the list of lists
'''



# ----------------------------------------------------------------------------------------------------

def allEqual(*arrays: np.ndarray) -> bool:
    '''Checks the given (N) arrays are all equal'''
    pairs = list(itertools.combinations(arrays, r = 2))

    for array1, array2 in pairs:
        if not np.allclose(array1, array2):
            return False
    return True






# Testing that the function works correctly:
def main():

    assert intersectDictValues(xs = {'A':set(['A','B']), 'B':set()},
                               ys = {'C':set(['1','2']), 'A':set(['A','C'])}) \
           == {'A': {'A'}, 'B': set(), 'C': {'1', '2'}}
