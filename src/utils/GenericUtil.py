from typing import *

# My type alases (for clarity)
from src.utils.TypeAliases import *


import itertools
import numpy as np





# Infix class to create infix operators in python,
# NOTE: Source code obtained from: http://code.activestate.com/recipes/384122/

# definition of an Infix operator class
# this recipe also works in jython
# calling sequence for the infix is either:
#  x |op| y
# or:
# x <<op>> y

class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)
    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __rshift__(self, other):
        return self.function(other)
    def __call__(self, value1, value2):
        return self.function(value1, value2)



# Creating function that helps to add two dicts (when no overlapping keys)


# Examples
'''

# simple multiplication
x=Infix(lambda x,y: x*y)
assert 2 |x| 4 == 8
# => 8

# class checking
isA = Infix(lambda x,y: x.__class__==y.__class__)
assert [1,2,3] |isA| []
assert [1,2,3] <<isA>> []


# inclusion checking
isIn = Infix(lambda x,y: x in y.keys())
assert 1 |isIn| {1:'one'}
assert 1 <<isIn>> {1:'one'}
# => True

# an infix div operator
import operator
# div = Infix(operator.div) # ERROR: module operator has no attribute 'div'
# assert 10 |div| (4 |div| 2) == 5


# functional programming (not working in jython, use the "curry" recipe! )
def curry(f,x):
    def curried_function(*args, **kw):
        return f(*((x,)+args),**kw)
    return curried_function

curry = Infix(curry)

#add5 = operator.add |curry| 5 # TypeError: unsupported operand type(s) for |: 'builtin_function_or_method' and 'function'
# assert add5(6) == 11
'''


# -------------------------------------------------------------------------------------------------------------------


def intersectVarStates(doubleTuple: Tuple[Tuple[Name, Set[State]], Tuple[Name, Set[State]]]):
    (keyA, setA), (keyB, setB) = doubleTuple

    if keyA == keyB:
        return [ (keyA, setA.intersection(setB)) ] # single-element tuple
    else:
        return list(doubleTuple) # two-element tuple



# Merges the tuple (tries) in sorted sorted, Returns list of the single merged tuple or list of the non-merged
# tuples, in sorted order by key
def intersectSortVarStates(doubleTuple: Tuple[Tuple[Name, Set[State]], Tuple[Name, Set[State]]]):
    (keyA, setA), (keyB, setB) = doubleTuple

    if keyA == keyB:
        return [(keyA, setA.intersection(setB))]
    elif  keyA < keyB:
        return [(keyA, setA), (keyB, setB)]
    else: #keyA > keyB:
        return [(keyB, setB), (keyA, setA)]



# TODO check if this actually works, as opposed to function below
'''
def intersectDictValues(xs: Dict[Name, Set[State]],
                        ys: Dict[Name, Set[State]]) -> Dict[Name, Set[State]]:

    # First sort the dictionaries by KEY so that we can satisfy the assumption of categorize tuple function, that there are no repetition of keys in either dicts:
    xsSorted = dict(sorted(xs.items()))
    ysSorted = dict(sorted(ys.items()))

    # merge the tuple values using intersection, else keep the set if either key is above or below the other
    listOfTupleList = map(lambda aTupBTup : mergeSortKeyStates(aTupBTup), zip(xsSorted.items(), ysSorted.items()))

    return dict(itertools.chain(*listOfTupleList)) # using `chain` to flatten the list of lists
'''



# ------------------------------------------------------------------------------------------------------------------------------

# Test cases for function below

#d1 = {'A': {1,2,5,8,3,4}, 'B' : {5,4,3,9} , 'E' : {4,7,2,1,3}, 'D' : {6,9,1}, 'C' : {1,10,11}}
#d2 = {'A': {22,2,3,6,8}, 'C' : {5,4,3,9} , 'Z' : {4,7,2}, 'H' : {5,6,8,1}, 'E' : {1,3}, 'D' : {1}, 'F':{33}, 'G':{5,
# 11,14}}

# Intersecting dict values
# TODO alternate shorter way: use sorted(list(d1.items()) + list(d2.items())) then merge the two following items in  the list (using i from 0 --> len-1 technique)

def addDictsOfSetStates(d1: Dict[Name, Set[State]],
                        d2: Dict[Name, Set[State]]) -> Dict[Name, Set[State]]:
    '''
    Intersects the set values of the two dicts. Returns a dict with all the keys from both dicts included,  mapped to the set intersection of the sets from the corresponding key.
    '''

    # STEP 1: sort the dictionaries by KEY so that we can satisfy the assumption of categorize tuple function, that there are no repetition of keys in either dicts:
    xsSorted = dict(sorted(d1.items()))
    ysSorted = dict(sorted(d2.items()))

    # STEP 1: merge the dicts, where the length is the same

    # Create combinations of tuples of the two dicts:
    combos = list(itertools.combinations(list(xsSorted.items()) + list(ysSorted.items()), r = 2))

    # Merge the tuples if possible
    mergedTuples = list(map(lambda doubleTup : intersectVarStates(doubleTup), combos))

    # Take only the length = 1 lists since those contain the merged tuple, others are not merged.
    mergedStateTuples: List[Tuple[Name, Set[State]]] = dict(itertools.chain(*filter(lambda lst: len(lst) == 1, mergedTuples)))


    # Keys to add from first dict: (from tuples that haveb't been merged)
    firstUnmergedKeys = set(mergedStateTuples.keys()).symmetric_difference(set(xsSorted.keys()))
    secondUnmergedKeys = set(mergedStateTuples.keys()).symmetric_difference(set(ysSorted.keys()))


    # STEP 2: adding keys missing from d1 and d2

    # The var - state tuples that are leftover from d2, that haven't been merged
    firstUnmerged: Dict[Name, Set[State]] = dict(filter(lambda varState : varState[0] in firstUnmergedKeys, d1.items()))
    # The var - state tuples that are leftover from d2, that haven't been merged
    secondUnmerged: Dict[Name, Set[State]] = dict(filter(lambda varState : varState[0] in secondUnmergedKeys, d2.items()))


    mergedStateTuples.update(firstUnmerged)
    mergedStateTuples.update(secondUnmerged)

    mergedDict: Dict[Name, Set[State]] = dict(sorted(mergedStateTuples.items()))

    return mergedDict


# Now after this is declared, we use this like dict1 |plus| dict2
# join
s = Infix(lambda d1, d2 : addDictsOfSetStates(d1, d2))


# ----------------------------------------------------------------------------------------

# Adds two dicts, assuming no keys are the same (so no var names are the same
# Used for quickly concatenating the backdoor states dict to the extra testing dict (like backdoor states of work
#  capacity and time dict added to the training level dict in the Car demo, for part 4/ intercausaul reasoning)
def addDictsOfOneState(d1: Dict[Name, State],
                       d2: Dict[Name, State]) -> Dict[Name, State]:

    if set(d1.keys()).intersection(set(d2.keys())):
        raise Exception("Will not merge dicts when there are duplicate keys!")

    return dict(sorted(list(d1.items()) + list(d2.items())))

# plus
o = Infix(lambda d1, d2 : addDictsOfOneState(d1, d2))



# todo explore how RandomVariable (named tuple) can be converted to DICT in Python 3.8



# ----------------------------------------------------------------------------------------------------
# TODO create foldleft and foldright functions using: https://www.burgaud.com/foldl-foldr-python

# ----------------------------------------------------------------------------------------------------

def allEqual(*arrays: np.ndarray) -> bool:
    '''Checks the given (N) arrays are all equal'''
    pairs = list(itertools.combinations(arrays, r = 2))

    for array1, array2 in pairs:
        if not np.allclose(array1, array2):
            return False
    return True



# ------------------------------------------------------------------------------------------------------




'''
# Testing that the function works correctly:
def main():

    assert intersectDictValues(xs = {'A':set(['A','B']), 'B':set()},
                               ys = {'C':set(['1','2']), 'A':set(['A','C'])}) \
           == {'A': {'A'}, 'B': set(), 'C': {'1', '2'}}
'''