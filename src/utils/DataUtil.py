

from typing import *
import itertools
import numpy as np


import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete.DiscreteFactor import DiscreteFactor

import collections


# Type aliases
VariableName = str
State = str
Probability = float


RandomVariable = collections.namedtuple("RandomVariable", ["var", "states"])




def cleanData(data: DataFrame) -> DataFrame:
    cleanedData: DataFrame = data.copy().dropna() # copying and dropping rows with NA

    # Removing whitespace from the column NAMES
    cleanedData: DataFrame = cleanedData.rename(columns = lambda x : x.strip(), inplace = False)

    # Removing whitespace from the column VALUES
    #cleanedData: DataFrame = cleanedData.applymap(lambda x: x.strip() if type(x) == str else x)
    # NOTE: the above approach ruins the dataframe printing capability (cannot show data frame as nice as it was
    # before, but instead it looks like messy string with \n values)

    for var in cleanedData.columns:
        valuesNoWhitespace: Series = cleanedData[var].str.strip()
        cleanedData[var] = valuesNoWhitespace

    return cleanedData


#------------------------------------------------------------------------------------------


# Going to pass this so that combinations of each of its values can be created
# Sending the combinations data to csv file so it can be biased and tweaked so we can create training data:
def makeData(dataDict: Dict[VariableName, List[State]], fileName: str = None) -> DataFrame:
    '''
    Arguments:
        data: pandas DataFrame
        dataVals: Dict[Variable, List[State]] = {var: data[var].unique() for var in data.columns}
        fileName: str file name of where to save the outputted data. (expect comman delimited csv file.
    '''
    #dataVals: Dict[Variable, List[State]] = {var: data[var].unique() for var in data.columns}

    combinations = list(itertools.product(*list(dataDict.values())))

    # Transferring temporarily to pandas data frame so can write to comma separated csv easily:
    rawCombData: DataFrame = pd.DataFrame(data = combinations, columns = dataDict.keys())


    # Now send to csv and tweak it:
    if fileName is not None:
        rawCombData.to_csv(path_or_buf = fileName , sep =',')


    return rawCombData



def makeWhiteNoiseData(dataDict: Dict[VariableName, List[State]],
                       signalDict: Dict[VariableName, List[State]],
                       fileName: str = None) -> DataFrame:
    '''
    Creates stub data combinations using the signalDict, which represents var-states that we used to create the test
    case data in another file.

    Arguments:
         dataDict: the data dictionary (values) of all the random variables with their states, in the ENTIRE data set

         signalDict: just a portion of the random variables mapped to a portion of their states, which we used to
         create test cases in another file. (For instance if we were only interested in making test data for
         EngineWiring relationship to ChemicalBurn then the signalDict['ProcessType'] = ['Engine-Wiring'] and
         signalDict['InjuryType'] = ['ChemicalBurn'].

         fileName: str file name where we should save outputted data

    Returns:
        The white noise stub values already included / combinatorialized with the other dataDict values that were not of interest when creating the signal data.
    '''

    # Step 1: create the white noise data
    whiteNoiseDict: Dict[VariableName, List[State]] = {}

    for var, states in dataDict.items():

        if var in signalDict.keys():
            whiteNoiseDict[var] = list(set(states) - set(signalDict[var]))
        else: # need to add the var
            whiteNoiseDict[var] = states # just substitute the entire data states

        # Do the empty operation:
        whiteNoiseDict[var] = states if whiteNoiseDict[var] == list() else whiteNoiseDict[var]

    whiteNoiseData: DataFrame = makeData(dataDict = whiteNoiseDict, fileName = fileName)

    return whiteNoiseData


# ----------------------------------------------------------------------------------------------

# TODO put data frame to tabular cpd over here


# TODO if no evidence then do not transpose
# TODO left off here

def conditionalDistDf(model: BayesianModel, query: RandomVariable) -> DataFrame:
    '''
    Given a query variable, gets its conditional TabularCPD and puts that into a pandas DataFrame
    '''
    # Get the Tabular CPD (learned) from the model:
    queryTCPD: TabularCPD = model.get_cpds(query.var)

    return tabularDf(cpd = queryTCPD)



# TODO if no evidence then do not transpose

def factorDf(factor: DiscreteFactor) -> DataFrame:
    '''
    Converts a pgmpy DiscreteFactor to pandas DataFrame for nicer viewing
    '''
    # Get names of variables (evidence vars) whose combos of states will go on top of the data frame
    evidenceVars: List[VariableName] = list(factor.state_names.keys())[1:]

    # Get all state names mapped to each variable
    states: List[State] = list(factor.state_names.values())

    # Create combinations of states to go on the horizontal part of the CPD dataframe
    evidenceStateCombos: List[Tuple[State, State]] = list(itertools.product(*states[1:]))

    # note: Avoiding error thrown when passing empty list of tuples to MultiIndex
    topColNames = [''] if evidenceVars == [] else pd.MultiIndex.from_tuples(evidenceStateCombos, names=evidenceVars)


    df: DataFrame = DataFrame(data = factor.values, index = states[0], columns = topColNames)
    df.index.name = factor.variables[0] # the query var name

    # NOTE: do not transpose because usually we are passed a discrete factor with no evidence vars,
    # and the single-var view is nicer when  not transposing
    return df #.transpose()





def tabularDf(cpd: TabularCPD) -> DataFrame:
    '''
    Converts a pgmpy TabularCPD to pandas DataFrame for nicer viewing
    '''
    # Get names of variables (evidence vars) whose combos of states will go on top of the data frame
    evidenceVars: List[VariableName] = list(cpd.state_names.keys())[1:]

    # Get all state names mapped to each variable
    states: List[State] = list(cpd.state_names.values())

    # Create combinations of states to go on the horizontal part of the CPD dataframe
    evidenceStateCombos: List[Tuple[State, State]] = list(itertools.product(*states[1:]))

    # note: Avoiding error thrown when passing empty list of tuples to MultiIndex
    topColNames = [''] if evidenceVars == [] else pd.MultiIndex.from_tuples(evidenceStateCombos, names=evidenceVars)


    df: DataFrame = DataFrame(data = cpd.get_values(), index = states[0], columns = topColNames)
    df.index.name = cpd.variable # the query var name

    return df.transpose()