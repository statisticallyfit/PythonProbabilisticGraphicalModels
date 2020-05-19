

from typing import *
import itertools
import numpy as np


import pandas as pd
from pandas.core.frame import DataFrame


from pgmpy.factors.discrete.CPD import TabularCPD



# Type aliases
Variable = str
State = str
Probability = float


def cleanData(data: DataFrame) -> DataFrame:
    cleanedData: DataFrame = data.copy()

    # Removing whitespace from the column NAMES
    cleanedData = cleanedData.rename(columns = lambda x : x.strip()) # inplace = False

    # Removing whitespace from the column VALUES
    cleanedData = cleanedData.apply(lambda x: str(x).strip() if x.dtype == "object" else x)

    return cleanedData



#------------------------------------------------------------------------------------------


# Going to pass this so that combinations of each of its values can be created
# Sending the combinations data to csv file so it can be biased and tweaked so we can create training data:
def makeData(dataDict: Dict[Variable, List[State]], fileName: str = None) -> DataFrame:
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



def makeWhiteNoiseData(dataDict: Dict[Variable, List[State]],
                       signalDict: Dict[Variable, List[State]],
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
    whiteNoiseDict: Dict[Variable, List[State]] = {}

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