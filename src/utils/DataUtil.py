

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
def makeData(dataValues: Dict[Variable, List[State]], fileName: str) -> DataFrame:
    '''
    Arguments:
        data: pandas DataFrame
        dataVals: Dict[Variable, List[State]] = {var: data[var].unique() for var in data.columns}
        fileName: str file name of where to save the outputted data. (expect comman delimited csv file.
    '''
    #dataVals: Dict[Variable, List[State]] = {var: data[var].unique() for var in data.columns}

    combinations = list(itertools.product(*list(dataValues.values())))

    # Transferring temporarily to pandas data frame so can write to comma separated csv easily:
    rawCombData: DataFrame = pd.DataFrame(data = combinations, columns = dataValues.keys())


    # Now send to csv and tweak it:
    rawCombData.to_csv(path_or_buf = fileName , sep =',')


    return rawCombData



# ----------------------------------------------------------------------------------------------

# TODO put data frame to tabular cpd over here