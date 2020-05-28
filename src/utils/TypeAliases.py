
from typing import *


# Declaring type aliases for clarity:
Name = str
State = str

import collections
# Create named tuple class with names "Names" and "Objects"
RandomVariable = collections.namedtuple("RandomVariable", ["var", "states"])




Probability = float
Trail = str


Table = str

Grid = List[List]

Color = str

#Value = str
#Desc = str

#Key = int
#Legend = Dict[Key , Value]
OriginAndWeightInfo = Dict # has shape {'origin': 'unknown', 'weight' : FLOAT_NUMBER}
CondProbDist = Dict