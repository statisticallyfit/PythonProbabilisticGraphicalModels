# %% markdown [markdown]
# # Monty Hall Problem
# Tutorial source

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

# %% markdown [markdown]
# Science imports:
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

import itertools
import collections



from src.utils.TypeAliases import *

from src.utils.GraphvizUtil import *
from src.utils.NetworkUtil import *
from src.utils.DataUtil import *
from src.utils.GenericUtil import *

from typing import *

# My type alias for clarity

import pandas as pd
from pandas.core.frame import DataFrame




# %% markdown [markdown]
# ## Monty Hall Problem Description:
# The Monty Hall Problem is a very famous problem in Probability Theory. The question goes like:
#
# Suppose you're on a game show, and you're given the choice of three doors: Behind one door is a car (the prize); behind the others, goats (not prizes). You pick a door, say `door A`, and the `Host`, who knows what's behind the doors, opens **another** door (so he can never open the same door you the `Contestant` chooses), say he opens `door C`, which has a goat. He then says to you, "Do you want to pick door `door B`?"
#
# The question we want to figure out is: Is it to your advantage to switch your choice?
#
# By intution it seems that there shouldn't be any increase in probability of getting the prize if we switch the door. But using Bayes' Theorem we can show that by switching the door you the `Contestant` have higher chance of winning.
#
# Monty hall wikipedia page: [https://en.wikipedia.org/wiki/Monty_Hall_problem](https://en.wikipedia.org/wiki/Monty_Hall_problem)
#
#
# ## Probabilistic Interpretation:
#
# There are $3$ random variables: the `Contestant`, the `Host`, and the `Prize`.
# * `Contestant` random variable: The contestant (you) can choose any of the doors, so `Contestant` random variable can take on the values of the doors, hence `Contestant` $\in \{ A, B, C\}$. Also it has been randomly placed behind the doors, so there is equal chance the `Contestant` chooses any of the doors. Thus: $P(\text{Contestant} = A) = P(\text{Contestant} = B) = P(\text{Contestant} = C) = \frac{1}{3}$.
# * `Host` random variable: the Host can choose any of the doors, depending on which house the prize or which have been chosen, so `Host` $\in \{ A, B, C\}$.
# * `Prize` random variable: The prize object has been placed behind the doors, so the `Prize` random variable can take on the values of the doors, hence `Prize` $\in \{ A, B, C\}$. Also it has been randomly placed behind the doors, so there is equal chance it could be behind any of the doors. Thus: $P(\text{Prize} = A) = P(\text{Prize} = B) = P(\text{Prize} = C) = \frac{1}{3}$.

# %% codecell
# Defining the network structure

# These values mean that that door was selected
doorA = 'A' # door A was selected
doorB = 'B' # door B was selected
doorC = 'C'

# Using them like Contestant = doorA means Contestant chose door A

Contestant = RandomVariable(var = "Contestant", states = {doorA, doorB, doorC})
Host = RandomVariable(var = "Host", states = {doorA, doorB, doorC})
Prize = RandomVariable(var = "Prize", states = {doorA, doorB, doorC})

montyModel: BayesianModel = BayesianModel([(Prize.var, Host.var), (Contestant.var, Host.var)])

# Defining the CPDs
cpd_C: TabularCPD = TabularCPD(variable = Contestant.var, variable_card = len(Contestant.states),
                               state_names = {Contestant.var : list(Contestant.states)},
                               values = [[1/3, 1/3, 1/3]])
tabularDf(cpd_C)
# %% codecell
cpd_P: TabularCPD = TabularCPD(variable = Prize.var, variable_card = len(Prize.states),
                               state_names = {Prize.var : list(Prize.states)},
                                  values = [[1/3, 1/3, 1/3]])
tabularDf(cpd_P)

# %% markdown [markdown]
# * Given that `Contestant` chose door `B` and ...
#   * Given the `Prize` is behind door `B`, then:
#       * the probability that the `Host` opens door `B` is $0$ (since the `Contestant` chose it and also the `Prize` is behind it)
#       * the probability that the `Host` opens door `A` is $0.5$ (because the other door `C` is still available as an option)
#       * the probability that the `Host` opens door `C` is $0.5$ (because other door `A` is still available as an option)
#
# * Given that `Contestant` chose door `B` and ...
#   * Given the `Prize` is behind door `A`, then:
#       * the probability that the `Host` opens door `B` is $0$ (since the Host can't open the door of the Contestant).
#       * the probability that the `Host` opens door `A` is $0$ (since opening it would reveal the prize).
#       * the probability that the `Host` opens door `C` is $1$ (since there are no other doors left to open).
#
# * Given that `Contestant` chose door `B` and ...
#   * Given the `Prize` is behind door `C`, then:
#       * the probability that the `Host` opens door `B` is $0$ (since the `Contestant` chose already door `B`)
#       * the probability that the `Host` opens door `A` is $1$ (since there is no other door left to open).
#       * the probability that the `Host` opens door `C` is $0$ (since opening it would reveal the prize).
#
# Similar reasoning applies for the other cases when `Contestant` chooses doors `A` or `C`.
# %% codecell
cpd_H: TabularCPD = TabularCPD(variable = Host.var, variable_card = len(Host.states),
                                 values = [[0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
                                           [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
                                           [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0]],
                                 evidence = [Contestant.var, Prize.var],
                                 evidence_card = [len(Contestant.states), len(Prize.states)],
                                 state_names = {Host.var : list(Host.states),
                                                Contestant.var : list(Contestant.states),
                                                Prize.var : list(Prize.states)})
tabularDf(cpd_H)

# %% codecell
# Associating the cpds to the model
montyModel.add_cpds(cpd_C, cpd_P, cpd_H)

# Check model structure and defined CPDs are correct
montyModel.check_model()

# %% markdown [markdown]
# ## Inferring Posterior Probability
# Given that the `Contestant` chooses `door A` and the `Host` chooses `door B`, the probability that the prize  is behind the other `door C` is higher than the probability that it is behind the `Contestant`'s chosen `door A`.
#
# **CONCLUSION**: the `Contestant` should switch his choice to get a higher chance of winning the prize.
# %% codecell
infer = VariableElimination(montyModel)

posteriorPrize: DiscreteFactor = infer.query(variables = [Prize.var],
                             evidence = {Contestant.var : doorA, Host.var : doorB})

#print(posteriorPrize)

factorDf(posteriorPrize)
