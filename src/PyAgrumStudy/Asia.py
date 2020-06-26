# %% markdown [markdown]
#
#
# # PyAgrum Tutorial 11: Structural Learning (Asia Bayesian Network)
# **Sources:**
# * [http://www-desir.lip6.fr/~phw/aGrUM/docs/last/notebooks/11-structuralLearning.ipynb.html](http://www-desir.lip6.fr/~phw/aGrUM/docs/last/notebooks/11-structuralLearning.ipynb.html)
# * [(book) Korb - Bayesian Artificial Intelligence](https://synergo.atlassian.net/wiki/spaces/~198416782/pages/1930723690/book+Korb+-+Bayesian+Artificial+Intelligence)

# %% markdown [markdown]
# Doing path-setting:
# %% codecell
import os
import sys
from typing import *
from typing import Union, List, Any

import itertools

os.getcwd()
# Setting the baseline:
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonProbabilisticGraphicalModels')


curPath: str = os.getcwd() + "/src/PyAgrumStudy/"

dataPath: str = curPath + "data/" # os.getcwd() + "/src/_data/"
imagePath: str = curPath + 'images/'

print("curPath = ", curPath, "\n")
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

import pyAgrum as gum
from pyAgrum.pyAgrum import BayesNet
from pyAgrum.pyAgrum import BNLearner
import pyAgrum.lib.notebook as gnb

from IPython.display import Image

from pylab import *
import matplotlib.pyplot as plt

from operator import mul
from functools import reduce

from src.utils.GraphvizUtil import *
from src.utils.NetworkUtil import *



# %% markdown [markdown]
# ## Problem Statement: 2.5.3 Asia
# **Original Medical Diagnosis Example: Lung cancer** A patient has been suffering from shortness of breath (called dyspnoea) and visits the doctor, worried that he has lung cancer. The doctor knows that other diseases, such as tuberculosis and bronchitis, are possible causes, as well as lung cancer. She also knows that other relevant information includes whether or not the patient is a smoker (increasing the chances of cancer and bronchitis) and what sort of air pollution he has been exposed to. A positive X-ray would indicate either TB or lung cancer.
#
# **Asia Problem:** Suppose that we wanted to expand our original medical diagnosis example to represent explicitly some other possible causes of shortness of breath, namely tuberculosis and bronchitis. Suppose also that whether the patient has recently visited Asia is also relevant, since TB is more prevalent there.
# %% codecell
# from IPython.display import Image
Image(filename = imagePath + "asiapic.png")
# %% markdown
# * **NOTE:** Two alternative networks for the Asia example are shown above. In both networks all the nodes are Boolean. The left-hand network is based on the Asia network of Lauritzen and Spiegelhalter (1988). Note the slightly odd intermediate node TBorC, indicating that the patient has either tuberculosis or bronchitis. This node is not strictly necessary; however it reduces the number of arcs elsewhere, by summarizing the similarities between TB and lung cancer in terms of their relationship to positive X-ray results and dyspnoea. Without this node, as can be seen on the right, there are two parents for X-ray and three for Dyspnoea, with the same probabilities repeated in different parts of the CPT. The use of such an intermediate node is an example of “divorcing,” a model structuring method described in section 10.3.6.
#
# **Source:** [(book) Korb - Bayesian Artificial Intelligence](https://synergo.atlassian.net/wiki/spaces/~198416782/pages/1930723690/book+Korb+-+Bayesian+Artificial+Intelligence)
# %% codecell

import collections

# Creating some names for the random variables (nodes) in the graph, to clarify meaning.

# Create named tuple class with names "Names" and "Objects"
RandomVariable = collections.namedtuple("RandomVariable", ["var", "states"])

VisitToAsia = RandomVariable(var = "visit_to_Asia", states = ['True', 'False'])
Smoking = RandomVariable(var = "smoking", states = ['True', 'False'])
Tuberculosis = RandomVariable(var = "tuberculosis", states = ['True', 'False'])
LungCancer = RandomVariable(var = "lung_cancer", states = ['True', 'False'])
TuberOrCancer = RandomVariable(var = "tuberculos_or_cancer", states = ['True', 'False'])
Bronchitis = RandomVariable(var = "bronchitis", states = ['True', 'False'])
PositiveXray = RandomVariable(var = "positive_XraY", states = ['True', 'False'])
Dyspnoea = RandomVariable(var = "dyspnoea", states = ['True', 'False'])


# %% codecell

asiaBN: BayesNet = gum.loadBN(filename = dataPath + "asia.bif")
asiaBN
# %% codecell
asiaBN.names()
# %% markdown
# Viewing the CPTs of the original network:
# %% codecell
asiaBN.cpt(VisitToAsia.var)
# %% codecell
asiaBN.cpt(Smoking.var)
# %% codecell
asiaBN.cpt(Tuberculosis.var)
# %% codecell
asiaBN.cpt(LungCancer.var)
# %% codecell
asiaBN.cpt(TuberOrCancer.var)
# %% codecell
asiaBN.cpt(Bronchitis.var)
# %% codecell
asiaBN.cpt(PositiveXray.var)
# %% codecell
asiaBN.cpt(Dyspnoea.var)

# %% codecell
from pyAgrum import DiscreteVariable

VisitToAsia_RV: DiscreteVariable = asiaBN.variableFromName(name = VisitToAsia.var)
Smoking_RV: DiscreteVariable = asiaBN.variableFromName(name = Smoking.var)
Tuberculosis_RV: DiscreteVariable = asiaBN.variableFromName(name = Tuberculosis.var)
LungCancer_RV: DiscreteVariable = asiaBN.variableFromName(name = LungCancer.var)
TuberOrCancer_RV: DiscreteVariable = asiaBN.variableFromName(name = TuberOrCancer.var)
Bronchitis_RV: DiscreteVariable = asiaBN.variableFromName(name = Bronchitis.var)
PositiveXray_RV: DiscreteVariable = asiaBN.variableFromName(name = PositiveXray.var)
Dyspnoea_RV: DiscreteVariable = asiaBN.variableFromName(name = Dyspnoea.var)


VisitToAsia_ID: int = asiaBN.nodeId(var = VisitToAsia_RV)
assert VisitToAsia_ID == 0
Smoking_ID: int = asiaBN.idFromName(name = Smoking.var)
assert Smoking_ID == 5
Tuberculosis_ID: int = asiaBN.nodeId(var = Tuberculosis_RV)
assert Tuberculosis_ID == 1
LungCancer_ID: int = asiaBN.nodeId(var = LungCancer_RV)
assert LungCancer_ID == 4
TuberOrCancer_ID: int = asiaBN.nodeId(var = TuberOrCancer_RV)
assert TuberOrCancer_ID == 2
Bronchitis_ID: int = asiaBN.nodeId(var = Bronchitis_RV)
assert Bronchitis_ID == 6
PositiveXray_ID: int = asiaBN.nodeId(var = PositiveXray_RV)
assert PositiveXray_ID == 3
Dyspnoea_ID: int = asiaBN.nodeId(var = Dyspnoea_RV)
assert Dyspnoea_ID == 7


assert VisitToAsia_RV.labels() == ('0', '1')
assert Smoking_RV.labels() == ('0', '1')
assert Tuberculosis_RV.labels() == ('0', '1')
assert LungCancer_RV.labels() == ('0', '1')
assert TuberOrCancer_RV.labels() == ('0', '1')
assert Bronchitis_RV.labels() == ('0', '1')
assert PositiveXray_RV.labels() == ('0', '1')
assert Dyspnoea_RV.labels() == ('0', '1')


# Create a separate model with different random variable state labels: (same copy so far)
asiaBN_states: BayesNet = BayesNet(asiaBN)


asiaBN_states.changeVariableLabel(VisitToAsia_ID, "0", "False")
asiaBN_states.changeVariableLabel(VisitToAsia_ID, "1", "True")

asiaBN_states.changeVariableLabel(Smoking_ID, "0", "False")
asiaBN_states.changeVariableLabel(Smoking_ID, "1", "True")

asiaBN_states.changeVariableLabel(Tuberculosis_ID, "0", "False")
asiaBN_states.changeVariableLabel(Tuberculosis_ID, "1", "True")

asiaBN_states.changeVariableLabel(LungCancer_ID, "0", "False")
asiaBN_states.changeVariableLabel(LungCancer_ID, "1", "True")

asiaBN_states.changeVariableLabel(TuberOrCancer_ID, "0", "False")
asiaBN_states.changeVariableLabel(TuberOrCancer_ID, "1", "True")

asiaBN_states.changeVariableLabel(Bronchitis_ID, "0", "False")
asiaBN_states.changeVariableLabel(Bronchitis_ID, "1", "True")

asiaBN_states.changeVariableLabel(PositiveXray_ID, "0", "False")
asiaBN_states.changeVariableLabel(PositiveXray_ID, "1", "True")

asiaBN_states.changeVariableLabel(Dyspnoea_ID, "0", "False")
asiaBN_states.changeVariableLabel(Dyspnoea_ID, "1", "True")


# NOTE: (because of C++ pointers??) the separate DiscreteVariable objects themselves are the ones changing state, via mutation of the BayesNet's state
assert VisitToAsia_RV.labels() == ('False', 'True')
assert Smoking_RV.labels() == ('False', 'True')
assert Tuberculosis_RV.labels() == ('False', 'True')
assert LungCancer_RV.labels() == ('False', 'True')
assert TuberOrCancer_RV.labels() == ('False', 'True')
assert Bronchitis_RV.labels() == ('False', 'True')
assert PositiveXray_RV.labels() == ('False', 'True')
assert Dyspnoea_RV.labels() == ('False', 'True')


# %% codecell
outPath: str = curPath + "out/sampledAsiaResults.csv"


gum.generateCSV(bn = asiaBN, name_out = outPath, n = 500000, visible = True)

# %% codecell
import pyAgrum.lib._utils.oslike as oslike


print("===\n  Size of the generated database\n===")
oslike.wc_l(outPath)
print("\n===\n  First lines\n===")
oslike.head(outPath)

# %% codecell

# Using the bayes net as template for variables
# TODO need to include just filename, not also the bayesnet else I get error, why is notebook different? https://hyp.is/gK6YXrb6EeqBWm9i4zX_qw/www-desir.lip6.fr/~phw/aGrUM/docs/last/notebooks/11-structuralLearning.ipynb.html

asiaLearner: BNLearner = gum.BNLearner(outPath , asiaBN)
# gum.BNLearner(outPath, asiaBN_states) # doesn't work on the labeled version??? why

asiaLearner
# %% codecell
asiaLearner.names()
# %% codecell
# Returns the column id corresponding to the variable name
assert asiaLearner.idFromName(VisitToAsia.var) == VisitToAsia_ID and VisitToAsia_ID == 0

assert asiaLearner.nameFromId(VisitToAsia_ID) == VisitToAsia.var
# %% markdown
# The BNLearner is capable of recognizing missing values in databases. For this purpose, just indicate as a last argument the list of the strings that represent missing values. Note that, currently, the BNLearner is not yet able to learn in the presence of missing values. This is the reason why, when it discovers that there exist such values, it raises a gum.MissingValueInDatabase exception.
# %% codecell
gum.BNLearner(outPath, asiaBN,  ['?', 'N/A'] )
# gum.BNLearner(outPath,   ['?', 'N/A'] )
# %% codecell
oslike.head(filename = dataPath + "asia_missing.csv")

# %% codecell
try:
    learner: BNLearner = gum.BNLearner(dataPath + "asia_missing.csv", asiaBN, ['?', 'N/A'])
except gum.MissingValueInDatabase:
    print("exception raised: there are missing values in the database")




# %% markdown
# ## Parameter Learning from the database
# We give the `asiaBN` bayesian network as a parameter for the learner in order to have the variables and order of labels for each variable.
# %% codecell
# using the BN as template for variables and labels
learner = gum.BNLearner(outPath, asiaBN)
learner.setInitialDAG(g = asiaBN.dag())

# Learn the parameters when structure is known:
asiaBN_learnedParams: BayesNet = learner.learnParameters()

gnb.showBN(asiaBN_learnedParams)
# gnb.showBN(asiaBN) # same thing

# %% codecell
# This is the bad example: learning without the initial template gets the nodes and structure wrong
learnerNoTemplate = gum.BNLearner(outPath)
learnerNoTemplate.setInitialDAG(g = asiaBN.dag())
asiaBNNoTemplate: BayesNet = learnerNoTemplate.learnParameters()

gnb.showBN(asiaBNNoTemplate)
# %% codecell
# This is what the DAG looks like
asiaBN.dag()
# %% codecell
asiaBNNoTemplate.dag() # same
# %% codecell
from IPython.display import HTML

HTML('<table><tr><td style="text-align:center;"><h3>original BN</h3></td>'+
     '<td style="text-align:center;"><h3>Learned BN</h3></td></tr>'+
     '<tr><td><center>'+
     gnb.getPotential(asiaBN.cpt(VisitToAsia.var))
     +'</center></td><td><center>'+
     gnb.getPotential(asiaBN_learnedParams.cpt(VisitToAsia.var))
     +'</center></td></tr><tr><td><center>'+
     gnb.getPotential(asiaBN.cpt (Tuberculosis.var))
     +'</center></td><td><center>'+
     gnb.getPotential(asiaBN_learnedParams.cpt(Tuberculosis.var))
     +'</center></td></tr></table>')
# HTML(gnb.getPotential(asiaBN.cpt(asiaBN.idFromName('visit_to_Asia'))))


# %% markdown
# ## Structural Learning a BN from the database
# Three algorithms for structural learning:
# * LocalSearchWithTabuList
# * GreedyHillClimbing
# * K2
#
# $\color{red}{\text{TODO:  understand these algorithms}}$
#
# **Using:** LocalSearchWithTabuList
# %% codecell
learner = gum.BNLearner(outPath, asiaBN) # using bn as template for variables

# Learn the structure of the BN
learner.useLocalSearchWithTabuList()

asiaBN_learnedStructure_localSearchAlgo = learner.learnBN()

print("Learned in {}ms".format(1000 * learner.currentTime()))

htmlInfo: str = gnb.getInformation(asiaBN_learnedStructure_localSearchAlgo)
gnb.sideBySide(asiaBN_learnedStructure_localSearchAlgo, htmlInfo)

# %% markdown
# Notice how the original .bif BN and parameter-learned BN and structure-learned BN are different:
# %% codecell
asiaBN
# %% codecell
asiaBN_learnedParams


# %% markdown
# [`ExactBNdistance`](https://hyp.is/1OhsSKy4EeqyemuIJO85ew/pyagrum.readthedocs.io/en/0.18.0/BNToolsCompar.html) is a class representing exacte computation of divergence and distance between BNs
# %% codecell
from pyAgrum import ExactBNdistance

exact: ExactBNdistance = gum.ExactBNdistance(asiaBN, asiaBN_learnedStructure_localSearchAlgo)
exact.compute()
# %% codecell
exact: ExactBNdistance = gum.ExactBNdistance(asiaBN_learnedParams, asiaBN_learnedStructure_localSearchAlgo)
exact.compute()
# %% codecell
exact: ExactBNdistance = gum.ExactBNdistance(asiaBN, asiaBN_learnedParams)
exact.compute()



# %% markdown
# **Using:** A Greedy Hill Climbing algorithm: (with insert, remove and change arc as atomic operations)
# %% codecell
learner = gum.BNLearner(outPath, asiaBN) # using bn as template for variables

# Learn the structure of the BN
learner.useGreedyHillClimbing()

asiaBN_learnedStructure_greedyAlgo = learner.learnBN()

print("Learned in {}ms".format(1000 * learner.currentTime()))

htmlInfo: str = gnb.getInformation(asiaBN_learnedStructure_greedyAlgo)
gnb.sideBySide(asiaBN_learnedStructure_greedyAlgo, htmlInfo)


# %% markdown
# Finding distance between these BNs:
# %% codecell
exact: ExactBNdistance = gum.ExactBNdistance(asiaBN, asiaBN_learnedStructure_greedyAlgo)
exact.compute()
# %% codecell
exact: ExactBNdistance = gum.ExactBNdistance(asiaBN_learnedStructure_localSearchAlgo, asiaBN_learnedStructure_greedyAlgo)
exact.compute()





# %% markdown
# **Using:** A K2 algorithm
# %% codecell
learner = gum.BNLearner(outPath, asiaBN) # using bn as template for variables

# Learn the structure of the BN

learner.useK2(list(asiaBN.nodes())) # needs the ids for some reason (??)

asiaBN_learnedStructure_k2Algo = learner.learnBN()

print("Learned in {}ms".format(1000 * learner.currentTime()))

htmlInfo: str = gnb.getInformation(asiaBN_learnedStructure_k2Algo)
gnb.sideBySide(asiaBN_learnedStructure_k2Algo, htmlInfo)


# %% markdown
# Finding distance between these BNs:
# %% codecell
exact: ExactBNdistance = gum.ExactBNdistance(asiaBN, asiaBN_learnedStructure_k2Algo)
exact.compute()
# %% codecell
exact: ExactBNdistance = gum.ExactBNdistance(asiaBN_learnedStructure_k2Algo, asiaBN_learnedStructure_greedyAlgo)
exact.compute()
# %% codecell
exact: ExactBNdistance = gum.ExactBNdistance(asiaBN_learnedStructure_k2Algo,
                                             asiaBN_learnedStructure_localSearchAlgo)
exact.compute()
