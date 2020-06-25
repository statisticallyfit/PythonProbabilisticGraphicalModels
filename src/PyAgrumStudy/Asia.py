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

VisitToAsia = RandomVariable(var = VisitToAsia.var, states = ['True', 'False'])
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

smokingRV: DiscreteVariable = asiaBN.variableFromName(name = Smoking.var)

asiaBN.idFromName(name = Smoking.var)

asiaBN.changeVariableLabel( 5,  "0", "False")


smokingRV_again = asiaBN.variableFromName(name = Smoking.var)
smokingRV.label(0)
smokingRV.label(1)
smokingRV_again.label(0)
smokingRV_again.label(1)

smokingRV.labels()
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
# TODO now it seems to work
asiaLearner: BNLearner = gum.BNLearner(outPath, asiaBN)
asiaLearner
# %% codecell
asiaLearner.names()
# %% codecell
# Returns the column id corresponding to the variable name
asiaLearner.idFromName('visit_to_Asia') # first row is 0
# %% codecell
asiaLearner.nameFromId(4)
# %% markdown
# The BNLearner is capable of recognizing missing values in databases. For this purpose, just indicate as a last argument the list of the strings that represent missing values. Note that, currently, the BNLearner is not yet able to learn in the presence of missing values. This is the reason why, when it discovers that there exist such values, it raises a gum.MissingValueInDatabase exception.
# %% codecell
gum.BNLearner(outPath, asiaBN,  ['?', 'N/A'] )
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
     gnb.getPotential(asiaBN.cpt(asiaBN.idFromName('visit_to_Asia')))
     +'</center></td><td><center>'+
     gnb.getPotential(asiaBN_learnedParams.cpt(asiaBN_learnedParams.idFromName('visit_to_Asia')))
     +'</center></td></tr><tr><td><center>'+
     gnb.getPotential(asiaBN.cpt (asiaBN.idFromName('tuberculosis')))
     +'</center></td><td><center>'+
     gnb.getPotential(asiaBN_learnedParams.cpt(asiaBN_learnedParams.idFromName('tuberculosis')))
     +'</center></td></tr></table>')
# HTML(gnb.getPotential(asiaBN.cpt(asiaBN.idFromName('visit_to_Asia'))))


# %% markdown
# ## Structural Learning a BN from the database
# Three algorithms for structural learning:
# * LocalSearchWithTabuList
# %% codecell
learner = gum.BNLearner(outPath, asiaBN) # using bn as template for variables
asiaBN_learnedBN = learner.learnBN()

# %% markdown
# Noting the differences between original BN and the different ways they were learned:
# %% codecell
asiaBN.variableNodeMap()
asiaBN.cpt(VisitToAsia.var)
asiaBN_learnedBN.cpt(VisitToAsia.var)
asiaBN_learnedParams.cpt(VisitToAsia.var)
