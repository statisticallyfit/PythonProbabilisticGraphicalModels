# %% codecell

from src.utils.NetworkUtil import *



# get observed tars that act to disable active trail nodes in causal, evidential, models (except common ev)
def all_getObservedVars(model: BayesianModel,
                    startVar: Variable,
                    endVar: Variable) -> List[Set[Variable]]:
    startBackdoors: Dict[Variable, List[Set[Variable]]] = backdoorAdjustSets(model, startVar, notation = None)
    endBackdoors: Dict[Variable, List[Set[Variable]]] = backdoorAdjustSets(model, endVar, notation = None)


    # Removing the None (no backdoor) variables:
    startTuples: List[Tuple[Set[Variable], Set[Variable]]] = \
        [( set([fromVar]), *adjustList ) if adjustList != [None] else ()
         for fromVar, adjustList in startBackdoors.items()]


    endTuples: List[Tuple[Set[Variable], Set[Variable]]] = \
        [( set([toVar]), *adjustList ) if adjustList != [None] else ()
         for toVar, adjustList in endBackdoors.items()]

    # Squashing the tuples:
    # And concatenating the results (the forward / backward backdoor searches):
    startEndBackdoorSets: List[Set[Variable]] = list(itertools.chain(* (startTuples + endTuples)))

    return startEndBackdoorSets


# %% codecell

mod:BayesianModel = BayesianModel([
    ('A', 'B'), ('A', 'X'), ('C', 'B'), ('C', 'Y'), ('B', 'X'), ('X', 'Y')
])

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

infmod2 = CausalInference(mod2)


infmod2.get_all_backdoor_adjustment_sets("B", "Y")
infmod2.get_all_frontdoor_adjustment_sets("B", "Y")
infmod2.get_all_frontdoor_adjustment_sets("Y", "B")
infmod2.get_all_backdoor_adjustment_sets("Y", "B")
mod2.is_active_trail(start = "B", end = "Y", observed = None)
mod2.is_active_trail(start = "B", end = "Y", observed = ['C', 'D', 'E'])
mod2.is_active_trail(start = "B", end = "Y", observed = ['A', 'D', 'E'])
mod2.is_active_trail(start = "B", end = "Y", observed = ['D', 'E', 'F'])
mod2.is_active_trail(start = "B", end = "Y", observed = ['D', 'E', 'X'])


# %% codecell
# ---------
observedVars(mod2, "X", "Y")

mod2.is_active_trail(start = "X", end = "Y", observed = None)

mod2.is_active_trail(start = "X", end = "Y", observed = ['C', 'D', 'F'])
mod2.is_active_trail(start = "X", end = "Y", observed = ['A', 'D', 'F'])
mod2.is_active_trail(start = "X", end = "Y", observed = ['D', 'E', 'F'])
mod2.is_active_trail(start = "X", end = "Y", observed = ['B', 'D', 'F'])

mod2.is_active_trail(start = "X", end = "Y", observed = ['C', 'D'])
mod2.is_active_trail(start = "X", end = "Y", observed = ['A', 'D'])
mod2.is_active_trail(start = "X", end = "Y", observed = ['D', 'E'])
mod2.is_active_trail(start = "X", end = "Y", observed = ['B', 'D'])

# -----
inf = CausalInference(carModel)
inf.get_all_backdoor_adjustment_sets(ToolType.var, AbsenteeismLevel.var)
inf.get_all_backdoor_adjustment_sets(AbsenteeismLevel.var, ToolType.var)
inf.get_all_frontdoor_adjustment_sets(ToolType.var, AbsenteeismLevel.var)
inf.get_all_frontdoor_adjustment_sets(AbsenteeismLevel.var, ToolType.var)


observedVars(carModel, ToolType.var, AbsenteeismLevel.var)


# TODO left off here
carModel.is_active_trail(start = ToolType.var, end = AbsenteeismLevel.var, observed = None)
carModel.is_active_trail(start = ToolType.var, end = AbsenteeismLevel.var, observed = [InjuryType.var] + [ProcessType.var])
# With startvar as observed, the active trail is nullified
#carModel.is_active_trail(start = ToolType.var, end = AbsenteeismLevel.var, observed = [InjuryType.var, ToolType.var])
# With endvar in observed, the active trail is nullified
#carModel.is_active_trail(start = ToolType.var, end = AbsenteeismLevel.var, observed = [InjuryType.var, AbsenteeismLevel.var])
# Verify how an exact set from backdoor adjustments actually worked here to nullify the active trail:
carModel.is_active_trail(start = ToolType.var, end = AbsenteeismLevel.var, observed = [InjuryType.var, ProcessType.var])
# BAD: a direct set from backdoor did not nullify the trail (because we don't include the intermediary required node InjuryType)
carModel.is_active_trail(start = ToolType.var, end = AbsenteeismLevel.var, observed = [ProcessType.var])

# -------
# TODO Example of how backdoor adjustment sets don't give the EXTRA node required in order to nullify the active trail. Here the intermediary node is Tooltype (need to put as observed anyway) but the queries for backdoor adjustment sets never yield that extra node ProcessType, which when put in the list with ToolType, nullifies the active trail between ProcessType and InjuryType
# HYPOTHESIS: I think when this is the case (or to do anyway), we need to provide the startvar and endvar in the backdoor adjustment sets just in case ??? is this just circumstance??
inf.get_all_backdoor_adjustment_sets(ProcessType.var, InjuryType.var)
inf.get_all_backdoor_adjustment_sets(InjuryType.var, ProcessType.var)
inf.get_all_frontdoor_adjustment_sets(ProcessType.var, InjuryType.var)
inf.get_all_frontdoor_adjustment_sets(InjuryType.var, ProcessType.var)


observedVars(carModel, ProcessType.var, InjuryType.var)


carModel.is_active_trail(start = ProcessType.var, end = InjuryType.var, observed = None)
carModel.is_active_trail(start = ProcessType.var, end = InjuryType.var, observed = [ToolType.var, ProcessType.var])
# With endvar as observed (InjuryType) the active trail is properly nullified
carModel.is_active_trail(start = ProcessType.var, end = InjuryType.var, observed = [ToolType.var, InjuryType.var])
