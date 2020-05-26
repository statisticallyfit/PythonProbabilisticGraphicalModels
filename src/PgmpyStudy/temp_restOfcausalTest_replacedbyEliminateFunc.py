

# %% markdown
# ### Causal Reasoning: Exertion - Absenteeism Effect
# %% codecell

# For early time, studying effects of varying experience level on absenteeism
XA_early_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Low', Time.var : EARLY})
XA_early_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Medium', Time.var : EARLY})
XA_early_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'High', Time.var : EARLY})
# %% codecell
# TODO Low Exertion for Early Time gives High Absenteeism (?????)  Not correct
print(XA_early_1)
# %% codecell
print(XA_early_2)
# %% codecell
print(XA_early_3)

# %% codecell
# For first-third time, studying effects of varying experience level on absenteeism
XA_onethird_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Low', Time.var : ONE_THIRD})
XA_onethird_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Medium', Time.var : ONE_THIRD})
XA_onethird_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'High', Time.var : ONE_THIRD})
# %% codecell
print(XA_onethird_1)
# %% codecell
print(XA_onethird_2)
# %% codecell
print(XA_onethird_3)

# %% codecell
# For two-third time, studying effects of varying experience level on absenteeism
XA_twothird_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Low', Time.var : TWO_THIRD})
XA_twothird_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Medium', Time.var : TWO_THIRD})
XA_twothird_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'High', Time.var : TWO_THIRD})
# %% codecell
print(XA_twothird_1)
# %% codecell
print(XA_twothird_2) # higher probability of absentee = High when Exertion = Medium for Time nearly to the end
# %% codecell
print(XA_twothird_3)

# %% codecell
# For late time, studying effects of varying experience level on absenteeism
XA_late_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Low', Time.var : LATE})
XA_late_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'Medium', Time.var : LATE})
XA_late_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {ExertionLevel.var : 'High', Time.var : LATE})
# %% codecell
print(XA_late_1)
# %% codecell
print(XA_late_2) # higher probability of absentee = High when Experience = Medium for Time nearly to the end
# %% codecell
# HIgher probability of absenteeism = High when Exertion = High, for Time = Late (so there is an overriding factor other than Exertion that influences Absenteeism), because I made High Exertion yield High Absenteeism.
print(XA_late_3)







# %% markdown
# ### Causal Reasoning: Training - Absenteeism
# %% codecell

# For early time, studying effects of varying experience level on absenteeism
TA_early_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Low', Time.var : EARLY})
TA_early_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Medium', Time.var : EARLY})
TA_early_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'High', Time.var : EARLY})
# %% codecell
# NOTE: at the beginning, Training and Absenteeism are oppositely correlated!
# Low training for Early time gives High Absenteeism
print(TA_early_1)
# %% codecell
print(TA_early_2)
# %% codecell
print(TA_early_3)

# %% codecell
# For first-third time, studying effects of varying experience level on absenteeism
TA_onethird_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Low', Time.var : ONE_THIRD})
TA_onethird_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Medium', Time.var : ONE_THIRD})
TA_onethird_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'High', Time.var : ONE_THIRD})
# %% codecell
# High Absentee results when Training is Low for Earlyish Time
print(TA_onethird_1)
# %% codecell
print(TA_onethird_2)
# %% codecell
print(TA_onethird_3)

# %% codecell
# For two-third time, studying effects of varying experience level on absenteeism
TA_twothird_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Low', Time.var : TWO_THIRD})
TA_twothird_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Medium', Time.var : TWO_THIRD})
TA_twothird_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'High', Time.var : TWO_THIRD})
# %% codecell
print(TA_twothird_1)
# %% codecell
print(TA_twothird_2) # higher probability of absentee = High when Training = Medium for Time nearly to the end, probably because workers are tired?
# %% codecell
print(TA_twothird_3)

# %% codecell
# For late time, studying effects of varying experience level on absenteeism
TA_late_1 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Low', Time.var : LATE})
TA_late_2 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'Medium', Time.var : LATE})
TA_late_3 = elim.query(variables = [AbsenteeismLevel.var], evidence = {TrainingLevel.var : 'High', Time.var : LATE})
# %% codecell
print(TA_late_1)
# %% codecell
print(TA_late_2) # higher probability of absentee = High when Experience = Medium for Time nearly to the end
# %% codecell
# HIgher probability of absenteeism = High when Training = High, for Time = Late (so there is an overriding factor other than Experience that influences Absenteeism), because I made High Experience yield Low Absenteeism.
# THUS: # NOTE: at the end of time, Training and Absenteeism are positively correlated! (while at the beginning they were negatively correlated)
print(TA_late_3)
