
a_set = {"EVALUATOR_ID": "3030", "EVALUATOR_NM": "이민정", "EVAL_PLAN_STEP_CD": "EVAL001", "EVAL_PLAN_STEP_NM": "하향평가", "EVAL_PLAN_CD": "EVPL20200622000002"}
b_set = {"EVALUATOR_ID": "3030", "EVALUATOR_NM": "이민정", "EVAL_PLAN_STEP_CD": "EVAL001", "EVAL_PLAN_STEP_NM": "하향평가", "EVAL_PLAN_CD": "EVPL20200622000003"}
c_set = {"EVALUATOR_ID": "3031", "EVALUATOR_NM": "이효정", "EVAL_PLAN_STEP_CD": "EVAL001", "EVAL_PLAN_STEP_NM": "하향평가", "EVAL_PLAN_CD": "EVPL20200622000003"}
datalist = [a_set, b_set, c_set]

EvalPlanCd = []
for ithSet in datalist:
    EvalPlanCd.append(ithSet["EVAL_PLAN_CD"])
EvalPlanCd = list(set(EvalPlanCd))



#  DOUBLE LIST 만들기
DoubleLi = []
for cd in EvalPlanCd:
    DoubleLi.append([cd])

for data in datalist:
    for i, evalcd in  enumerate(DoubleLi):
        if data["EVAL_PLAN_CD"] == evalcd[0]:
            DoubleLi[i].append(data)

DoubleLi[0][1:]
DoubleLi[1][1:]












set1 = {}
for cd in EvalPlanCd:
    print(cd)
    set1.update({cd : None})

set1['EVPL20200622000003'] = 1

PLANDICT = EvalPlanCd
PLANDICT[0]
for CD in EvalPlanCd:
    print(CD)
    for data in datalist:
        if data["EVAL_PLAN_CD"] == CD:
            PLANDICT[CD] = data