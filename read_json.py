import json

with open('C:/Users/75129/Desktop/nni实验记录/mixed_1d(16)-0.8350.json','r') as f:
    line = f.read()
    d = json.loads(line)
    # data = d['trialMessage']['finalMetricData']['data']
auc=[]
for i in range(100):
    try:
        auc.append(d['trialMessage'][i]['finalMetricData'][0]['data'])
    except KeyError as identifier:
        auc.append(d['trialMessage'][i]['intermediate'][-1]['data'])
print(max(auc))   