import csv
import itertools
from pathlib import Path

import numpy as np
from dpipe.io import load
metric = 'dice'
base_path = Path('/home/dsi/shaya/unsup_resres_zoom/')
combs = list(itertools.permutations(range(6), 2))
# all_scores = {'oracle':{'best':{}},'pretrain':{'best':{}},'their':{'best':{},'last':{},'worst_source':{}},'adaBN':{'best':[],'last':[],'worst_source':[]},'clustering_finetune':{'best':[],'last':[],'worst_source':[]}}
all_scores = {'oracle':{},'pretrain':{},'their':{},'adaBN':{},'clustering_finetune':{}}
for (source,target) in combs:
    all_scores['oracle'][source] =  load(base_path/ f'source_{source}' / 'pretrain'/ 'scores_end.json')[f'{metric}_end/test_best']
    for mode in ['their','adaBN','clustering_finetune']:
        score_path  =base_path/f'source_{source}_target_{target}' / mode /'scores_end.json'
        scores = load(score_path)
        if mode == 'adaBN':
            all_scores[mode][f'source_{source}_target_{target}'] =scores[f'{metric}/test_best']
            # all_scores[mode]['best'].append(scores['sdice/test_best'])
            # all_scores[mode]['last'].append(scores['sdice/test_best'])
            # all_scores[mode]['worst_source'].append(scores['sdice/test_best'])
        else:
            if mode == 'clustering_finetune':
                all_scores['pretrain'][f'source_{source}_target_{target}'] = load(base_path/f'source_{source}_target_{target}' / mode /'scores_start.json')[f'{metric}_start/test']
                # all_scores['pretrain']['best'].append(load(base_path/f'source_{source}_target_{target}' / mode /'scores_start.json')['sdice_start/test'])
            all_scores[mode][f'source_{source}_target_{target}'] = scores[f'{metric}_end/test_low_source_on_target']
            # all_scores[mode]['best'].append(scores['sdice_end/test_best'])
            # all_scores[mode]['last'].append(scores['sdice_end/test'])
            # all_scores[mode]['worst_source'].append(scores['sdice_end/test_low_source_on_target'])

print(all_scores)
with open('/home/dsi/shaya/res.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for k,v in all_scores.items():
        spamwriter.writerow([k])
        for i in range(6):
            row  = []
            if k =='oracle':
                row = [round(v[i]*100,3)]
            else:
                for j in range(6):
                    if i ==j:
                        row.append('--')
                    else:

                        row.append(round(v[f'source_{i}_target_{j}']* 100,3))
            spamwriter.writerow(row)
        print(k,np.mean(list(v.values())),len(v))