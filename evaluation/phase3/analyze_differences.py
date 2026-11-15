import json

def analyze():
    with open('evaluation/phase3/phase3_per_query_comparison.json', encoding='utf-8') as f:
        data = json.load(f)

    print('=== R1 vs R0 differences (Answerable N=94) ===')
    r1_gains = []
    r1_losses = []
    for qid, q in data.items():
        if not q['answerable']:
            continue
        eval_c = q['systems']['system_c']['answer_evaluation']
        eval_r0 = q['systems']['config_r0']['answer_evaluation']
        s_c = (eval_c['factual_correctness'] + eval_c['completeness'] + eval_c['citation_correctness']) / 3.0
        s_r0 = (eval_r0['factual_correctness'] + eval_r0['completeness'] + eval_r0['citation_correctness']) / 3.0
        diff = s_c - s_r0
        if diff > 0.01:
            r1_gains.append((qid, q['category'], q['multi_document'], diff, s_c, s_r0, q['original_query'], eval_c, eval_r0))
        elif diff < -0.01:
            r1_losses.append((qid, q['category'], q['multi_document'], diff, s_c, s_r0, q['original_query'], eval_c, eval_r0))

    print(f'R1 Gains ({len(r1_gains)}):')
    for g in r1_gains:
        print(f'  {g[0]} [{g[1]} | multi={g[2]}]: diff={g[3]:+.2f} (R1={g[4]:.2f}, R0={g[5]:.2f}) - {g[6]}')
        print(f'     R1: fact={g[7]["factual_correctness"]}, comp={g[7]["completeness"]}, cit={g[7]["citation_correctness"]}')
        print(f'     R0: fact={g[8]["factual_correctness"]}, comp={g[8]["completeness"]}, cit={g[8]["citation_correctness"]}')

    print(f'\nR1 Losses ({len(r1_losses)}):')
    for l in r1_losses:
        print(f'  {l[0]} [{l[1]} | multi={l[2]}]: diff={l[3]:+.2f} (R1={l[4]:.2f}, R0={l[5]:.2f}) - {l[6]}')
        print(f'     R1: fact={l[7]["factual_correctness"]}, comp={l[7]["completeness"]}, cit={l[7]["citation_correctness"]}')
        print(f'     R0: fact={l[8]["factual_correctness"]}, comp={l[8]["completeness"]}, cit={l[8]["citation_correctness"]}')

    print('\n=== System C vs System A (N=94) ===')
    c_gains = []
    c_losses = []
    for qid, q in data.items():
        if not q['answerable']:
            continue
        eval_c = q['systems']['system_c']['answer_evaluation']
        eval_a = q['systems']['system_a']['answer_evaluation']
        s_c = (eval_c['factual_correctness'] + eval_c['completeness'] + eval_c['citation_correctness']) / 3.0
        s_a = (eval_a['factual_correctness'] + eval_a['completeness'] + eval_a['citation_correctness']) / 3.0
        diff = s_c - s_a
        if diff > 0.01:
            c_gains.append((qid, q['category'], q['multi_document'], diff, s_c, s_a, q['original_query']))
        elif diff < -0.01:
            c_losses.append((qid, q['category'], q['multi_document'], diff, s_c, s_a, q['original_query']))

    print(f'System C Gains vs System A: {len(c_gains)}')
    print(f'System C Losses vs System A: {len(c_losses)}')

if __name__ == '__main__':
    analyze()
