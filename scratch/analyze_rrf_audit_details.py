import json
import os

with open('evaluation/results/rrf_weight_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)['experiments']

exp10 = data['weight_1.0']
exp15 = data['weight_1.5']
exp20 = data['weight_2.0']

q10 = {r['query_id']: r for r in exp10['per_query_results']}
q15 = {r['query_id']: r for r in exp15['per_query_results']}
q20 = {r['query_id']: r for r in exp20['per_query_results']}

d10 = exp10['per_query_details']
d15 = exp15['per_query_details']
d20 = exp20['per_query_details']

print("=== 1.5 vs 1.0 Attribution ===")
imp_15 = []
reg_15 = []
same_15 = []
for qid in sorted(q10.keys()):
    m10 = q10[qid]['metrics']
    m15 = q15[qid]['metrics']
    # Deterministic rule: MRR change takes precedence, then NDCG@10 / Recall@10
    if m15['mrr'] > m10['mrr'] or m15['recall@10'] > m10['recall@10'] or m15['ndcg@10'] > m10['ndcg@10']:
        imp_15.append((qid, q10[qid]['query'], m10, m15))
    elif m15['mrr'] < m10['mrr'] or m15['recall@10'] < m10['recall@10'] or m15['ndcg@10'] < m10['ndcg@10']:
        reg_15.append((qid, q10[qid]['query'], m10, m15))
    else:
        same_15.append((qid, q10[qid]['query']))

print(f"Improved ({len(imp_15)}): {[x[0] for x in imp_15]}")
for x in imp_15:
    print(f"  {x[0]}: {x[1]}")
    print(f"    1.0: {x[2]}")
    print(f"    1.5: {x[3]}")
print(f"Regressed ({len(reg_15)}): {[x[0] for x in reg_15]}")
for x in reg_15:
    print(f"  {x[0]}: {x[1]}")
    print(f"    1.0: {x[2]}")
    print(f"    1.5: {x[3]}")
print(f"Unchanged ({len(same_15)}): {[x[0] for x in same_15]}")

print("\n=== 2.0 vs 1.0 Attribution ===")
imp_20 = []
reg_20 = []
same_20 = []
for qid in sorted(q10.keys()):
    m10 = q10[qid]['metrics']
    m20 = q20[qid]['metrics']
    if m20['mrr'] > m10['mrr'] or m20['recall@10'] > m10['recall@10'] or m20['ndcg@10'] > m10['ndcg@10']:
        imp_20.append((qid, q10[qid]['query'], m10, m20))
    elif m20['mrr'] < m10['mrr'] or m20['recall@10'] < m10['recall@10'] or m20['ndcg@10'] < m10['ndcg@10']:
        reg_20.append((qid, q10[qid]['query'], m10, m20))
    else:
        same_20.append((qid, q10[qid]['query']))

print(f"Improved ({len(imp_20)}): {[x[0] for x in imp_20]}")
for x in imp_20:
    print(f"  {x[0]}: {x[1]}")
    print(f"    1.0: {x[2]}")
    print(f"    2.0: {x[3]}")
print(f"Regressed ({len(reg_20)}): {[x[0] for x in reg_20]}")
for x in reg_20:
    print(f"  {x[0]}: {x[1]}")
    print(f"    1.0: {x[2]}")
    print(f"    2.0: {x[3]}")
print(f"Unchanged ({len(same_20)}): {[x[0] for x in same_20]}")

print("\n=== Failure Queries (Q004, Q005, Q008, Q014, Q016, Q017, Q020, Q024, Q027, Q047) ===")
failure_qids = ["Q004", "Q005", "Q008", "Q014", "Q016", "Q017", "Q020", "Q024", "Q027", "Q047"]
for fqid in failure_qids:
    q_info = d10[fqid]
    gt = q_info['relevant_chunks']
    print(f"\n--- {fqid} ({q_info['category']}): '{q_info['query']}' | GT: {gt} ---")
    for w_name, d_exp in [("Weight 1.0", d10), ("Weight 1.5", d15), ("Weight 2.0", d20)]:
        chunks = d_exp[fqid]['retrieved_chunks']
        # Find target ranks
        gt_ranks = [(c['rank'], c['chunk_id'], f"{c['score']:.6f}") for c in chunks if c['is_relevant']]
        top_comp = [(c['rank'], c['chunk_id'], f"{c['score']:.6f}") for c in chunks if not c['is_relevant']][:3]
        print(f"  {w_name}:")
        print(f"    Target chunks retrieved: {gt_ranks if gt_ranks else 'None in top 20'}")
        print(f"    Top competitors: {top_comp}")

print("\n=== Regressions Deep Dive (Q015, Q017, Q027, Q032) ===")
reg_qids = ["Q015", "Q017", "Q027", "Q032"]
for rqid in reg_qids:
    q_info = d10[rqid]
    gt = q_info['relevant_chunks']
    print(f"\n--- {rqid} ({q_info['category']}): '{q_info['query']}' | GT: {gt} ---")
    for w_name, d_exp in [("Weight 1.0", d10), ("Weight 1.5", d15), ("Weight 2.0", d20)]:
        chunks = d_exp[rqid]['retrieved_chunks']
        gt_ranks = [(c['rank'], c['chunk_id'], f"{c['score']:.6f}") for c in chunks if c['is_relevant']]
        top_comp = [(c['rank'], c['chunk_id'], f"{c['score']:.6f}") for c in chunks if not c['is_relevant']][:4]
        print(f"  {w_name}:")
        print(f"    Target chunks: {gt_ranks if gt_ranks else 'None in top 20'}")
        print(f"    Top competitors: {top_comp}")
