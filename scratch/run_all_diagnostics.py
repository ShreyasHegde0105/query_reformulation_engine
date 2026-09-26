import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from evaluation.diagnose_query import diagnose_query, print_diagnostic_report

failed_query_ids = ["Q004", "Q005", "Q008", "Q014", "Q016", "Q017", "Q020", "Q024", "Q027", "Q047"]
typo_query_ids = ["Q016", "Q017", "Q018", "Q019", "Q020"]

all_ids = list(dict.fromkeys(failed_query_ids + typo_query_ids))

results = {}
for qid in all_ids:
    diag = diagnose_query(qid)
    results[qid] = diag
    print_diagnostic_report(diag)

with open("scratch/diagnostics_dump.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)

print("Saved diagnostics dump for all investigated queries.")
