"""Corpus absence check for proposed unanswerable queries.

For each proposed unanswerable query, count the chunks containing each of its key terms
(case-insensitive, whole word or phrase). A query is kept as "proposed_unanswerable" only
when at least one *required* term has zero matches in the corpus. This is evidence for a
human annotator, not a label: the annotator still confirms every query.

    python evaluation/datasets/check_unanswerable.py
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHUNKS = os.path.join(ROOT, "Data", "chunks", "chunks.jsonl")

# (query, [required terms: the topic the corpus would have to mention to answer it])
PROPOSED = [
    ("Does the university provide any scholarship for purchasing a laptop?", ["laptop scholarship", "laptop loan", "laptop grant", "laptop subsidy"]),
    ("How do I book the campus swimming pool?", ["swimming"]),
    ("What is the gym membership fee?", ["gym", "gymnasium", "fitness centre", "fitness center"]),
    ("Can I keep a pet dog in my hostel room?", ["pet", "pets", "dog", "dogs"]),
    ("Is flying a drone allowed on campus?", ["drone", "drones", "uav"]),
    ("What is the maternity leave policy for faculty?", ["maternity"]),
    ("How many days of paternity leave can staff take?", ["paternity"]),
    ("When is the annual sports day held?", ["sports day"]),
    ("Where can I rent a convocation gown?", ["gown", "gowns"]),
    ("What are the canteen food prices?", ["menu price", "food price", "meal price", "price list"]),
    ("Is there a campus ATM?", ["atm"]),
    ("How do I get my printing quota increased?", ["printing quota", "print quota"]),
    ("Does the university offer a bicycle sharing scheme?", ["bicycle", "cycle sharing", "bike sharing"]),
    ("What is the COVID-19 vaccination requirement for students?", ["covid", "vaccination", "vaccine"]),
    ("How do I apply for a sabbatical as a professor?", ["sabbatical"]),
    ("What are the criteria for faculty promotion to full professor?", ["promotion to professor", "faculty promotion", "career advancement scheme"]),
    ("When is the faculty salary credited each month?", ["salary credit", "payroll date", "pay day", "payday"]),
    ("Can students do part-time jobs off campus during the semester?", ["part-time job", "part time job", "part-time employment", "off-campus employment"]),
    ("How do I book the music practice room?", ["music room", "music practice"]),
    ("Is there a laundry service in the hostel?", ["laundry"]),
    ("Can my parents stay overnight in the hostel guest room?", ["guest room", "guest house"]),
    ("What is the dress code for students on campus?", ["dress code"]),
    ("Does the university have a creche or day care for staff children?", ["creche", "day care", "daycare"]),
    ("How do I transfer credits from a MOOC like Coursera?", ["mooc", "coursera", "nptel", "swayam"]),
    ("What is the fee for a duplicate parking sticker?", ["parking sticker", "parking permit"]),
    ("Can I bring my car to the hostel?", ["car parking", "four-wheeler", "four wheeler"]),
    ("What is the procedure to change my name on the degree certificate after marriage?", ["name change", "change of name"]),
    ("Is there a quota for sports students in admission?", ["sports quota"]),
    ("How do I join the university NCC unit?", ["ncc", "national cadet corps"]),
    ("What is the refund policy for the campus bookstore?", ["bookstore", "book store"]),
    ("Are there any scholarships for students from Nepal?", ["nepal"]),
    ("How do I apply for an education loan through the university bank tie-up?", ["education loan", "bank tie-up", "loan scheme"]),
    ("What is the night shuttle bus timing?", ["night shuttle"]),
    ("Can I get a refund for unused mess coupons?", ["mess coupon", "coupons"]),
    ("Does the library lend board games?", ["board game", "board games"]),
    ("Are yoga classes free for students?", ["yoga"]),
    ("Does the campus have a movie theatre?", ["movie", "cinema", "theatre", "theater"]),
    ("Can I use e-cigarettes in designated smoking zones?", ["e-cigarette", "e-cigarettes", "vaping", "vape"]),
    ("Is there a barber shop or salon on campus?", ["barber", "salon"]),
    ("What is the fine for plucking flowers in the campus garden?", ["flowers", "garden"]),
]


def main() -> None:
    texts = []
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                texts.append((c["chunk_id"], " ".join(str(c.get(k) or "") for k in ("title", "section", "text")).lower()))
    out = []
    for query, terms in PROPOSED:
        counts = {}
        for t in terms:
            pat = re.compile(r"(?<![a-z0-9])" + re.escape(t.lower()) + r"(?![a-z0-9])")
            hits = [cid for cid, txt in texts if pat.search(txt)]
            counts[t] = {"chunks": len(hits), "examples": hits[:3]}
        absent = all(v["chunks"] == 0 for v in counts.values())
        out.append({"query": query, "term_counts": counts, "all_terms_absent": absent})
        flag = "ABSENT " if absent else "PRESENT"
        print(f"{flag} {query[:70]:70s} " + ", ".join(f"{t}={v['chunks']}" for t, v in counts.items()))
    path = os.path.join(ROOT, "evaluation", "datasets", "unanswerable_absence_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(sum(o["all_terms_absent"] for o in out), "of", len(out), "proposals have no matching corpus terms")


if __name__ == "__main__":
    sys.exit(main())
