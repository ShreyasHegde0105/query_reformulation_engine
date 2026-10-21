import json
import unittest

from reformulation.glossary import build_glossary, load_glossary


DOCS = [
    {"title": "Acceptable Use Policy", "department": "Information Technology Services",
     "systems": ["Campus Wi-Fi (802.1X)", "ResolveDesk"], "keywords": ["web filtering"],
     "retrieval_signals": ["websites blocked on campus"]},
    {"title": "Hostel Curfew Rules", "department": "Hostel Administration and Residence Life",
     "systems": ["DormTrack", "resolvedesk"]},
    {"title": "acceptable use policy", "department": "Information Technology Services", "systems": []},
]


class TestBuildGlossary(unittest.TestCase):
    def test_sections_sorted_and_deduplicated(self):
        glossary = build_glossary(DOCS)
        self.assertEqual(glossary.splitlines(), [
            "Departments: Hostel Administration and Residence Life; Information Technology Services",
            "Systems and portals: Campus Wi-Fi (802.1X); DormTrack; ResolveDesk",
            "Document titles:",
            "- Acceptable Use Policy",
            "- Hostel Curfew Rules",
        ])

    def test_excludes_keywords_and_retrieval_signals(self):
        glossary = build_glossary(DOCS)
        self.assertNotIn("web filtering", glossary)
        self.assertNotIn("websites blocked", glossary)

    def test_empty_documents(self):
        self.assertEqual(build_glossary([]), "")


class TestLoadGlossary(unittest.TestCase):
    def test_loads_documents_wrapper(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "docs.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"metadata": {}, "documents": DOCS}, f)
            self.assertEqual(load_glossary(path), build_glossary(DOCS))

    def test_default_corpus_file(self):
        glossary = load_glossary()
        self.assertIn("Document titles:", glossary)
        self.assertIn("LearnSphere LMS", glossary)


if __name__ == "__main__":
    unittest.main()
