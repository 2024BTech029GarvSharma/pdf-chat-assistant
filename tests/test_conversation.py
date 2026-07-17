import unittest

from utils.qa import build_messages, select_context


class FakeStore:
    def __init__(self):
        self.calls = []

    def search(self, question, top_k):
        self.calls.append((question, top_k))
        return [{"text": "fresh source", "source": "guide.pdf", "page": 1}]


HISTORY = [{
    "question": "What is the difference between Mitra and admin?",
    "answer": "Mitra receives per-candidate cash; admin has platform duties.",
    "sources": [{"text": "Mitra per-candidate cash", "source": "guide.pdf", "page": 4}],
}]


class ConversationRoutingTests(unittest.TestCase):
    def test_new_question_searches_the_vector_store(self):
        store = FakeStore()
        chunks, mode = select_context("What can an admin do?", store, HISTORY)
        self.assertEqual(mode, "new_question")
        self.assertEqual(store.calls, [("What can an admin do?", 8)])
        self.assertEqual(chunks[0]["text"], "fresh source")

    def test_follow_up_reuses_previous_sources_without_searching(self):
        store = FakeStore()
        question = "As you said, what does per-candidate cash mean?"
        chunks, mode = select_context(question, store, HISTORY)
        self.assertEqual(mode, "follow_up")
        self.assertEqual(chunks, HISTORY[-1]["sources"])
        self.assertEqual(store.calls, [])

    def test_acknowledgement_does_not_search(self):
        store = FakeStore()
        chunks, mode = select_context("okay", store, HISTORY)
        self.assertEqual(mode, "acknowledgement")
        self.assertEqual(chunks, [])
        self.assertEqual(store.calls, [])

    def test_history_and_document_context_are_both_supplied(self):
        messages = build_messages(
            "What does that mean?", HISTORY[-1]["sources"], HISTORY
        )
        self.assertEqual(messages[1]["content"], HISTORY[0]["question"])
        self.assertIn("Document context for this turn", messages[-1]["content"])
        self.assertIn("only evidence for factual claims", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
