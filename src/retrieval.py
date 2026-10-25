
"""Simple local retrieval over the supplied ZentroMart Markdown knowledge base."""

from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeBase:
    def __init__(self, kb_dir):
        self.chunks = []

        # Load Markdown knowledge-base files.
        for path in sorted(Path(kb_dir).glob("*.md")):
            raw = path.read_text(encoding="utf-8")

            # Read document status, if present.
            status = re.search(r"(?mi)^status:\s*(.+)$", raw)
            doc_status = (
                status.group(1).strip().lower()
                if status
                else "unknown"
            )

            # Exclude superseded policies.
            if (
                doc_status == "superseded"
                or "superseded" in path.name.lower()
            ):
                continue

            # Split documents into sections while preserving section IDs.
            parts = re.split(r"(?m)(?=^##\s+\[)", raw)

            for part in parts:
                part = part.strip()

                if len(part) < 30:
                    continue

                match = re.search(
                    r"(?m)^##\s+\[([^\]]+)\]\s*(.*)$",
                    part,
                )

                section_id = match.group(1) if match else path.stem
                title = match.group(2).strip() if match else path.stem

                self.chunks.append({
                    "id": section_id,
                    "title": title,
                    "source": path.name,
                    "text": part,
                })

        if not self.chunks:
            raise RuntimeError(
                f"No knowledge documents found in {kb_dir}"
            )

        # Convert knowledge-base sections into TF-IDF vectors.
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

        self.analyzer = self.vectorizer.build_analyzer()

    def search(self, query, k=4):
        """Return the most relevant knowledge-base sections."""

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(
            query_vector, self.matrix
        ).ravel()

        query_lower = query.lower()

        # Ignore generic terms when checking topic overlap.
        generic_terms = {
            "zentromart",
            "customer",
            "support",
            "please",
            "tell",
            "help",
            "know",
            "information",
        }

        query_terms = set(self.analyzer(query)) - generic_terms

        if not query_terms:
            return []

        # Identify the main topic of the user's question.
        damaged_query = bool(re.search(
            r"\b(damaged|damage|defective|wrong|missing|expired)\b",
            query_lower,
        ))

        safety_query = bool(re.search(
            r"\b(swelling battery|burning smell|sparks|overheating|"
            r"overheat|injury|safety risk)\b",
            query_lower,
        ))

        high_value_query = bool(re.search(
            r"(?:₹\s*)?10,?000|high[- ]value|worth more",
            query_lower,
        ))

        refund_query = bool(re.search(
            r"\b(refund|replacement|replace|return|money back)\b",
            query_lower,
        ))

        # Add related terms to help match natural customer wording.
        if damaged_query:
            query_terms.update({
                "item",
                "items",
                "replacement",
                "refund",
                "photos",
                "photo",
                "48",
                "hours",
                "delivery",
            })

        if refund_query:
            query_terms.update({
                "refund",
                "replacement",
                "return",
                "delivery fees",
            })

        ranked = []

        for index, raw_score in enumerate(scores):
            chunk = self.chunks[index]
            score = float(raw_score)
            boost = 0.0
            section_id = chunk["id"]

            # Prioritize the damaged/wrong/missing item policy.
            if damaged_query and section_id in {"DWM-1", "DWM-2"}:
                boost += 0.35

            # Safety issues must prioritize the safety policy and handoff.
            if safety_query and section_id in {"DWM-5", "ESC-1"}:
                boost += 0.50

            # High-value claims need the dedicated policy and handoff.
            if high_value_query and section_id in {"DWM-3", "ESC-1"}:
                boost += 0.40

            # Refund and replacement questions should find resolution rules.
            if refund_query and section_id == "DWM-2":
                boost += 0.20

            ranked.append((
                index,
                score + boost,
                score,
            ))

        # Sort by adjusted relevance score, highest first.
        ranked.sort(key=lambda item: item[1], reverse=True)

        results = []

        for index, ranking_score, raw_score in ranked:
            # Ignore results that have neither sufficient similarity
            # nor a targeted policy boost.
            if ranking_score < 0.10:
                continue

            chunk = self.chunks[index]
            document_terms = set(self.analyzer(chunk["text"]))
            overlap = len(query_terms & document_terms)

            minimum_overlap = 2 if len(query_terms) >= 3 else 1

            # A specific policy boost can allow a relevant section
            # through even when the user's wording differs from the KB.
            targeted_match = (
                damaged_query
                and chunk["id"] in {"DWM-1", "DWM-2"}
            ) or (
                safety_query
                and chunk["id"] in {"DWM-5", "ESC-1"}
            ) or (
                high_value_query
                and chunk["id"] in {"DWM-3", "ESC-1"}
            ) or (
                refund_query
                and chunk["id"] == "DWM-2"
            )

            if overlap < minimum_overlap and not targeted_match:
                continue

            results.append({
                **chunk,
                "score": raw_score,
            })

            if len(results) >= k:
                break

        return results
