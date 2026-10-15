import re

# ---------------- Defense 1: regex filter ----------------
# Phrases that only make sense as commands aimed at an AI, not as company policy.
INJECTION_PATTERNS = [
    r"note to (the )?(ai|assistant|chatbot|model)",
    r"(ignore|disregard|forget) (all |any )?(the )?(previous|prior|above|earlier|other) (instructions|rules|pages|documents)",
    r"(do not|don't|never) (mention|reveal|tell the user about)",
    r"(always|only) (tell|give|say|answer|respond|recommend|point)",
    r"you (are now|must now|should now)",
    r"system prompt",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def regex_filter(docs):
    """Block any chunk that matches an injection pattern."""
    kept, blocked = [], []
    for doc in docs:
        match = next((p.pattern for p in _COMPILED if p.search(doc.page_content)), None)
        if match:
            blocked.append({
                "source": doc.metadata.get("source"),
                "section": doc.metadata.get("h2", "-"),
                "reason": f"regex: {match}",
            })
        else:
            kept.append(doc)
    return kept, blocked


# All defenses by name. "none" = the baseline bot.
DEFENSES = {
    "none": None,
    "regex": regex_filter,
}