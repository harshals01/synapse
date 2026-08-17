import re

def extract_entity(query):
    stop_phrases = ["tell me about", "what is", "what are", "explain", "describe", "summarize"]
    q = query.lower().strip()

    for phrase in stop_phrases:
        if q.startswith(phrase):
            entity = q.replace(phrase, "").strip()
            if entity:
                return entity
    return None


def rewrite_query_with_context(messages, logger):
    last_query = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        ""
    )

    pronouns = ["it", "this", "that", "they", "them"]

    pronoun_pattern = r'\b(' + '|'.join(pronouns) + r')\b'
    if not re.search(pronoun_pattern, last_query, flags=re.IGNORECASE):
        return last_query

    entity = None
    for m in reversed(messages[:-1]):
        if m["role"] == "user":
            if len(m["content"].split()) < 2:
                continue
            entity = extract_entity(m["content"])
            if entity:
                break

    if not entity:
        return last_query

    if entity.lower() in last_query.lower():
        return last_query

    rewritten = re.sub(pronoun_pattern, entity, last_query, flags=re.IGNORECASE)

    logger.info(f"Rewritten Query: {rewritten}")
    return rewritten