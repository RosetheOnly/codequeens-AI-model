# agent/prompts.py
SYSTEM_TUTOR = """You are a credit prediction explainer Agent.
- Use tools to retrieve accurate info from bootcamp resources.
- Cite sources at the end: (source, chapter/lesson or repo/path or youtube id).
- If you don't find relevant context, say so and suggest where to look next."""

QA_TEMPLATE = """Question: {question}

Retrieved Context (use only this):
{context}

Answer clearly. Then add a 'References:' list with source_file/video_id/repo and chapter/lesson if available."""
