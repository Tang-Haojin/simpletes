"""
Generation prompt templates.
"""

INSPIRATION_TEMPLATE = """\

--- Inspiration {index} ---
Score: {score}
Metrics:
{metrics_text}{reflection_block}
Code:
```{code_fence_tag}
{code}
```
"""

FAILURE_PATTERNS_TEMPLATE = """\

[FAILURE PATTERNS] (common errors to avoid)
{failure_lines}
"""

GENERATION_PROMPT_TEMPLATE = """\
Task: {instruction}

Generation instruction (must follow exactly):
1) Only the code between `{start_marker_line}` and `{end_marker_line}` is extracted.
2) The final program is reconstructed as EXACT_PREFIX + evolved_block + EXACT_SUFFIX.
3) Keep marker lines exactly as written.
4) Return one {language_name} code block that includes both EVOLVE-BLOCK markers.

EXACT_PREFIX (kept unchanged):
```{code_fence_tag}
{prefix}
```

EXACT_SUFFIX (kept unchanged):
```{code_fence_tag}
{suffix}
```
{available_packages_text}
=== REFERENCE SOLUTIONS ===
{policy_context_section}
[SAMPLED INSPIRATIONS] ({num_inspirations} solutions sampled for detailed reference)
Learn from these specific implementations - study their patterns and techniques.
{inspirations_text}
{failure_text}
=== GENERATION STRATEGY ===
- Prioritize NOVEL approaches not yet seen in the elite pool
- Only refine existing approaches if you identify clear improvement potential
- Combine insights from multiple solutions when beneficial
- Avoid the listed failure patterns

Generate an improved solution with higher score:
"""


STRUCTURED_GENERATION_PROMPT_TEMPLATE = """\
Task: {instruction}

Structured generation instruction (must follow exactly):
1) Return exactly one JSON object that conforms to the response contract described by the task and enforced by the backend.
2) Return only that JSON object: no Markdown code fence, no EVOLVE-BLOCK marker, no commentary, and no surrounding text.
3) Put the complete candidate payload in the object. Solve the task rather than returning an example or placeholder.
4) The backend validates the object and adds the extraction markers after generation; do not add those markers yourself.
{available_packages_text}
=== REFERENCE SOLUTIONS ===
{policy_context_section}
[SAMPLED INSPIRATIONS] ({num_inspirations} solutions sampled for detailed reference)
Learn from these specific implementations - study their patterns and techniques.
Code fences in the references below quote prior inputs only; they do not change the required JSON-only response format.
{inspirations_text}
{failure_text}
=== GENERATION STRATEGY ===
{generation_strategy}

Return the improved candidate as the single contract-conforming JSON object now:
"""
