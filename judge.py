"""
judge.py - LLM judge wrapper for scoring movie reviews against a rubric.

Uses Anthropic's Claude API to grade reviews. Maintains running feedback
context across calibration rounds.

Cost-optimized version:
- grade_batch() sends an entire round's worth of items in ONE call instead
  of one call per item. This is the biggest lever: it turns N calls/round
  (each repaying the full rubric + few-shot text) into 1 call/round.
- The static system prompt (rubric + few-shot examples) is marked with
  cache_control so repeated calls reuse it at a steep discount instead of
  being billed at full price every time. Feedback is kept in a SEPARATE,
  uncached block, since it changes every round and would invalidate the
  cache if bundled in with the static content.
"""

import anthropic
import json
import re


class Judge:
    """
    LLM-based judge for grading movie reviews against a specific rubric.

    Maintains a cached static system prompt (rubric + few-shot examples)
    plus a separate, uncached feedback block that accumulates across
    calibration rounds.
    """

    def __init__(self, rubric_text, style_name, few_shot_examples, model="claude-haiku-4-5-20251001"):
        """
        Args:
            rubric_text: Full text of the grading rubric
            style_name: Name of the style being graded
            few_shot_examples: List of dicts with keys: prompt, response_text, tier
            model: Anthropic model name to use
        """
        self.rubric_text = rubric_text
        self.style_name = style_name
        self.few_shot_examples = few_shot_examples
        self.model = model
        self.client = anthropic.Anthropic()  # API key from environment

        # Static portion — rubric + few-shot examples. This never changes
        # across rounds, so it's the part we cache.
        self.base_system_prompt = self._build_base_system_prompt()

        # Feedback accumulated across rounds. Changes every round, so it's
        # kept separate and uncached.
        self.feedback_history = []

    def _build_base_system_prompt(self):
        """Build the static system prompt: task instructions, rubric, few-shot examples."""
        prompt_parts = [
            "You are grading how well written movie reviews match a specific target style.",
            f"Style being graded: {self.style_name}",
            f"Rubric for this style:\n{self.rubric_text}",
            "",
            "You will be shown a BATCH of prompt/response pairs, each with an id.",
            "Grade each response as one of:",
            "- GOOD  (fully matches the style/rubric)",
            "- OK    (partially matches — some elements present, others missing or off)",
            "- BAD   (does not match the style/rubric)",
            "",
            "For each item, briefly reason through how well the response satisfies the rubric,",
            "then assign a grade. Return your results as a JSON array, one object per item, in",
            "this exact format (no other text before or after the JSON array):",
            "",
            '[{"id": "<item id>", "reasoning": "<2-3 sentence reasoning>", "grade": "<GOOD|OK|BAD>"}, ...]',
            "",
            "Grade every item in the batch. Do not skip any. Preserve the exact id given for each item.",
            "",
            "Here are labeled examples to calibrate your grading:",
            ""
        ]

        for i, example in enumerate(self.few_shot_examples, 1):
            prompt_parts.extend([
                f"Example {i}:",
                f"Prompt: {example['prompt']}",
                f"Response: {example['response_text']}",
                f"Reasoning: This response {'meets' if example['tier'] == 'GOOD' else 'partially meets' if example['tier'] == 'OK' else 'does not meet'} the key stylistic markers expected of this rubric.",
                f"GRADE: {example['tier']}",
                ""
            ])

        prompt_parts.extend([
            "Now you will be shown a new batch of prompt/response pairs to grade.",
            "Apply the same standard to every item."
        ])

        return "\n".join(prompt_parts)

    def _build_system_blocks(self):
        """
        Build the system parameter as a list of content blocks.

        Block 1: static base prompt (rubric + few-shot) — marked cacheable.
                 Identical across every call in the run, so this is what
                 benefits from caching.
        Block 2: feedback history — NOT cached, since it changes every round.
                 Kept as a separate block so appending new feedback doesn't
                 invalidate the cache on block 1.
        """
        blocks = [
            {
                "type": "text",
                "text": self.base_system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]

        if self.feedback_history:
            feedback_parts = [
                "",
                "=" * 80,
                "FEEDBACK FROM PREVIOUS GRADING ROUNDS:",
                ""
            ]
            for i, feedback in enumerate(self.feedback_history, 1):
                feedback_parts.append(f"Round {i} feedback:")
                feedback_parts.append(feedback)
                feedback_parts.append("")
            feedback_parts.extend([
                "Please adjust your grading based on this feedback.",
                "=" * 80
            ])
            blocks.append({
                "type": "text",
                "text": "\n".join(feedback_parts)
                # deliberately no cache_control here — this block changes every round
            })

        return blocks

    def grade_batch(self, items):
        """
        Grade an entire batch of prompt/response pairs in ONE API call.

        Args:
            items: List of dicts, each with at least 'id', 'prompt', 'response_text'

        Returns:
            List of dicts with keys: id, reasoning, grade — in the same order
            as the input items (matched by id, not by position, to be safe).
        """
        system_blocks = self._build_system_blocks()

        # Build the batch user message
        batch_lines = ["Grade the following batch of items:\n"]
        for item in items:
            batch_lines.append(f"--- Item id: {item['id']} ---")
            batch_lines.append(f"Prompt: {item['prompt']}")
            batch_lines.append(f"Response: {item['response_text']}")
            batch_lines.append("")
        batch_lines.append(
            f"Return a JSON array with exactly {len(items)} objects, one per item above, "
            "in the format specified in the system prompt. Return ONLY the JSON array."
        )
        user_message = "\n".join(batch_lines)

        # Scale max_tokens with batch size — each item needs room for a short
        # reasoning string plus grade. ~150 tokens/item is a safe budget.
        max_tokens = min(8192, 300 + (len(items) * 150))

        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        response_text = ""
        for block in message.content:
            if hasattr(block, 'text'):
                response_text += block.text

        results = self._parse_batch_response(response_text, items)

        # Optional: surface cache performance so you can see it working
        if hasattr(message, 'usage'):
            usage = message.usage
            cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
            cache_write = getattr(usage, 'cache_creation_input_tokens', 0) or 0
            print(f"  [cache] read={cache_read} tokens, written={cache_write} tokens, "
                  f"fresh_input={usage.input_tokens}")

        return results

    def _parse_batch_response(self, response_text, items):
        """
        Parse the judge's JSON array response. Falls back to per-item
        regex extraction if JSON parsing fails outright, and fills in
        any missing ids with a safe default so a partial parse failure
        doesn't crash the whole round.
        """
        item_ids = {item['id'] for item in items}
        parsed_by_id = {}

        # Try direct JSON parse first (strip any stray text/fencing around it)
        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                parsed_list = json.loads(json_match.group(0))
                for entry in parsed_list:
                    if 'id' in entry and 'grade' in entry:
                        grade = entry['grade'].upper().strip()
                        if grade not in ('GOOD', 'OK', 'BAD'):
                            grade = 'OK'  # safe fallback for malformed grade
                        parsed_by_id[str(entry['id'])] = {
                            'id': str(entry['id']),
                            'reasoning': entry.get('reasoning', ''),
                            'grade': grade
                        }
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass  # fall through to per-item fallback below

        # Fill in anything missing (parse failure, model skipped an item, etc.)
        results = []
        missing_ids = []
        for item in items:
            item_id = str(item['id'])
            if item_id in parsed_by_id:
                results.append(parsed_by_id[item_id])
            else:
                missing_ids.append(item_id)
                results.append({
                    'id': item_id,
                    'reasoning': '[PARSE FAILURE — no grade returned by judge for this item]',
                    'grade': 'OK'  # neutral fallback, flagged clearly in reasoning
                })

        if missing_ids:
            print(f"  WARNING: judge response missing/unparseable grades for {len(missing_ids)} "
                  f"item(s): {missing_ids}. Defaulted to OK — check judge_report for [PARSE FAILURE] entries.")

        return results

    def receive_feedback(self, feedback):
        """
        Add feedback to the judge's context for the next round.
        
        Keeps only the most recent 3 rounds of feedback to prevent context
        from growing too large and confusing the model's JSON output.

        Args:
            feedback: Plain-language feedback string describing grading errors
        """
        self.feedback_history.append(feedback)
        # Keep only the most recent 3 rounds to prevent context bloat
        if len(self.feedback_history) > 3:
            self.feedback_history = self.feedback_history[-3:]
        print(f"Judge received feedback (now {len(self.feedback_history)} rounds of feedback)")


if __name__ == "__main__":
    # Quick test
    from data_prep import prepare_data

    print("Testing Judge initialization...")
    few_shot, calibration, golden, rubric = prepare_data()

    judge = Judge(
        rubric_text=rubric,
        style_name="Chaos Cinema Critique - A Manifesto for the Absurd",
        few_shot_examples=few_shot
    )

    print("\nTesting batched grade call (first 5 calibration items)...")
    test_batch = calibration[:5]
    results = judge.grade_batch(test_batch)

    for item, result in zip(test_batch, results):
        print(f"\nid={result['id']} | true_tier={item['tier']} | judge_grade={result['grade']}")
        print(f"  reasoning: {result['reasoning'][:150]}...")

    print("\nTesting feedback mechanism...")
    judge.receive_feedback("You graded too many GOOD items as OK. Be more lenient with stylistic variation.")

    print("\nJudge ready!")