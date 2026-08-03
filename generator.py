"""
generator.py - LLM generator for writing movie reviews in target style.

Structurally similar to judge.py: static system prompt (rubric + few-shot)
with cache_control, batched generation in ONE API call per round, and
per-item feedback kept separate from the cached static block.

Fix applied: claude-sonnet-5 runs adaptive thinking by default even when no
`thinking` param is passed (unlike earlier models, where omitting it meant
no thinking). Thinking tokens count against max_tokens, so on the original
budget this silently ate into the space needed for review text, causing
truncated responses and downstream JSON parse failures ("Unterminated
string..."). Fix: explicitly disable thinking (not needed for this
creative-writing task) and raise the token budget, since (1) Sonnet 5's
tokenizer produces ~30% more tokens for the same text than earlier models,
and (2) with thinking off, the full budget now goes to actual review text.

Default model switched from claude-sonnet-5 to claude-haiku-4-5-20251001:
same model already used by judge.py, cheaper per token, and predates the
Sonnet 5 tokenizer change (so less token inflation for the same review
length). thinking={"type": "disabled"} is kept regardless of model, since
it costs nothing and removes ambiguity either way. One tradeoff worth
knowing: Haiku is a smaller/faster model, so review quality/creativity on
an elaborate style like this may have a lower ceiling than Sonnet 5 --
worth watching for in the round-over-round quality trend, since a flat
curve could now mean "hit Haiku's capability ceiling" rather than "feedback
loop isn't working."
"""

import anthropic
import json
import re


class Generator:
    """
    LLM-based generator for writing movie reviews in a target style.
    
    Maintains a cached static system prompt (style + rubric + few-shot examples)
    plus a separate, uncached per-item feedback block that accumulates across rounds.
    """
    
    def __init__(self, rubric_text, style_name, few_shot_examples, model="claude-haiku-4-5-20251001"):
        """
        Initialize the generator with style guidance and few-shot examples.
        
        Args:
            rubric_text: Full text of the style rubric
            style_name: Name of the style to write in
            few_shot_examples: List of dicts with keys: prompt, response_text, tier
            model: Anthropic model name to use
        """
        self.rubric_text = rubric_text
        self.style_name = style_name
        self.few_shot_examples = few_shot_examples
        self.model = model
        self.client = anthropic.Anthropic()
        
        # Static portion - rubric + few-shot examples (cached)
        self.base_system_prompt = self._build_base_system_prompt()
        
        # Per-item feedback accumulated across rounds (uncached)
        self.feedback_history = []
    
    def _build_base_system_prompt(self):
        """Build the static system prompt with style guidance and few-shot examples."""
        prompt_parts = [
            "You are writing movie reviews in a specific target style.",
            f"Style: {self.style_name}",
            f"Style guidance:\n{self.rubric_text}",
            "",
            "Here are examples of reviews written in this style, including one partial-match",
            "example so you can see what a near-miss looks like versus a strong hit:",
            ""
        ]
        
        # Add few-shot examples with tier labels
        # Mix includes: GOOD (strong matches), OK (partial match), BAD (anti-example)
        for i, example in enumerate(self.few_shot_examples, 1):
            if example['tier'] == 'GOOD':
                label = "(strong match - emulate this quality)"
            elif example['tier'] == 'OK':
                label = "(partial match - some elements present)"
            else:  # BAD
                label = "(does NOT match - avoid this style)"
            
            prompt_parts.extend([
                f"Example {i} {label}:",
                f"Prompt: {example['prompt']}",
                f"Response: {example['response_text']}",
                ""
            ])
        
        prompt_parts.extend([
            "You will be given a BATCH of prompts (movies/topics to review).",
            "For each prompt, write a review in the target style.",
            "Keep each review roughly 150-300 words. Favor a few vivid, well-chosen",
            "images over exhaustively listing many -- the style comes from voice and",
            "rhythm, not from length.",
            "Return your results as a JSON array, one object per prompt, in this exact format",
            "(no other text before or after the JSON array):",
            "",
            '[{"prompt_id": "<id>", "review": "<your generated review text>"}, ...]',
            "",
            "Write every review in the batch. Do not skip any. Preserve the exact prompt_id",
            "given for each prompt. Output ONLY the review text in the 'review' field,",
            "no preamble or meta-commentary."
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_system_blocks(self):
        """
        Build system prompt as content blocks: static (cached) + feedback (uncached).
        
        Same pattern as Judge: keep feedback separate so it doesn't invalidate
        the cache on the static rubric + few-shot content.
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
                "FEEDBACK FROM JUDGE ON PREVIOUS ROUNDS:",
                "",
                "The judge scored your previous generations and provided specific feedback.",
                "Use this to improve your reviews:",
                ""
            ]
            
            # Per-item feedback (not aggregate like judge got)
            for feedback_item in self.feedback_history:
                feedback_parts.append(feedback_item)
                feedback_parts.append("")
            
            feedback_parts.extend([
                "Apply these lessons to write better reviews in the target style.",
                "=" * 80
            ])
            
            blocks.append({
                "type": "text",
                "text": "\n".join(feedback_parts)
                # No cache_control - this changes every round
            })
        
        return blocks
    
    def generate_batch(self, prompts):
        """
        Generate reviews for a batch of prompts in ONE API call.
        
        Args:
            prompts: List of dicts with keys: id, prompt
            
        Returns:
            List of dicts with keys: prompt_id, prompt, generated_review
        """
        system_blocks = self._build_system_blocks()
        
        # Build batch user message
        batch_lines = ["Write reviews for the following batch of prompts:\n"]
        for item in prompts:
            batch_lines.append(f"--- Prompt ID: {item['id']} ---")
            batch_lines.append(f"Topic: {item['prompt']}")
            batch_lines.append("")
        
        batch_lines.append(
            f"Return a JSON array with exactly {len(prompts)} objects, one per prompt above, "
            "in the format specified in the system prompt. Return ONLY the JSON array."
        )
        user_message = "\n".join(batch_lines)
        
        # Scale max_tokens - reviews can be longer than judge grades.
        # NOTE: the first version of this budget (800 base + 900/item, capped
        # at 16000) still undershot -- an 8-item batch hit exactly 8000/8000
        # and truncated. Raised substantially here; the length guidance added
        # to the system prompt above should also help keep actual usage well
        # under this ceiling rather than relying on the ceiling alone.
        # Haiku 4.5 supports up to 64K output tokens, so there's real
        # headroom to raise this without hitting a hard model limit.
        max_tokens = min(40000, 1200 + (len(prompts) * 1800))
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            # claude-sonnet-5 runs adaptive thinking BY DEFAULT even when no
            # `thinking` param is passed -- unlike earlier models, where
            # omitting it meant no thinking. This was silently eating into
            # max_tokens and causing the truncation / JSON parse failures
            # ("Unterminated string...") seen in debugging. Not needed for
            # a creative-writing task like this, so disable explicitly.
            thinking={"type": "disabled"},
            system=system_blocks,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        # Extract response text
        response_text = ""
        for block in message.content:
            if hasattr(block, 'text'):
                response_text += block.text
        
        results = self._parse_batch_response(response_text, prompts)
        
        # Show cache performance + surface truncation immediately if it
        # happens again, rather than only discovering it via parse failures
        if hasattr(message, 'usage'):
            usage = message.usage
            cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
            cache_write = getattr(usage, 'cache_creation_input_tokens', 0) or 0
            print(f"  [cache] read={cache_read} tokens, written={cache_write} tokens, "
                  f"fresh_input={usage.input_tokens}, output={usage.output_tokens}/{max_tokens}")

        if message.stop_reason == "max_tokens":
            print(f"  WARNING: response hit max_tokens ({max_tokens}) and was truncated. "
                  f"Increase the per-item token budget in generate_batch() if this recurs.")
        
        return results
    
    def _parse_batch_response(self, response_text, prompts):
        """Parse the generator's JSON array response."""
        parsed_by_id = {}
        
        # Try direct JSON parse
        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                parsed_list = json.loads(json_match.group(0))
                for entry in parsed_list:
                    if 'prompt_id' in entry and 'review' in entry:
                        parsed_by_id[str(entry['prompt_id'])] = entry['review']
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        
        # Build results, with fallback for missing items
        results = []
        missing_ids = []
        for item in prompts:
            item_id = str(item['id'])
            if item_id in parsed_by_id:
                results.append({
                    'prompt_id': item_id,
                    'prompt': item['prompt'],
                    'generated_review': parsed_by_id[item_id]
                })
            else:
                missing_ids.append(item_id)
                results.append({
                    'prompt_id': item_id,
                    'prompt': item['prompt'],
                    'generated_review': '[GENERATION FAILURE - no review returned by generator]'
                })
        
        if missing_ids:
            print(f"  WARNING: generator failed to produce reviews for {len(missing_ids)} "
                  f"prompt(s): {missing_ids}")
        
        return results
    
    def receive_feedback(self, feedback):
        """
        Add per-item feedback to generator's context for next round.
        
        Args:
            feedback: String containing per-item feedback from judge
        """
        self.feedback_history.append(feedback)
        # Keep only recent feedback to prevent context bloat (same as judge)
        if len(self.feedback_history) > 8:  # More items since it's per-review feedback
            self.feedback_history = self.feedback_history[-8:]
        print(f"Generator received feedback (now {len(self.feedback_history)} items in history)")


if __name__ == "__main__":
    from data_prep import prepare_data
    from generator_prompts import prepare_generator_data
    
    print("Testing Generator initialization...")
    
    # Load data
    judge_few_shot, judge_calibration, judge_golden, rubric = prepare_data()
    gen_few_shot, gen_tracking, gen_held_out = prepare_generator_data(
        judge_few_shot, judge_calibration, judge_golden
    )
    
    # Initialize generator
    from data_prep import extract_rubric_name
    generator = Generator(
        rubric_text=rubric,
        style_name=extract_rubric_name(rubric),
        few_shot_examples=gen_few_shot
    )
    
    print("\nTesting batched generation (all 8 round-tracking prompts, matching prior failure size)...")
    test_prompts = gen_tracking[:8]
    results = generator.generate_batch(test_prompts)
    
    for result in results:
        print(f"\nPrompt ID: {result['prompt_id']}")
        print(f"Topic: {result['prompt'][:60]}...")
        print(f"Generated review:\n{result['generated_review'][:200]}...")
    
    print("\nTesting feedback mechanism...")
    generator.receive_feedback(
        "Prompt ex_123: Review was too formal. Rubric wants chaotic, fragmented energy."
    )
    
    print("\nGenerator ready!")