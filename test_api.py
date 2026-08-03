"""Quick test to find available Anthropic models."""

import anthropic

client = anthropic.Anthropic()

# Try a few common model names
test_models = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
]

print("Testing Anthropic API connection and available models...\n")

for model in test_models:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print(f"✓ {model} - WORKS!")
        break  # Found a working model
    except anthropic.NotFoundError:
        print(f"✗ {model} - Not found")
    except Exception as e:
        print(f"? {model} - Error: {e}")

print("\nTo manually test a specific model, run:")
print('python -c "import anthropic; c=anthropic.Anthropic(); print(c.messages.create(model=\'MODEL_NAME\', max_tokens=10, messages=[{\'role\':\'user\',\'content\':\'Hi\'}]))"')
