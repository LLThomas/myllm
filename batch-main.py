import argparse
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from my_llm import models
from my_llm.batch import batch_generate


# Prompts for testing
shanghai_wikipedia = """
Shanghai[a] is a direct-administered municipality and the most populous urban area in China. The city is located on the Chinese shoreline on the southern estuary of the Yangtze River, with the Huangpu River flowing through it. The population of the city proper is the second largest in the world after Chongqing, with around 24.87 million inhabitants in 2023, while the urban area is the most populous in China, with 29.87 million residents. As of 2022, the Greater Shanghai metropolitan area was estimated to produce a gross metropolitan product (nominal) of nearly 13 trillion RMB ($1.9 trillion).[13] Shanghai is one of the world's major centers for finance, business and economics, research, science and technology, manufacturing, transportation, tourism, and culture. The Port of Shanghai is the world's busiest container port.
""".strip()

shanghai_wikipedia += "Based on the previous information, "

prompts = [
    shanghai_wikipedia + "Where is Shanghai?",
    shanghai_wikipedia + "How much is the population of Shanghai?",
    shanghai_wikipedia + "What is the GDP of Shanghai?",
    shanghai_wikipedia + "What is the population of Shanghai?",
    shanghai_wikipedia + "What is the second largest city proper in China?",
    shanghai_wikipedia + "What is Shanghai known for?",
    shanghai_wikipedia + "What are the rivers in Shanghai?",
    shanghai_wikipedia + "Shanghai is the major center for what?",
    "What is the capital of France?",
    "Where is New York City?",
    "Where is Tokyo?",
    "What is the capital of China?",
    "Where is Pittsburgh?",
    "Where is Vancouver?",
    "Where is Toronto?",
    "Give me a short introduction to large language model.",
]


def main():
    parser = argparse.ArgumentParser(description="Batch generation with continuous batching")
    parser.add_argument("--model", type=str, default="qwen2-0.5b",
                        help="Model name or path (default: qwen2-0.5b)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to run on (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Maximum concurrent requests (default: 5)")
    parser.add_argument("--prefill-step", type=int, default=128,
                        help="Tokens to prefill per chunk (default: 128)")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    parser.add_argument("--enable-flash-attn", action="store_true",
                        help="Enable flash attention (requires compiled extension)")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Don't shuffle prompts")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable progress output")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # Shuffle prompts for variety
    prompts_copy = prompts.copy()
    if not args.no_shuffle:
        random.shuffle(prompts_copy)

    args.model = models.shortcut_name_to_full_name(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    hf_model = hf_model.to(device)
    model = models.dispatch_model(
        args.model, hf_model, week=2, enable_flash_attn=args.enable_flash_attn
    ).to(device)

    encoded_prompts = []
    for idx, prompt in enumerate(prompts_copy):
        print(f"Prompt {idx}: {prompt[:50]}...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded_prompts.append(chat_prompt)

    print(f"\nStarting batch generation with batch_size={args.batch_size}, prefill_step={args.prefill_step}")
    print("=" * 60)

    verbose = not args.quiet
    result = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=encoded_prompts,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        prefill_steps=args.prefill_step,
        device=device,
        verbose=verbose,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for prompt_idx, text in result:
        print(f"\n--- Prompt {prompt_idx} ---")
        print(f"Q: {prompts_copy[prompt_idx][:100]}...")
        print(f"A: {text}")

    print(f"\nTotal: {len(result)} responses generated")


if __name__ == "__main__":
    main()