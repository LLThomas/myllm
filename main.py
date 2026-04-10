import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen2-0.5b")
parser.add_argument("--draft-model", type=str, default=None)
parser.add_argument(
    "--prompt",
    type=str,
    default="Give me a short introduction to large language model.",
)
parser.add_argument("--solution", type=str, default="myllm")
parser.add_argument("--loader", type=str, default="week1")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--sampler-temp", type=float, default=0)
parser.add_argument("--sampler-top-p", type=float, default=None)
parser.add_argument("--sampler-top-k", type=int, default=None)
parser.add_argument("--enable-thinking", action="store_true")
parser.add_argument("--enable-flash-attn", action="store_true")

args = parser.parse_args()

use_hf = False
if args.solution == "myllm":
    print("Using your myllm solution")
    from my_llm import (
        models,
        simple_generate,
        simple_generate_with_kv_cache,
        # speculative_generate,
        sampler,
    )
elif args.solution == "hf":
    use_hf = True
    print("Using the original HuggingFace model")
else:
    raise ValueError(f"Solution {args.solution} not supported")

device = torch.device(
    "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
)

args.model = models.shortcut_name_to_full_name(args.model)
hf_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(
    device
)
tokenizer = AutoTokenizer.from_pretrained(args.model)

# if args.draft_model:
#     args.draft_model = models.shortcut_name_to_full_name(args.draft_model)
#     draft_hf_model = AutoModelForCausalLM.from_pretrained(
#         args.draft_model,
#         dtype=torch.float16
#     ).to(device)
#     draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
#     if args.loader == "week1":
#         raise ValueError("Draft model not supported for week1")
# else:
#     draft_hf_model = None
#     draft_tokenizer = None

if use_hf:
    myllm_model = hf_model
else:
    if args.loader == "week1":
        print(f"Using week1 loader for {args.model}")
        myllm_model = models.dispatch_model(args.model, hf_model, week=1).to(device)
    elif args.loader == "week2":
        print(
            f"Using week2 loader with flash_attn={args.enable_flash_attn} "
            f"thinking={args.enable_thinking} for {args.model}"
        )
        myllm_model = models.dispatch_model(
            args.model, hf_model, week=2, enable_flash_attn=args.enable_flash_attn
        ).to(device)
        # if draft_hf_model is not None:
        #     print(f"Using draft model {args.draft_model}")
        #     draft_myllm_model = models.dispatch_model(
        #         args.draft_model,
        #         draft_hf_model,
        #         week=2,
        #         enable_flash_attn=args.enable_flash_attn,
        #     )
        # else:
        #     draft_myllm_model = None
    else:
        raise ValueError(f"Loader {args.loader} not supported")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": args.prompt},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

if not use_hf:
    sampler_obj = sampler.make_sampler(
        args.sampler_temp, top_p=args.sampler_top_p, top_k=args.sampler_top_k
    )
    if args.loader == "week1":
        simple_generate(myllm_model, tokenizer, prompt, sampler=sampler_obj)
    elif args.loader == "week2":
        # if draft_myllm_model is not None:
        #     speculative_generate(
        #         draft_myllm_model,
        #         myllm_model,
        #         draft_tokenizer,
        #         tokenizer,
        #         prompt,
        #     )
        # else:
        simple_generate_with_kv_cache(myllm_model, tokenizer, prompt, sampler=sampler_obj)
else:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = hf_model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=args.sampler_temp if args.sampler_temp > 0 else 1.0,
            top_p=args.sampler_top_p,
            top_k=args.sampler_top_k,
            do_sample=args.sampler_temp > 0,
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)