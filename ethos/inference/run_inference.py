from asyncio.log import logger
import json
import multiprocessing as mp
import re
import time
import traceback
import psutil
import numpy as np
import os
from fastapi.responses import JSONResponse

import torch as th
from tqdm import tqdm

from .constants import Test


def get_process_info():
    proc_name = mp.current_process().name
    print(f"Process name: {proc_name}")
    
    # Extract the last number (handles cases like 'LokyProcess-8:7')
    match = re.search(r"(\d+)$", proc_name)  
    proc_num = int(match.group(1)) if match else 1  

    return proc_name, proc_num


def run_inference(loader, args, num_gpus: int = 8):
    model, device, vocab, stoi, results_dir, test_name, suffix, no_compile = args

    proc_name, proc_num = get_process_info()
    if device == "cuda":
        device = f"cuda:{proc_num % num_gpus}"
        th.cuda.set_device(device)
    model.to(device)
    if not no_compile:
        model = th.compile(model)

    dataset = loader.dataset.dataset
    context_len = dataset.context_len
    timeline_len = dataset.timeline_len
    max_timeline_size = context_len + timeline_len
    time_limit = 30 / 365.25 if test_name == Test.READMISSION else 2
    toi = th.tensor(vocab.encode(stoi), device=device, dtype=th.long)

    results = []
    for timeline, ground_truth in tqdm(
        loader, proc_name, total=len(loader), position=proc_num, smoothing=0
    ):
        timeline = timeline.to(device)
        gen_token_num = 0
        offset = 0
        while True:
            if test_name == Test.SOFA_PREDICTION and gen_token_num == 1:
                # append a sofa token to the timeline and continue generating
                last_token = th.tensor(
                    vocab.encode(["SOFA"]), device=timeline.device, dtype=th.long
                )
            else:
                last_token, probs = model.get_next_token(timeline[None, ...], return_probs=True)

            if not offset and len(timeline) == max_timeline_size:
                offset = 1

            timeline = th.cat(
                (timeline[:context_len], timeline[context_len + offset :], last_token.view(-1)),
            )
            gen_token_num += 1
            # stop criterion
            if timeline[-1] in toi:
                stop_reason = "token_of_interest"
                break
            elif test_name == Test.READMISSION or gen_token_num > timeline_len:
                timeline_time = vocab.get_timeline_total_time(
                    timeline[-gen_token_num:].cpu(), decode=True
                )
                if timeline_time > time_limit:
                    stop_reason = "time_limit"
                    break
            elif test_name == Test.SOFA_PREDICTION and gen_token_num == 3:
                # if there are 3 tokens generated and none of them is a toi,
                # then sofa experiment failed as the 3rd token should always be a quantile
                stop_reason = "sofa_fail"
                break
            elif test_name == Test.DRG_PREDICTION:
                stop_reason = "drg_fail"
                break

        expected = ground_truth.pop("expected")
        if test_name in (
            Test.ADMISSION_MORTALITY,
            Test.ICU_MORTALITY,
            Test.SINGLE_ADMISSION,
            Test.SOFA_PREDICTION,
            Test.DRG_PREDICTION,
        ):
            expected = vocab.decode(expected)
        actual = vocab.decode(last_token.item())
        actual_prob = probs[0][last_token.item()].item()
        toi_probs = dict(zip(stoi, probs[0][toi].tolist()))
        timeline_time = vocab.get_timeline_total_time(timeline[-gen_token_num:].cpu(), decode=True)

        results.append(
            {
                "expected": expected,
                "actual": actual,
                "stop_reason": stop_reason,
                "actual_prob": actual_prob,
                **toi_probs,
                **ground_truth,
                "token_dist": gen_token_num,
                "token_time": timeline_time,
            }
        )

    res_file = results_dir / f"part_{proc_num}{f'_{suffix}' if suffix is not None else ''}"
    
    with res_file.with_suffix(".json").open("w") as f:
        json.dump(results, f, indent=4)

    th.cuda.empty_cache()
    return JSONResponse(content=results)

def profile_inference(loader, args, num_gpus: int = 8, save_timeline: bool = True):
    model, device, vocab, stoi, results_dir, test_name, suffix, no_compile = args

    print("\n🚀 **Starting Profiling...**")
    proc_name, proc_num = get_process_info()
    # if device == "cuda":
    #     device = f"cuda:{proc_num % num_gpus}"
    #     th.cuda.set_device(device)
    model.to(device)
    
    if not no_compile:
        model = th.compile(model)

    dataset = loader.dataset.dataset
    context_len = dataset.context_len
    timeline_len = dataset.timeline_len
    max_timeline_size = context_len + timeline_len
    time_limit = 30 / 365.25 if test_name == Test.READMISSION else 2
    toi = th.tensor(vocab.encode(stoi), device=device, dtype=th.long)

    num_params = sum(p.numel() for p in model.parameters())
    param_size_mb = (num_params * 4) / (1024 * 1024)  # Assuming FP32 (4 bytes per param)
    flops = num_params * 2  # Approximate FLOPs per forward pass

    total_tokens = 0
    total_time = 0
    total_model_time = 0
    total_calls = 0

    timeline_output = []

    overall_start_time = time.time()  # Start timing full inference

    total_input_tokens = sum(len(timeline) for timeline, _ in loader)  # Count input tokens
    print(f"\n📊 **Total Input Tokens in Dataset:** {total_input_tokens:,}\n")
    try:
        for timeline, _ in tqdm(loader, desc="Profiling", total=len(loader), position=proc_num):
            timeline = timeline.to(device)
            
            # Enforce 1000-token input limit
            if len(timeline) > 1000:
                timeline = timeline[:1000]  # Truncate
            else:
                padding = th.zeros(1000 - len(timeline), dtype=timeline.dtype, device=device)
                timeline = th.cat((timeline, padding))  # Pad

            gen_token_num = 0
            offset = 0
            
            while True:
                start_time = time.time()
                model_start_time = time.time()
                
                last_token, probs = model.get_next_token(timeline[None, ...], return_probs=True)

                model_time = time.time() - model_start_time
                elapsed_time = time.time() - start_time
                total_model_time += model_time
                total_time += elapsed_time
                total_calls += 1
                total_tokens += 1
                avg_inference_time = total_time / total_tokens
                requests_per_sec = total_calls / total_time if total_time > 0 else 0
                current_throughput = 1 / elapsed_time if elapsed_time > 0 else 0
                cpu_utilization = psutil.cpu_percent(interval=0.1)

                timeline_output.append(vocab.decode(last_token.item()))


                print(f"\n🟢 Token {total_tokens}: '{vocab.decode(last_token.item())}'")
                print(f"   - Time per token: {elapsed_time:.4f} sec")
                print(f"   - Model Execution Time: {model_time:.4f} sec")
                print(f"   - Avg Time per Inference: {avg_inference_time:.4f} sec")
                print(f"   - Requests per Second: {requests_per_sec:.2f} req/sec")
                print(f"   - Current Throughput: {current_throughput:.2f} tokens/sec")
                print(f"   - CPU Utilization: {cpu_utilization}%")


                if not offset and len(timeline) == max_timeline_size:
                    offset = 1

                timeline = th.cat((timeline[:context_len], timeline[context_len + offset :], last_token.view(-1)),)
                gen_token_num += 1


                # Stop criterion
                if timeline[-1] in toi:
                    break
                elif test_name == Test.READMISSION or gen_token_num > timeline_len:
                    timeline_time = vocab.get_timeline_total_time(
                        timeline[-gen_token_num:].cpu(), decode=True
                    )
                    if timeline_time > time_limit:
                        break
    except Exception as e:
        logger.error(f"Inference error: {traceback.format_exc()}")

    token_array_size = len(timeline)
    print(f"Output Tokens Array Size: {token_array_size}")
    overall_end_time = time.time()
    full_inference_time = overall_end_time - overall_start_time
    effective_throughput = total_tokens / full_inference_time if full_inference_time > 0 else 0

    print("\n✅ **Final Profiling Results:**")
    print(f"- 🔁 Total Tokens Generated: {total_tokens}")
    print(f"- ⏱️ Total Inference Time: {full_inference_time:.2f} sec")
    print(f"- 🚀 Effective Throughput: {effective_throughput:.2f} tokens/sec")
    print(f"- 🕒 Average Latency (get_next_token): {total_model_time / total_tokens:.4f} sec/token")
    print(f"- 🏗️ Parameter Size: {param_size_mb:.2f} MB")
    print(f"- 📊 **Total Input Tokens Processed:** {total_input_tokens:,}")
    print(f"- 🚀 **Tokens Per Inference (Limited to 1000):** {1000}")
    
    
    # Print the first 20 tokens of the generated timeline
    print("\n📝 **Generated Timeline (First 20 Tokens):**")
    print(" ".join(timeline_output[:20]))
      # Final Results Dictionary
    results = {
        "total_tokens": total_tokens,
        "total_inference_time": round(full_inference_time, 2),
        "effective_throughput": round(effective_throughput, 2),
        "average_latency": round(total_model_time / total_tokens, 4) if total_tokens > 0 else 0,
        "parameter_size_mb": round(param_size_mb, 2),
        "total_input_tokens": total_input_tokens,
        "tokens_per_inference": 1000,
        "generated_timeline": timeline_output[:20],  # First 20 tokens only
    }

    # Save the full timeline if requested
    if save_timeline:
        with open("generated_timeline.json", "w") as f:
            json.dump(timeline_output, f, indent=4)
        print("\n📄 Full timeline saved to `generated_timeline.json`")

    th.cuda.empty_cache()
    print("\n✅ **Profiling Completed!**\n")
    return JSONResponse(content=results)

def model_weights(loader, args, num_gpus: int = 8, save_timeline: bool = True):
    """
    Extracts and prints detailed attention weights during inference
    
    Args:
        loader: Data loader
        args: Tuple containing (model, device, vocab, stoi, results_dir, test_name, suffix, no_compile)
        num_gpus: Number of available GPUs
        save_timeline: Whether to save results to disk
        
    Returns:
        Dictionary with attention weights and metadata
    """
    model, device, vocab, stoi, results_dir, test_name, suffix, no_compile = args
    
    # Initialize storage
    all_attention_data = []
    current_step = 0
    
    # Hook setup
    def attention_hook(module, input, output, layer_name: str):
        if isinstance(output, tuple) and len(output) >= 2:
            weights = output[1].detach().cpu().numpy()  # [batch, heads, q_len, k_len]
            
            # Print detailed attention information
            print(f"\n🔍 Attention Layer: {layer_name}")
            print(f"Shape: {weights.shape} (batch, heads, query_len, key_len)")
            
            for head_idx in range(weights.shape[1]):
                print(f"\nHead {head_idx + 1}/{weights.shape[1]}:")
                for q_pos in range(weights.shape[2]):
                    for k_pos in range(weights.shape[3]):
                        print(f"  Query[{q_pos}] -> Key[{k_pos}]: {weights[0, head_idx, q_pos, k_pos]:.4f}")
            
            return weights
    
    # Register hooks with layer names
    hooks = []
    for name, module in model.named_modules():
        if 'attention' in name.lower() or isinstance(module, th.nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(
                lambda m, i, o, name=name: attention_hook(m, i, o, name))
            )
    
    try:
        for timeline, _ in tqdm(loader, desc="Processing Attention"):
            timeline = timeline.to(device)
            
            # Pad/truncate to 1000 tokens (matches profile_inference)
            if len(timeline) > 1000:
                timeline = timeline[:1000]
                print(f"⚠️ Timeline truncated to 1000 tokens")
            else:
                padding = th.zeros(1000 - len(timeline), dtype=timeline.dtype, device=device)
                timeline = th.cat((timeline, padding))
            
            input_tokens = [vocab.decode(tid.item()) for tid in timeline if tid != 0]
            print(f"\n\n=== New Timeline ===")
            print(f"Input Tokens: {input_tokens}")
            
            while True:
                print(f"\n🚀 Generation Step {current_step}")
                print("-" * 50)
                
                with th.no_grad():
                    output = model.get_next_token(timeline.unsqueeze(0))
                    if isinstance(output, tuple):
                        output_token_id = output[0].item()
                    else:
                        output_token_id = output.item()
                    
                    output_token = vocab.decode(output_token_id)
                    print(f"Generated Token: '{output_token}' (ID: {output_token_id})")
                
                # Store complete data
                all_attention_data.append({
                    "step": current_step,
                    "input_tokens": input_tokens,
                    "output_token": output_token,
                    "output_token_id": output_token_id
                })
                
                # Update for next step
                timeline = th.cat([timeline[1:], th.tensor([output_token_id], device=device)])
                input_tokens = input_tokens[1:] + [output_token]
                current_step += 1
                
                # Break conditions
                if output_token_id in vocab.encode(stoi):
                    print(f"⏹️ Stop token generated: '{output_token}'")
                    break
                if current_step >= 1000:
                    print("⏹️ Reached maximum generation steps (1000)")
                    break
    
    finally:
        # Cleanup
        for hook in hooks:
            hook.remove()
        th.cuda.empty_cache()
        
        # Save results if requested
        if save_timeline:
            output_path = f"{results_dir}/attention_weights_{test_name}{suffix}.json"
            with open(output_path, 'w') as f:
                json.dump({
                    "metadata": {
                        "test_name": test_name,
                        "steps_generated": current_step,
                        "device": device,
                        "vocab_size": len(vocab)
                    },
                    "attention_data": all_attention_data
                }, f, indent=2)
            print(f"\n💾 Saved detailed attention data to {output_path}")
    
    return {
        "attention_data": all_attention_data,
        "metadata": {
            "test_name": test_name,
            "steps_generated": current_step,
            "device": device
        }
    }