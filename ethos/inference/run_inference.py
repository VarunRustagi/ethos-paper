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
from ethos.model import Ethos, ModelConfig
import torch as th
from tqdm import tqdm

from .constants import Test

# Initialization
config = ModelConfig()
model = Ethos(config, return_attention=True)
model.eval()

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
                last_token = th.tensor(
                    vocab.encode(["SOFA"]), device=timeline.device, dtype=th.long
                )
            else:
                # Assuming model.get_next_token() returns attention weights as well
                last_token, probs, attn_weights = model.get_next_token(
                    timeline[None, ...], return_probs=True, return_attention=True
                )

            if not offset and len(timeline) == max_timeline_size:
                offset = 1

            timeline = th.cat(
                (timeline[:context_len], timeline[context_len + offset :], last_token.view(-1)),
            )
            gen_token_num += 1

            # Stop criterion
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
                stop_reason = "sofa_fail"
                break
            elif test_name == Test.DRG_PREDICTION:
                stop_reason = "drg_fail"
                break

        # Gather the results
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

        # Log attention weights along with other results
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
                "attention_weights": attn_weights.cpu().detach().numpy()  # Log attention weights
            }
        )

    # Save the results with attention weights
    res_file = results_dir / f"part_{proc_num}{f'_{suffix}' if suffix is not None else ''}"
    with res_file.with_suffix(".json").open("w") as f:
        json.dump(results, f, indent=4)

    th.cuda.empty_cache()

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
    model, device, vocab, stoi, results_dir, test_name, suffix, no_compile = args
    
    all_attention_data = []
    current_step = 0

    # Enable built-in attention tracking
    if hasattr(model, 'return_attention'):
        original_setting = model.return_attention
        model.return_attention = True
        # Initialize attention_weights if it doesn't exist
        if not hasattr(model, 'attention_weights'):
            model.attention_weights = []
        print("✅ Enabled built-in attention tracking")
    else:
        raise RuntimeError("Model does not support return_attention")

    try:
        for timeline, _ in tqdm(loader, desc="Extracting Attention"):
            timeline = timeline.to(device)

            # Safely clear attention weights
            if hasattr(model, 'attention_weights') and model.attention_weights is not None:
                model.attention_weights.clear()

            # Truncate or pad timeline
            if len(timeline) > 1000:
                timeline = timeline[:1000]
                print("⚠️ Timeline truncated to 1000 tokens")
            else:
                pad = th.zeros(1000 - len(timeline), dtype=timeline.dtype, device=device)
                timeline = th.cat((timeline, pad))

            while True:
                print(f"\n🚀 Step {current_step}")
                print("-" * 50)

                with th.no_grad():
                    output = model.get_next_token(timeline.unsqueeze(0))
                    token_id = output[0].item() if isinstance(output, tuple) else output.item()

                # Safely extract attention
                if hasattr(model, 'attention_weights') and model.attention_weights:
                    try:
                        attn_weights = model.attention_weights[-1][0]  # [num_heads, q_len, k_len]
                        last_q_pos = attn_weights.shape[1] - 1
                        last_token_attn = attn_weights[:, last_q_pos, :].mean(dim=0)  # [k_len]
                        
                        # Convert to serializable format
                        attn_data = {
                            "step": current_step,
                            "token_id": int(token_id),  # Ensure JSON-serializable
                            "attention": last_token_attn.cpu().tolist()  # Convert tensor to list
                        }
                        all_attention_data.append(attn_data)
                    except Exception as e:
                        print(f"⚠️ Error processing attention: {str(e)}")

                # Prepare next step
                timeline = th.cat([timeline[1:], th.tensor([token_id], device=device)])
                current_step += 1

                # Stop conditions
                if token_id in vocab.encode(stoi) or current_step >= 1000:
                    print("⏹️ Generation complete")
                    break

    finally:
        # Restore original settings
        if hasattr(model, 'return_attention'):
            model.return_attention = original_setting

        # Save results (ensure all data is JSON-serializable)
        if save_timeline:
            out_path = f"{results_dir}/attention_scores_{test_name}{suffix}.json"
            try:
                with open(out_path, 'w') as f:
                    json.dump({
                        "metadata": {
                            "test_name": str(test_name),  # Ensure string
                            "device": str(device),
                            "steps_generated": int(current_step),
                            "vocab_size": int(len(vocab))
                        },
                        "attention_data": all_attention_data
                    }, f, indent=2)
                print(f"💾 Saved to {out_path}")
            except TypeError as e:
                print(f"❌ Failed to save: {str(e)}")
                # Debug: Print problematic items
                for i, item in enumerate(all_attention_data):
                    try:
                        json.dumps(item)
                    except TypeError:
                        print(f"Non-serializable item at index {i}: {item}")

    return {
        "attention_data": all_attention_data,
        "metadata": {
            "test_name": str(test_name),
            "steps_generated": int(current_step),
            "device": str(device)
        }
    }
