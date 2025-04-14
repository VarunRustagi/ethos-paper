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

def model_weights(loader, args, num_gpus: int = 8):
    """Analyze and profile model weights during inference.

    Args:
        loader: Data loader containing timelines and ground truth
        args: Tuple containing (model, device, vocab, stoi, results_dir, test_name, suffix, no_compile)
        num_gpus: Number of available GPUs
    """
    model, device, vocab, stoi, results_dir, test_name, suffix, no_compile = args

    print("\n🔍 **Starting Model Weights Analysis...**")
    proc_name, proc_num = get_process_info()
    model.to(device)

    if not no_compile:
        model = th.compile(model)

    # Model architecture and parameter stats
    print("\n **Model Architecture:**")
    print(model)
    print(f"\n⚙️  Device: {device}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n **Parameter Count:**")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")
    print(f"- Non-trainable parameters: {total_params - trainable_params:,}")

    # Weight statistics
    weight_stats = {
        'layer_stats': {},
        'total_params': total_params,
        'trainable_params': trainable_params
    }

    print("\n **Layer-wise Weight Analysis:**")
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_stats = {
                'shape': list(param.shape),
                'mean': th.mean(param.data).item(),
                'std': th.std(param.data).item(),
                'min': th.min(param.data).item(),
                'max': th.max(param.data).item(),
                'numel': param.numel(),
                'dtype': str(param.dtype)
            }

            weight_stats['layer_stats'][name] = layer_stats

            print(f"\n🔹 Layer: {name}")
            print(f"   - Shape: {layer_stats['shape']}")
            print(f"   - Parameters: {layer_stats['numel']:,}")
            print(f"   - Mean: {layer_stats['mean']:.6f}")
            print(f"   - Std: {layer_stats['std']:.6f}")
            print(f"   - Range: [{layer_stats['min']:.6f}, {layer_stats['max']:.6f}]")
            print(f"   - Dtype: {layer_stats['dtype']}")

    param_size = total_params * 4 / (1024 ** 2)
    print(f"\n **Memory Usage:**")
    print(f"- Estimated size: {param_size:.2f} MB (float32)")

    print("\n **Running Sample Inference to Observe Weight Behavior...**")
    try:
        sample_timeline, _ = next(iter(loader))
        sample_timeline = sample_timeline.to(device)
        input_token_ids = sample_timeline.tolist()
        input_tokens = vocab.decode(sample_timeline)

        print("\n **Full Input Token Timeline:**")
        print(f"- Token IDs: {input_token_ids}")
        print(f"- Decoded Input Tokens:\n{input_tokens}")

        print("\n **Weight Gradients Before Forward Pass:**")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad exists (norm: {th.norm(param.grad):.4f})")
            else:
                print(f"{name}: no gradient")

        with th.no_grad():
            last_token, probs = model.get_next_token(sample_timeline[None, ...], return_probs=True)

        print("\n🧮 **Weight Gradients After Forward Pass (Inference Mode):**")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad exists (norm: {th.norm(param.grad):.4f})")
            else:
                print(f"{name}: no gradient (inference only)")

        generated_token_id = last_token.item()
        generated_token = vocab.decode(generated_token_id)
        new_token_ids = input_token_ids + [generated_token_id]
        new_decoded = vocab.decode(th.tensor(new_token_ids).to(device))

        print(f"\n🟢 **Generated Token:**")
        print(f"   - ID: {generated_token_id}")
        print(f"   - Token: '{generated_token}'")

        print(f"\n📈 **Updated Token Timeline:**")
        print(f"- New Token IDs: {new_token_ids}")
        print(f"- New Decoded Tokens:\n{new_decoded}")

        # Top-k prediction details
        topk = 5
        top_probs, top_indices = th.topk(probs[0, -1], k=topk)
        print(f"\n🔢 **Top {topk} Predicted Tokens:**")
        for i in range(topk):
            tok_id = top_indices[i].item()
            tok_prob = top_probs[i].item()
            tok_str = vocab.decode(tok_id)
            print(f"  {i+1}. Token: '{tok_str}' (ID: {tok_id}, Prob: {tok_prob:.4f})")

        print(f"\n📐 Output Shape: {list(probs.shape)}")

    except Exception as e:
        logger.error(f"Sample inference error: {traceback.format_exc()}")

    weights_file = results_dir / f"weight_stats_{proc_num}.json"
    with weights_file.open("w") as f:
        json.dump(weight_stats, f, indent=4)
    print(f"\n✅ Saved weight statistics to {weights_file}")

    th.cuda.empty_cache()
    print("\n✅ **Model Weights Analysis Completed!**\n")
    return JSONResponse(content=weight_stats)
