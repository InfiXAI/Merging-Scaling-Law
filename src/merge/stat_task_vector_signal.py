import argparse, os, json, re
import torch
from transformers import AutoModelForCausalLM

def should_exclude(name: str, patterns):
    if not patterns:
        return False
    return any(re.search(pat, name) for pat in patterns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--models_to_merge", type=str, required=True, help="Comma-separated list")
    ap.add_argument("--exclude_param_names_regex", type=str, default="", help="Comma-separated regex list to exclude params")
    ap.add_argument("--abs_threshold", type=float, default=0.0, help="Treat |delta|<=threshold as 0")
    ap.add_argument("--use_gpu", action="store_true", default=False)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    models_to_merge = [p for p in args.models_to_merge.split(",") if p]
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    thr = float(args.abs_threshold)
    exclude_list = [s for s in args.exclude_param_names_regex.split(",") if s]

    print("=== Task-vector TOTAL signal (summary-only) ===")
    print(f"Base: {args.base_model}")
    print(f"Experts: {models_to_merge}")
    print(f"Use GPU: {device=='cuda'} | dtype: bfloat16 | abs_threshold: {thr}")
    if exclude_list:
        print(f"Exclude regex: {exclude_list}")

    # 仅需 state_dict 即可，无需前向
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    base_sd = {k: v.to(device) for k, v in base.state_dict().items()}
    del base  # 省显存

    signal_total = 0  # 核心：∑_{m} ∑_{j} sign(delta_{m,j})
    total_elems = 0   # 统计用途（非必须）

    with torch.no_grad():
        for mpath in models_to_merge:
            ft = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.bfloat16)
            ft_sd = {k: v.to(device) for k, v in ft.state_dict().items()}
            del ft

            for name, base_param in base_sd.items():
                if should_exclude(name, exclude_list):
                    continue
                if name not in ft_sd:
                    continue
                ft_param = ft_sd[name]
                if ft_param.shape != base_param.shape:
                    continue

                delta = ft_param - base_param
                if thr > 0.0:
                    # |delta|<=thr 视作 0 票；其余按正(+1)/负(-1)
                    vote_pos = (delta >  thr).to(torch.int32)
                    vote_neg = (delta < -thr).to(torch.int32)
                    vote = vote_pos - vote_neg
                else:
                    vote = torch.sign(delta).to(torch.int32)  # -1, 0, +1

                signal_total += vote.to(torch.int64).sum().item()
                total_elems  += vote.numel()

                # 释放中间张量
                del ft_param, delta, vote

            # 清理 per-model state_dict，降低峰值显存
            del ft_sd
            if device == "cuda":
                torch.cuda.empty_cache()

    # 输出 summary.json（只含核心值 + 少量上下文）
    js_path = os.path.join(args.output_dir, "summary.json")
    summary = {
        "signal_total": int(signal_total),          # 你要的最终标量
        "num_models": len(models_to_merge),
        "total_elements_per_model": total_elems // max(len(models_to_merge), 1),
        "abs_threshold": thr,
        "base_model": args.base_model,
        "models": models_to_merge,
    }
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Done ===")
    print(f"Saved summary to: {js_path}")
    print(f"SIGNAL_TOTAL {signal_total}")  # 便于 bash 直接提取

if __name__ == "__main__":
    main()

