# https://github.com/PGCRT/CRT-Nodes/blob/main/py/LoraLoaderZImage.py

import torch


@torch.inference_mode()
def convert_z_image_lora(lora: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_lora = {}

    # Temporary storage to group QKV keys
    # Structure: { "diffusion_model.layers.0.attention": { "q": {A, B}, "k": {A, B}, "v": {A, B} } }
    qkv_groups = {}

    for k, v in lora.items():
        new_k = k

        # IMPORTANT: Do NOT strip 'diffusion_model.' prefix.
        # ComfyUI's internal state dict expects it.

        # 1. Output Projection Fix (to_out.0 -> out)
        # Matches Base Key: diffusion_model.layers.X.attention.out.weight
        if ".attention.to_out.0." in new_k:
            new_k = new_k.replace(".attention.to_out.0.", ".attention.out.")
            new_lora[new_k] = v
            continue

        # 2. Handle QKV Separation
        # LoRA Key: diffusion_model.layers.X.attention.to_q.lora_A.weight
        if ".attention.to_" in new_k:
            # Extract layer base: diffusion_model.layers.0.attention
            # Extract type: q, k, or v
            # Extract param: lora_A.weight or lora_B.weight
            parts = new_k.split(".attention.to_")
            base_prefix = parts[0] + ".attention"
            remainder = parts[1]  # e.g. q.lora_A.weight

            qkv_type = remainder[0]  # 'q', 'k', or 'v'
            suffix = remainder[2:]  # 'lora_A.weight'

            if base_prefix not in qkv_groups:
                qkv_groups[base_prefix] = {"q": {}, "k": {}, "v": {}}

            qkv_groups[base_prefix][qkv_type][suffix] = v
            continue

        # 3. Pass through everything else
        # (mlp, adaLN_modulation should match if we keep prefix)
        new_lora[new_k] = v

    # --- FUSE QKV ---
    for base_key, group in qkv_groups.items():
        # Base Key Target: diffusion_model.layers.X.attention.qkv.weight

        # Check A weights (Down)
        ak_a = "lora_A.weight"
        if ak_a in group["q"] and ak_a in group["k"] and ak_a in group["v"]:
            q_a = group["q"][ak_a]
            k_a = group["k"][ak_a]
            v_a = group["v"][ak_a]

            # Stack A vertically: (3*rank, dim_in)
            fused_A = torch.cat([q_a, k_a, v_a], dim=0)
            new_lora[fused_A_key := f"{base_key}.qkv.lora_A.weight"] = fused_A

        # Check B weights (Up)
        ak_b = "lora_B.weight"
        if ak_b in group["q"] and ak_b in group["k"] and ak_b in group["v"]:
            q_b = group["q"][ak_b]
            k_b = group["k"][ak_b]
            v_b = group["v"][ak_b]

            # Block Diagonal B: (3*dim_out, 3*rank)
            out_dim, rank = q_b.shape

            fused_B = torch.zeros((out_dim * 3, rank * 3), dtype=q_b.dtype, device=q_b.device)
            fused_B[0:out_dim, 0:rank] = q_b
            fused_B[out_dim : 2 * out_dim, rank : 2 * rank] = k_b
            fused_B[2 * out_dim : 3 * out_dim, 2 * rank : 3 * rank] = v_b

            new_lora[fused_B_key := f"{base_key}.qkv.lora_B.weight"] = fused_B

        # Handle Alphas if necessary (Generic copy)
        ak_alpha = "lora_alpha"
        if ak_alpha in group["q"]:
            # Use Q's alpha for the fused block
            new_lora[f"{base_key}.qkv.lora_alpha"] = group["q"][ak_alpha]

    return new_lora
