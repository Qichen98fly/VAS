import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import cv2

from src.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from src.conversation import conv_templates, SeparatorStyle
from src.model.builder import load_pretrained_model
from src.utils import disable_torch_init
from src.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from src.stash import MetadataStation
from src.logic.logic import DimProspector, HeadFork

import requests
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def detect_sink_tokens(hidden_states, layer, tau=20, dim_sink=[310, 3917]):
    """
    Detect sink tokens based on RMSNorm values at specific dimensions.
    Returns indices of detected sink tokens.
    """
    # RMSNorm calculation
    hidden_states_float = hidden_states.to(torch.float32)
    variance = hidden_states_float.pow(2).mean(-1, keepdim=True)
    rms_norm_hs = torch.abs(hidden_states_float * torch.rsqrt(variance + 1e-6))

    # Get values at sink dimensions
    rms_values = torch.stack([rms_norm_hs[:, :, idx] for idx in dim_sink], dim=-1)  # [bsz, tok, 2]
    max_rms_values = torch.max(rms_values, dim=-1)[0]  # [bsz, tok]

    # Find indices where max RMS values exceed threshold
    sink_indices = torch.nonzero(max_rms_values > tau)[:, 1]  # Get token axis

    return sink_indices


def visualize_attention_map(attention_weights, image, sink_token_indices, output_path,
                           layer_idx=None, head_idx=None, target_resolution=256):
    """
    Visualize attention map with sink tokens highlighted.

    Args:
        attention_weights: Attention weights [batch, heads, query, key]
        image: Original PIL Image
        sink_token_indices: Indices of sink tokens
        output_path: Path to save visualization
        layer_idx: Which layer to visualize (if None, use last layer)
        head_idx: Which head to visualize (if None, average all heads)
        target_resolution: Target resolution for visualization (128 or 256)
    """
    # Get image token attention
    # Assuming image tokens start after system/role tokens
    im_start = MetadataStation.segments['begin_pos']['image']
    vis_len = MetadataStation.metadata['vis_len']

    # Extract attention to image tokens [batch, heads, query, image_tokens]
    image_attn = attention_weights[:, :, -1, im_start:im_start+vis_len]  # Use last query token

    if head_idx is not None:
        # Use specific head
        attn_map = image_attn[0, head_idx].cpu().detach().numpy()
    else:
        # Average over all heads
        attn_map = image_attn[0].mean(dim=0).cpu().detach().numpy()

    # Reshape to 2D (assuming 576 tokens = 24x24 grid)
    grid_size = int(np.sqrt(vis_len))
    if vis_len != grid_size * grid_size:
        print(f"Warning: vis_len={vis_len} is not a perfect square. Using grid_size={grid_size}")

    attn_map_2d = attn_map.reshape(grid_size, grid_size)

    # Upsample to target resolution using nearest neighbor interpolation
    attn_map_upsampled = cv2.resize(attn_map_2d, (target_resolution, target_resolution),
                                    interpolation=cv2.INTER_NEAREST)

    # Normalize attention map to [0, 1]
    attn_map_normalized = (attn_map_upsampled - attn_map_upsampled.min()) / (attn_map_upsampled.max() - attn_map_upsampled.min() + 1e-8)

    # Create heatmap using colormap
    cmap = cm.get_cmap('jet')
    heatmap = cmap(attn_map_normalized)[:, :, :3]  # Get RGB channels
    heatmap = (heatmap * 255).astype(np.uint8)

    # Resize original image to match target resolution
    image_resized = image.resize((target_resolution, target_resolution), Image.BILINEAR)
    image_np = np.array(image_resized)

    # Create overlay (blend image with heatmap)
    alpha = 0.5
    overlay = (alpha * image_np + (1 - alpha) * heatmap).astype(np.uint8)

    # Create visualization with sink tokens highlighted
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original image
    axes[0, 0].imshow(image_resized)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Attention heatmap
    im1 = axes[0, 1].imshow(attn_map_normalized, cmap='jet')
    axes[0, 1].set_title(f'Attention Map (Layer {layer_idx if layer_idx else "avg"})')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Overlay
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Attention Overlay')
    axes[1, 0].axis('off')

    # Sink tokens visualization
    sink_mask = np.zeros((grid_size, grid_size))
    vis_sink_indices = [idx - im_start for idx in sink_token_indices if im_start <= idx < im_start + vis_len]
    for idx in vis_sink_indices:
        row = idx // grid_size
        col = idx % grid_size
        if 0 <= row < grid_size and 0 <= col < grid_size:
            sink_mask[row, col] = 1

    # Upsample sink mask
    sink_mask_upsampled = cv2.resize(sink_mask, (target_resolution, target_resolution),
                                     interpolation=cv2.INTER_NEAREST)

    # Create overlay with sink tokens highlighted in red
    overlay_with_sinks = overlay.copy()
    red_overlay = np.zeros_like(overlay_with_sinks)
    red_overlay[:, :, 0] = 255  # Red channel
    mask_3d = np.stack([sink_mask_upsampled] * 3, axis=-1)
    overlay_with_sinks = np.where(mask_3d > 0.5,
                                   0.7 * overlay_with_sinks + 0.3 * red_overlay,
                                   overlay_with_sinks).astype(np.uint8)

    axes[1, 1].imshow(overlay_with_sinks)
    axes[1, 1].set_title(f'Attention with Sink Tokens (n={len(vis_sink_indices)})')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to {output_path}")
    print(f"Total sink tokens in image region: {len(vis_sink_indices)}")


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # Prepare query
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # Determine conversation mode
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Load images
    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # Setup metadata for sink token detection
    MetadataStation.set_vis_len(576)  # LLaVA typically uses 576 image tokens (24x24)
    MetadataStation.segments['begin_pos']['image'] = (input_ids == IMAGE_TOKEN_INDEX).nonzero()[0, 1].item()

    # Hook to capture attention weights and hidden states
    attention_weights_dict = {}
    hidden_states_dict = {}

    def attention_hook(module, input, output, layer_idx):
        # output is (attn_output, attn_weights, past_key_value)
        if len(output) > 1 and output[1] is not None:
            attention_weights_dict[layer_idx] = output[1].detach()

    def hidden_states_hook(module, input, output, layer_idx):
        # Capture input hidden states for sink token detection
        hidden_states_dict[layer_idx] = input[0].detach()

    # Register hooks for specific layers
    hooks = []
    target_layers = args.vis_layers if args.vis_layers else [len(model.model.layers) - 1]

    for layer_idx in target_layers:
        if layer_idx < len(model.model.layers):
            hook = model.model.layers[layer_idx].self_attn.register_forward_hook(
                lambda m, i, o, idx=layer_idx: attention_hook(m, i, o, idx)
            )
            hooks.append(hook)

            hook_hs = model.model.layers[layer_idx].register_forward_hook(
                lambda m, i, o, idx=layer_idx: hidden_states_hook(m, i, o, idx)
            )
            hooks.append(hook_hs)

    # Run inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            output_attentions=True,
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Decode output
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print("\n" + "="*50)
    print("Model Output:")
    print("="*50)
    print(outputs)
    print("="*50 + "\n")

    # Visualize attention maps
    if attention_weights_dict:
        os.makedirs(args.output_dir, exist_ok=True)

        for layer_idx, attn_weights in attention_weights_dict.items():
            print(f"\nProcessing Layer {layer_idx}...")

            # Detect sink tokens
            if layer_idx in hidden_states_dict:
                hidden_states = hidden_states_dict[layer_idx]
                sink_indices = detect_sink_tokens(hidden_states, layer_idx,
                                                  tau=args.tau,
                                                  dim_sink=args.dim_sink)
                print(f"Detected {len(sink_indices)} sink tokens in layer {layer_idx}")
            else:
                sink_indices = torch.tensor([])

            # Visualize for each head if requested
            if args.visualize_all_heads:
                num_heads = attn_weights.shape[1]
                for head_idx in range(num_heads):
                    output_path = os.path.join(args.output_dir,
                                              f"attention_layer{layer_idx}_head{head_idx}.png")
                    visualize_attention_map(attn_weights, images[0], sink_indices,
                                          output_path, layer_idx, head_idx,
                                          args.resolution)
            else:
                # Average over all heads
                output_path = os.path.join(args.output_dir,
                                          f"attention_layer{layer_idx}_avg.png")
                visualize_attention_map(attn_weights, images[0], sink_indices,
                                      output_path, layer_idx, None,
                                      args.resolution)
    else:
        print("\nWarning: No attention weights captured. Make sure output_attentions=True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # Visualization parameters
    parser.add_argument("--output-dir", type=str, default="./attention_visualizations",
                       help="Directory to save attention visualizations")
    parser.add_argument("--vis-layers", type=int, nargs='+', default=None,
                       help="Which layers to visualize (default: last layer)")
    parser.add_argument("--visualize-all-heads", action="store_true",
                       help="Visualize each attention head separately")
    parser.add_argument("--resolution", type=int, default=256, choices=[128, 256],
                       help="Target resolution for visualization (128 or 256)")

    # Sink token detection parameters
    parser.add_argument("--tau", type=float, default=20,
                       help="Threshold for sink token detection")
    parser.add_argument("--dim-sink", type=int, nargs='+', default=[310, 3917],
                       help="Dimensions to check for sink tokens")

    args = parser.parse_args()

    eval_model(args)
