import torch
from PIL import Image
from utils.data import save_video
from wan_video_new import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.float16,   # safer for Kaggle
    device="cuda",
    model_configs=[

        # Diffusion model
        ModelConfig(
            model_id="models/PAI/Wan2.1-Fun-V1.1-1.3B-InP",
            origin_file_pattern="diffusion_pytorch_model.safetensors"
        ),

        # T5 text encoder
        ModelConfig(
            path="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
        ),

        # VAE
        ModelConfig(
            path="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
        ),

        # CLIP image encoder
        ModelConfig(
            path="models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        ),
    ],

    tokenizer_config=ModelConfig(
        path="models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
    ),
)

image = Image.open("test_image.png").convert("RGB")

video = pipe(
    prompt=(
        "A small boat bravely moves forward through strong winds and crashing waves. "
        "The deep blue ocean is turbulent, with white foam striking the sides of the boat, "
        "yet the boat shows no fear and continues steadily toward the horizon. "
        "Sunlight reflects on the surface of the water, creating a warm golden shimmer."
    ),
    negative_prompt=(
        "overly saturated, overexposed, static, blurry, subtitles, low quality, "
        "jpeg artifacts, deformed, extra limbs, cluttered background"
    ),
    input_image=image,
    seed=0,
    tiled=True,
    num_frames=150
)

save_video(video, "video_Wan2.1-10s.mp4", fps=15, quality=5)
# import torch
# from PIL import Image
# from utils.data import save_video
# from wan_video_new import WanVideoPipeline, ModelConfig

# pipe = WanVideoPipeline.from_pretrained(
#     torch_dtype=torch.float16,   # safer for Kaggle
#     device="cuda",
#     model_configs=[

#         # Diffusion model
#         ModelConfig(
#             model_id="/workspace/Video-Generation/models/PAI/Wan2.1-Fun-V1.1-1.3B-InP",
#             origin_file_pattern="diffusion_pytorch_model.safetensors"
#         ),

#         # T5 text encoder
#         ModelConfig(
#             path="/workspace/Video-Generation/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
#         ),

#         # VAE
#         ModelConfig(
#             path="/workspace/Video-Generation/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
#         ),

#         # CLIP image encoder
#         ModelConfig(
#             path="/workspace/Video-Generation/models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
#         ),
#     ],

#     tokenizer_config=ModelConfig(
#         path="/workspace/Video-Generation/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
#     ),
# )

# image = Image.open("test_image.png").convert("RGB")

# video = pipe(
#     prompt=(
#         "A small boat bravely moves forward through strong winds and crashing waves. "
#         "The deep blue ocean is turbulent, with white foam striking the sides of the boat, "
#         "yet the boat shows no fear and continues steadily toward the horizon. "
#         "Sunlight reflects on the surface of the water, creating a warm golden shimmer."
#     ),
#     negative_prompt=(
#         "overly saturated, overexposed, static, blurry, subtitles, low quality, "
#         "jpeg artifacts, deformed, extra limbs, cluttered background"
#     ),
#     input_image=image,
#     seed=0,
#     tiled=True,
#     num_frames=150
# )

# save_video(video, "video_Wan2.1-10s.mp4", fps=15, quality=5)
