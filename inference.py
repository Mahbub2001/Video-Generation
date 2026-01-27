import torch
from PIL import Image
from utils.data import save_video
from wan_video_new import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# Text-to-video
video = pipe(
    prompt="一名宇航员身穿太空服，面朝镜头骑着一匹机械马在火星表面驰骋。红色的荒凉地表延伸至远方，点缀着巨大的陨石坑和奇特的岩石结构。机械马的步伐稳健，扬起微弱的尘埃，展现出未来科技与原始探索的完美结合。宇航员手持操控装置，目光坚定，仿佛正在开辟人类的新疆域。背景是深邃的宇宙和蔚蓝的地球，画面既科幻又充满希望，让人不禁畅想未来的星际生活。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
)
save_video(video, "video_Wan2.1-T2V-14B.mp4", fps=15, quality=5)


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
