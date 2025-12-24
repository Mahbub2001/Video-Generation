import torch
from PIL import Image
from utils.data import save_video, VideoData
from wan_video_new import WanVideoPipeline, ModelConfig
# from modelscope import dataset_snapshot_download


# pipe = WanVideoPipeline.from_pretrained(
#     torch_dtype=torch.bfloat16,
#     device="cuda",
#     model_configs=[
#         ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
#         ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
#         ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="Wan2.1_VAE.pth"),
#         ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
#     ],
#     tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
# )
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # Diffusion model
        ModelConfig(
            model_id="/kaggle/input/pai/other/default/1/Wan2.1-Fun-V1.1-1.3B-InP",
            origin_file_pattern="diffusion_pytorch_model.safetensors"
        ),

        # T5 / UMT5 text encoder
        ModelConfig(
            model_id="/kaggle/input/wan-ai-new/other/default/1/Wan2.1-T2V-1.3B",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"
        ),

        # VAE
        ModelConfig(
            model_id="/kaggle/input/wan-ai-new/other/default/1/Wan2.1-T2V-1.3B",
            origin_file_pattern="Wan2.1_VAE.pth"
        ),

        # CLIP image encoder
        ModelConfig(
            model_id="/kaggle/input/wan-ai-new/other/default/1/Wan2.1-I2V-14B-480P",
            origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        ),
    ],

    tokenizer_config=ModelConfig(
        model_id="/kaggle/input/wan-ai-new/other/default/1/Wan2.1-T2V-1.3B",
        origin_file_pattern="google/umt5-xxl/"
    ),
)
# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=f"data/examples/wan/input_image.jpg"
# )
# image = Image.open("data/examples/wan/input_image.jpg")
image = Image.open("test_image.png").convert("RGB")


# First and last frame to video
video = pipe(
    prompt=(
    "A small boat bravely moves forward through strong winds and crashing waves. "
    "The deep blue ocean is turbulent, with white foam striking the sides of the boat, "
    "yet the boat shows no fear and continues steadily toward the horizon. "
    "Sunlight reflects on the surface of the water, creating a warm golden shimmer, "
    "giving the scene a sense of strength, courage, and determination."
),    
    negative_prompt=(
    "overly saturated colors, overexposed, static scene, blurry details, subtitles, "
    "low quality, jpeg artifacts, ugly, deformed, distorted shapes, extra limbs, "
    "poorly drawn hands or face, motionless frame, cluttered background"
),
    input_image=image,
    seed=0, tiled=True
    # You can input `end_image=xxx` to control the last frame of the video.
    # The model will automatically generate the dynamic content between `input_image` and `end_image`.
)
save_video(video, "video_Wan2.1-Fun-V1.1-1.3B-InP.mp4", fps=15, quality=5)