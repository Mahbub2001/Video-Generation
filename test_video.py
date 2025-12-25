
HF_TOKEN = "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"

from huggingface_hub import InferenceClient
client = InferenceClient(
    provider="replicate",
    api_key=HF_TOKEN,
)

import os

try:
    from save_video import save_video
    from wan_video_new import WanVideoPipeline
    from loader.config import ModelConfig


    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="diffusion_pytorch_model.safetensors"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    )

    video = pipe(
        prompt="A young man walking confidently down a sunny city street, realistic motion, cinematic camera, natural lighting, highly detailed, 4K",
        seed=42,
        tiled=True,
    )
    save_video(video, "video1.mp4", fps=16)
    print("Local generation completed!")

except Exception as e:
    print(f"Local model not available ({type(e).__name__}) → using main cloud pipeline (recommended)\n")

video_url = client.text_to_video(
    prompt="A young man walking confidently down a sunny city street, realistic motion, cinematic camera, natural lighting, highly detailed, 4K",
    model="Wan-AI/Wan2.1-T2V-1.3B",
)

print("\nVideo generated in seconds!")
print("Download:", video_url)

response = requests.get(video_url, stream=True, timeout=300)
response.raise_for_status()

total_size = int(response.headers.get('content-length', 0))
block_size = 1024 * 1024 
output_path = "generated_video.mp4"

with open(output_path, "wb") as f, tqdm(
    total=total_size,
 unit='MB',
 unit_scale=True,
 desc="Downloading video"
) as pbar:
    for chunk in response.iter_content(chunk_size=block_size):
        if chunk:
            f.write(chunk)
            pbar.update(len(chunk))

print(f"\nVideo successfully saved as → {output_path}")
print(f"Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")

try:
    from IPython.display import Video, display
    display(Video(url=video_url, embed=True, width=900))
except:
    pass