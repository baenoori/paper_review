import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import time

st = time.time()
prompt = "A cat, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. \
    The cat's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other cats gather, \
        watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. \
            The cat's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage,\
                enhancing the peaceful and magical atmosphere of this unique musical performance."

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "./output_cat.mp4", fps=8)

et = time.time()

print('시간 :', round(et-st))

# 시간 : 4842

