import streamlit as st
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Streamlit UI
st.title("AI Video Generator")
st.write("Enter a prompt to generate a video:")

prompt = st.text_area("Prompt", "A fluffy golden retriever puppy and a tiny tabby kitten cuddle together on a soft, knitted blanket in a cozy living room.")

if st.button("Generate Video"):
    # Load pipeline
    with st.spinner("Loading model..."):
        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-5b",
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()

    # Generate video
    with st.spinner("Generating video..."):
        video_frames = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

        # Save video
        output_path = "./output_video_cat_1.mp4"
        export_to_video(video_frames, output_path, fps=8)

    # Display video
    st.success("Video generation completed!")
    st.video(output_path)
    