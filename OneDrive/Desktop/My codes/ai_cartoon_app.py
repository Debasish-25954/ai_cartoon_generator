import streamlit as st
import os
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DiffusionPipeline
from diffusers import DPMSolverSinglestepScheduler
import imageio
from transformers import pipeline

# App title and configuration
st.set_page_config(page_title="AI Cartoon Video Generator", layout="wide")
st.title("🎨 AI Cartoon Video Generator")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model",
        ["Cartoon Diffusion", "Toon You", "Disney Style", "Anime Style"]
    )
    
    # Video generation parameters
    num_frames = st.slider("Number of frames", 10, 60, 24)
    duration = st.slider("Duration per frame (ms)", 50, 500, 100)
    resolution = st.selectbox("Resolution", ["512x512", "768x768", "1024x1024"])
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        cfg_scale = st.slider("Creativity (CFG Scale)", 1.0, 20.0, 7.5)
        steps = st.slider("Generation Steps", 10, 100, 25)
        seed = st.number_input("Seed", value=42)

# Main app interface
tab1, tab2 = st.tabs(["Text to Cartoon Video", "Image to Cartoon Video"])

with tab1:
    st.subheader("Create cartoon video from text")
    prompt = st.text_area("Enter your prompt", "A cute cartoon dog playing with a ball in the park, sunny day")
    
    if st.button("Generate Cartoon Video", key="text_to_video"):
        with st.spinner("Generating your cartoon video..."):
            try:
                # Load model (in a real app, you'd cache this)
                pipe = DiffusionPipeline.from_pretrained(
                    "damo-vilab/text-to-video-ms-1.7b",
                    torch_dtype=torch.float16,
                    variant="fp16"
                )
                pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
                pipe.enable_model_cpu_offload()
                
                # Generate frames
                frames = pipe(
                    prompt,
                    num_inference_steps=steps,
                    num_frames=num_frames,
                    guidance_scale=cfg_scale
                ).frames
                
                # Convert to video
                video_path = "cartoon_video.mp4"
                imageio.mimsave(video_path, frames, fps=1000/duration)
                
                # Display result
                st.success("Video generated successfully!")
                st.video(video_path)
                
                # Download button
                with open(video_path, "rb") as f:
                    st.download_button(
                        "Download Video",
                        f,
                        file_name="ai_cartoon_video.mp4",
                        mime="video/mp4"
                    )
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with tab2:
    st.subheader("Convert image to cartoon video")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        prompt = st.text_area("Enter animation description", "Make the character wink and smile")
        
        if st.button("Generate Cartoon Video", key="image_to_video"):
            with st.spinner("Animating your image..."):
                try:
                    # Load img2img model
                    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        torch_dtype=torch.float16,
                    )
                    pipe = pipe.to("cuda")
                    
                    # Generate variations
                    frames = []
                    for i in range(num_frames):
                        # Modify prompt slightly for each frame
                        frame_prompt = f"{prompt}, frame {i}/{num_frames}"
                        
                        result = pipe(
                            prompt=frame_prompt,
                            image=image,
                            strength=0.5,
                            guidance_scale=cfg_scale,
                            num_inference_steps=steps
                        ).images[0]
                        
                        frames.append(np.array(result))
                    
                    # Create video
                    video_path = "cartoon_animation.mp4"
                    imageio.mimsave(video_path, frames, fps=1000/duration)
                    
                    # Display result
                    st.success("Animation created successfully!")
                    st.video(video_path)
                    
                    # Download button
                    with open(video_path, "rb") as f:
                        st.download_button(
                            "Download Animation",
                            f,
                            file_name="ai_cartoon_animation.mp4",
                            mime="video/mp4"
                        )
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("🛠️ Powered by Stable Diffusion and Diffusers library")