import glob
import os
import gradio as gr

from v_express.scripts.extract_kps_sequence_and_audio import extract_kps_sequence
from v_express.inference_code import convert_video

# Funcs for Stage 1
def extract_kps_sequence_gr(video_extract, crop_face, video_extract_save_kps_path, video_extract_save_audio_path, video_extract_device, video_extract_gpu_id, video_extract_insightface_model_path, video_extract_height, video_extract_width):
    
    # Create folder kps_sequence if it doesn't exist
    if not os.path.exists("kps_sequence"):
        os.mkdir("kps_sequence")
    
    video_extract_save_kps_folder = os.path.dirname(video_extract_save_kps_path)
    if not os.path.exists(video_extract_save_kps_folder):
        os.mkdir(video_extract_save_kps_folder)
    
    video_extract_save_audio_folder = os.path.dirname(video_extract_save_audio_path)
    if not os.path.exists(video_extract_save_audio_folder):
        os.mkdir(video_extract_save_audio_folder)
    
    # Call the extraction function with the appropriate parameters
    kps_sequence_save_path, audio_save_path = extract_kps_sequence(
        video_extract,
        video_extract_save_kps_path,
        video_extract_save_audio_path,
        device=video_extract_device,
        gpu_id=video_extract_gpu_id,
        insightface_model_path=video_extract_insightface_model_path,
        height=video_extract_height,
        width=video_extract_width,
        crop=crop_face
    )
    
    print("Done")
    # Return the paths to the output files
    return [kps_sequence_save_path, audio_save_path], audio_save_path

def change_extractor_path(model_name):
    # Create folder kps_sequence/model_name_folder if it doesn't exist
    if not os.path.exists("kps_sequence"):
        os.mkdir("kps_sequence")
    
    if model_name == "":
        model_name = "test"
    
    return f"kps_sequence/{model_name}/{model_name}_sequence.pt", f"kps_sequence/{model_name}/{model_name}_audio.mp3"


# Funcs for stage 2

def reload_available_sequences():
        sequence_paths = []
        for root, dirs, files in os.walk("kps_sequence"):
            for file in files:
                if file.endswith(".pt"):
                    sequence_paths.append(os.path.join(root, file))
        return sequence_paths

def convert_video_gr(converter_save_path, converter_ref_img, converter_ref_audio, converted_prepared_sequence, convert_video_extract, convert_video_extract_crop, convert_video_extract_device, convert_video_extract_gpu_id, convert_video_extract_insightface_model_path, convert_video_extract_height, convert_video_extract_width, convert_steps, converter_strategy, converter_reference_attention_weight, converter_audio_attention_weight, converter_seed, converter_cfg, converter_device, converter_gpu_id, converter_dtype, converter_fps, converter_context_frames, converter_context_stride, context_overlap, converter_num_pad_audio_frames, converter_standard_audio_sampling_rate):
    print("Save Path:", converter_save_path)
    print("Reference Image:", converter_ref_img)
    print("Reference Audio:", converter_ref_audio)
    print("Prepared Sequence:", converted_prepared_sequence)
    print("Input Video:", convert_video_extract)
    print("Crop Face:", convert_video_extract_crop)
    print("Device:", convert_video_extract_device)
    print("GPU ID:", convert_video_extract_gpu_id)
    print("Insightface Model Path:", convert_video_extract_insightface_model_path)
    print("Height:", convert_video_extract_height)
    print("Width:", convert_video_extract_width)
    print("Steps:", convert_steps)
    print("Retarget Strategy:", converter_strategy)
    print("Reference Attention Weight:", converter_reference_attention_weight)
    print("Audio Attention Weight:", converter_audio_attention_weight)
    print("Seed:", converter_seed)
    print("CFG Scale:", converter_cfg)
    print("Device:", converter_device)
    print("GPU ID:", converter_gpu_id)
    print("Dtype:", converter_dtype)
    print("FPS:", converter_fps)
    print("Context Frames:", converter_context_frames)
    print("Context Stride:", converter_context_stride)
    print("Context Overlap:", context_overlap)
    print("Number of Pad Audio Frames:", converter_num_pad_audio_frames)
    print("Standard Audio Sampling Rate:", converter_standard_audio_sampling_rate)


# Prepare data

sequence_paths = []
sequence_paths_value = []

sequence_paths = reload_available_sequences()

if len(sequence_paths) > 0:
    sequence_paths_value = sequence_paths[0]


with gr.Blocks() as demo:
    
    # STAGE 1
    with gr.Tab("1 - Extractor"):
        with gr.Row():
            with gr.Column():
                video_extract_model_name = gr.Textbox(label="Model Name")
                video_extract = gr.Video(label="Input")
                video_extract_crop = gr.Checkbox(label="Crop the face on the video according to official guidelines", value=True)
                with gr.Accordion("Extra settings", open=False):
                    video_extract_save_kps_path = gr.Textbox(label="Save KPS Sequence Path", value="./kps_sequence/test/kps_sequence.pt")
                    video_extract_save_audio_path = gr.Textbox(label="Save Audio Path", value="./kps_sequence/test/audio.mp3")
                    with gr.Row():
                        video_extract_device = gr.Radio(label="Device", choices=["cpu", "cuda"], value="cuda")
                        video_extract_gpu_id = gr.Slider(label="GPU ID", minimum=0, maximum=10, step=1, value=0)
                    video_extract_insightface_model_path = gr.Textbox(label="Insightface Model Path", value="./model_ckpts/insightface_models/")
                    with gr.Row():
                        video_extract_height = gr.Number(label="Height", value=512)
                        video_extract_width = gr.Number(label="Width", value=512)
            with gr.Column():
                video_extract_label = gr.Label(value="Output")
                video_extract_audio = gr.Audio(interactive=False)
                video_extract_files = gr.Files(label="Output Files", interactive=False)
                video_extract_btn = gr.Button("Extract")
                
                
    # STAGE 2 
    with gr.Tab("2 - Extractor"):
            with gr.Row():
                with gr.Column():
                    converter_save_path = gr.Textbox(label="Save Path", value="./output")
                    converter_ref_img = gr.Image(label="Reference Image",type="filepath")
                    converter_ref_audio = gr.Audio(label="Reference Audio",type="filepath")
                    with gr.Tab("Prepared Sequence"):
                        with gr.Row():                           
                            converted_prepared_sequence = gr.Dropdown(label="Prepared Sequence", choices=sequence_paths, value=sequence_paths_value)
                            converted_prepared_sequence_refresh = gr.Button("Refresh")
                    with gr.Tab("Sequence from video"):
                        convert_video_extract = gr.Video(label="Input")
                        convert_video_extract_crop = gr.Checkbox(label="Crop the face on the video according to official guidelines", value=True)
                        with gr.Accordion("Extra settings", open=False):
                            with gr.Row():
                                convert_video_extract_device = gr.Radio(label="Device", choices=["cpu", "cuda"], value="cuda")
                                convert_video_extract_gpu_id = gr.Slider(label="GPU ID", minimum=0, maximum=10, step=1, value=0)
                            convert_video_extract_insightface_model_path = gr.Textbox(label="Insightface Model Path", value="./model_ckpts/insightface_models/")
                            with gr.Row():
                                convert_video_extract_height = gr.Number(label="Height", value=512)
                                convert_video_extract_width = gr.Number(label="Width", value=512)
                    with gr.Accordion("Converter settings", open=True):
                        convert_steps = gr.Slider(label="Steps", minimum=1, maximum=200, step=1, value=25)
                        converter_strategy = gr.Dropdown(label="Retarget Strategy", choices=["no_retarget", "fix_face","offset_retarget","naive_retarget"],value="no_retarget")
                        converter_reference_attention_weight = gr.Slider(label="Reference Attention Weight", minimum=0, maximum=1, step=0.01, value=0.5)
                        converter_audio_attention_weight = gr.Slider(label="Audio Attention Weight", minimum=1, maximum=5, step=0.01, value=1)
                        with gr.Accordion("Extra Settings", open=False):

                            converter_seed = gr.Textbox(label="Seed", value=42)
                            converter_cfg = gr.Slider(label="CFG Scale", minimum=0.01, maximum=20, step=0.01, value=3.5)

                            gr.Markdown("### Device Settings")
                            with gr.Row():
                                converter_device = gr.Radio(label="Device", choices=["cpu", "cuda"], value="cuda")
                                converter_gpu_id = gr.Slider(label="GPU ID", minimum=0, maximum=10, step=1, value=0)
                                converter_dtype = gr.Dropdown(label="Dtype", choices=["fp16", "fp32"], value="fp16")

                            gr.Markdown("### Video Settings")
                            with gr.Row():
                                converter_fps = gr.Slider(label="FPS", minimum=1, maximum=120, step=1, value=30)
                                converter_context_frames = gr.Slider(label="Context Frames", minimum=0, maximum=60, step=1, value=12)
                                converter_context_stride = gr.Slider(label="Context Stride", minimum=1, maximum=60, step=1, value=1)
                                context_overlap = gr.Slider(label="Context Overlap", minimum=0, maximum=60, step=1, value=4)

                            gr.Markdown("### Audio Settings")
                            with gr.Row():
                                converter_num_pad_audio_frames = gr.Slider(label="Number of Pad Audio Frames", minimum=0, maximum=60, step=1, value=2)
                                converter_standard_audio_sampling_rate = gr.Dropdown(label="Standard Audio Sampling Rate", choices=[8000, 16000, 32000, 44000, 48000], value=16000)

                with gr.Column():
                    converter_status = gr.Label(value="Status")
                    converter_ready_video = gr.Video(label="Ready Video", interactive=False)
                    converter_btn = gr.Button("Convert")
                
                
            
    # Stage 1 hanlders
    
    video_extract_model_name.change(change_extractor_path, inputs=[video_extract_model_name], outputs=[video_extract_save_kps_path, video_extract_save_audio_path])
    video_extract_btn.click(extract_kps_sequence_gr, inputs=[ 
        video_extract, 
        video_extract_crop, 
        video_extract_save_kps_path, 
        video_extract_save_audio_path, 
        video_extract_device, 
        video_extract_gpu_id, 
        video_extract_insightface_model_path, 
        video_extract_height, 
        video_extract_width
    ], outputs=[video_extract_files,video_extract_audio])
    
    # Stage 2 handlers
            
    def reload_available_sequences_gr():
        filepaths = reload_available_sequences()
        return gr.update(choices=filepaths, value=filepaths[0])
        
    
    converted_prepared_sequence_refresh.click(fn=reload_available_sequences, outputs=[converted_prepared_sequence])
    converter_btn.click(convert_video_gr, 
    inputs=
    [
        converter_save_path, converter_ref_img, converter_ref_audio, converted_prepared_sequence, convert_video_extract, convert_video_extract_crop, convert_video_extract_device, convert_video_extract_gpu_id, convert_video_extract_insightface_model_path, convert_video_extract_height, convert_video_extract_width, convert_steps, converter_strategy, converter_reference_attention_weight, converter_audio_attention_weight, converter_seed, converter_cfg, converter_device, converter_gpu_id, converter_dtype, converter_fps, converter_context_frames, converter_context_stride, context_overlap, converter_num_pad_audio_frames, converter_standard_audio_sampling_rate
    ], 
    outputs=
    [
            converter_status, converter_ready_video
    ])


if __name__ == "__main__":
    demo.queue()
    demo.launch(inbrowser=True)
