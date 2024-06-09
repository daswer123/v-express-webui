import os
import gradio as gr

from v_express.scripts.extract_kps_sequence_and_audio import extract_kps_sequence

# Funcs
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

with gr.Blocks() as demo:
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

if __name__ == "__main__":
    demo.queue()
    demo.launch(inbrowser=True)
