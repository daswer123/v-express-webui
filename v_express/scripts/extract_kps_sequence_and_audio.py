import os
import cv2
import torch
from insightface.app import FaceAnalysis
from imageio_ffmpeg import get_ffmpeg_exe
import dlib
from PIL import Image

def crop_face(image, face):
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    top_left_x = max(x - w // 2, 0)
    top_left_y = max(y - h // 2, 0)
    bottom_right_x = min(x + 3 * w // 2, image.shape[1])
    bottom_right_y = min(y + 3 * h // 2, image.shape[0])
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    return cropped_image

def extract_kps_sequence(video_path, kps_sequence_save_path, audio_save_path, device='cuda', gpu_id=0, insightface_model_path='./model_ckpts/insightface_models/', height=512, width=512,crop = True):
    # Initialize the face detection model
    detector = dlib.get_frontal_face_detector()
    
    app = FaceAnalysis(
        providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'],
        provider_options=[{'device_id': gpu_id}] if device == 'cuda' else [],
        root=insightface_model_path,
    )
    app.prepare(ctx_id=0, det_size=(height, width))

    os.system(f'{get_ffmpeg_exe()} -i "{video_path}" -q:a 0 -y -vn "{audio_save_path}"')

    kps_sequence = []
    video_capture = cv2.VideoCapture(video_path)
    frame_idx = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        assert len(faces) == 1, f'There are {len(faces)} faces in the {frame_idx}-th frame. Only one face is supported.'

        # Crop the face
        frame = crop_face(frame, faces[0])
        frame = cv2.resize(frame, (width, height))
        
        # Get keypoints
        faces = app.get(frame)
        assert len(faces) == 1, f'There are {len(faces)} faces in the {frame_idx}-th frame after cropping. Only one face is supported.'

        kps = faces[0].kps[:3]
        kps_sequence.append(kps)

        frame_idx += 1

    torch.save(kps_sequence, kps_sequence_save_path)
    
    # Return path to audio and kps sequence
    return kps_sequence_save_path, audio_save_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='')
    parser.add_argument('--kps_sequence_save_path', type=str, default='')
    parser.add_argument('--audio_save_path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--insightface_model_path', type=str, default='./model_ckpts/insightface_models/')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument("--crop", action="store_true")
    args = parser.parse_args()

    extract_kps_sequence(args.video_path, args.kps_sequence_save_path, args.audio_save_path, args.device, args.gpu_id, args.insightface_model_path, args.height, args.width, args.crop)
