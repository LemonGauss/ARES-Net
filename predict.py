import os
import pickle
import numpy as np
import torch
import warnings
import shutil
import torchaudio
import random
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)


sys.path.insert(0, project_root)
import transforms as T

# HuggingFace / Transformers imports
from transformers import Wav2Vec2FeatureExtractor

from args import get_parser
from lib import segmentation

warnings.filterwarnings("ignore")


OVERLAY_COLOR = [0, 255, 0]
ALPHA = 0.5  

def get_transform(args):
    transforms = [
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return T.Compose(transforms)

def save_overlay_image(original_img_pil, mask_np, save_path, alpha=0.5, color=[0, 255, 0]):
    img_np = np.array(original_img_pil)

    
    
    mask_binary = (mask_np > 0).astype(float)
    mask_expanded = mask_binary[..., None]  

    overlay_color = np.array(color)

    
    # (1 - alpha * mask) * img + alpha * mask * color
    overlay = (1 - alpha * mask_expanded) * img_np + \
              alpha * mask_expanded * overlay_color

    overlay = np.clip(overlay, 0, 255)  
    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    overlay_img.save(save_path)

def predict_single_pair(args, model, output_dir, image_path, audio_path, gt_mask_path=None):

    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    a_name = os.path.splitext(os.path.basename(audio_path))[0]

    print(f"Processing pair: Image={os.path.basename(image_path)}, Audio={os.path.basename(audio_path)}")

    
    print("Extracting audio features...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.hubert_path,
        sampling_rate=16000,
        return_attention_mask=True,
        padding_value=0.0,
        do_normalize=True,
        max_length=16000 * 10
    )

    waveform, orig_sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform[0, :].unsqueeze(0)
# ========================================================
    
    # ========================================================
    target_sr = 16000
    if orig_sample_rate != target_sr:
        print(f"Notice: Audio sample rate is {orig_sample_rate}Hz. Resampling to {target_sr}Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        orig_sample_rate = target_sr  
    # ========================================================
    inputs = feature_extractor(
        waveform.squeeze(0),
        sampling_rate=orig_sample_rate,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=16000 * 10
    )

    processed_waveform = inputs.input_values.squeeze(0).cuda().unsqueeze(0)
    # attention_mask = inputs.attention_mask.squeeze(0).cuda()

    
    print("Processing image...")
    
    raw_img = Image.open(image_path).convert('RGB')
    o_W, o_H = raw_img.size

    
    transformimg = get_transform(args)
    img_tensor, _ = transformimg(raw_img, raw_img)
    img_tensor = img_tensor.unsqueeze(0).cuda()

    
    print("Running inference...")
    with torch.no_grad():
        output = model(img_tensor, processed_waveform, orig_sample_rate)
        
        if output.shape[1] > 1:
            output = output.argmax(1)

        output = output.squeeze(0)  # (H, W)

    
    zeros_tensor = torch.zeros_like(output)
    
    output_mask = torch.where(output > 0, torch.full_like(output, 255), zeros_tensor)
    output_mask_np = np.array(output_mask.cpu()).astype(np.uint8)

    
    
    image_pil = Image.fromarray(output_mask_np)
    resized_mask = image_pil.resize((o_W, o_H), Image.NEAREST)
    resized_mask_np = np.array(resized_mask)

    
    print(f"Saving results to {output_dir}...")

    
    save_mask_name = os.path.join(output_dir, a_name + '_predict.png')
    resized_mask.save(save_mask_name)

    
    save_overlay_name = os.path.join(output_dir, a_name + '_overlay.jpg')
    save_overlay_image(raw_img, resized_mask_np, save_overlay_name,
                       alpha=ALPHA, color=OVERLAY_COLOR)

    
    # shutil.copy2(image_path, os.path.join(output_dir, os.path.basename(image_path)))

    
    if gt_mask_path is not None:
        if os.path.exists(gt_mask_path):
            
            shutil.copy2(gt_mask_path, os.path.join(output_dir, base_name + '_gt.png'))
            print("Ground Truth mask copied.")
        else:
            print(f"Warning: Provided GT mask path does not exist: {gt_mask_path}")

    print("Done.")

if __name__ == '__main__':
    parse = get_parser()
    args = parse.parse_args()

    
    if torch.cuda.is_available():
        
        local_rank = args.local_rank if hasattr(args, 'local_rank') else 0
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU.")

    cudnn.benchmark = True
    cudnn.deterministic = True

    
    print(f"Initializing model: {args.model}")
    
    single_model = segmentation.__dict__[args.model](pretrained='', args=args)

    
    weight_path = ""
    if not os.path.exists(weight_path):
         print(f"Error: Weight file not found at {weight_path}")
         exit()

    print(f"Loading checkpoint from: {weight_path}")
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    single_model.load_state_dict(checkpoint['model'], strict=False)

    model = single_model.to(device)
    model.eval()

    # ==========================================
    
    # ==========================================
    
    
    target_image_path = "" 

    
    
    target_audio_path = "" 

    
    
    target_gt_mask_path = None 

    # ==========================================
    
    # ==========================================

    
    save_root = ''

    print("\n--- Starting Single Pair Prediction ---")
    
    predict_single_pair(
        args=args,
        model=model,
        output_dir=save_root,
        image_path=target_image_path,
        audio_path=target_audio_path,
        gt_mask_path=target_gt_mask_path
    )
