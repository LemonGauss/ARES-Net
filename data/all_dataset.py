import os
import json

import cv2
from PIL import Image
import torch.utils.data as data
import torchaudio
from tqdm import tqdm  
from transformers import BertTokenizer, Wav2Vec2FeatureExtractor
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from pycocotools import mask

class All_Dataset(data.Dataset):
    def __init__(self,
                 args,
                 image_transforms=None,
                 max_tokens=40,
                 split='train',
                 eval_mode=True,
                 logger=None) -> None:
        """
        parameters:
            args: argparse obj
            image_transforms: transforms apply to image and mask
            max_tokens: determined the max length of token
            split: ['train','val','testA','testB']
            eval_mode: whether in training or evaluating
        """

        self.classes = []
        self.image_transforms = image_transforms
        self.split = split
        self.args = args
        self.eval_mode = eval_mode
        self.max_tokens = max_tokens
        self.image_path = args.imageFolder
        self.ref_file = args.ref_file_path
        self.instances_path=args.instances_path
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
         
        self.audio_path = args.audioFolder
        self.sample_rate = 16000
        self.audio_max_len = args.audio_max_len  
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.hubert_path,
            sampling_rate=16000,
            return_attention_mask=True,
            padding_value=0.0,
            do_normalize=True,
            max_length=16000 * 10  
        )
        
        self.maskToAudioPath,self.maskToSentence,self.maskToimageName,self.maskTosplit,self.maskToCategory,self.maskToDecodedRLE,self.masktoArea,self.list_mask = self.load_captions()
        # self.refToInput,self.refToAttention,self.refToExist=self.get_sent_embeddings()
        if logger:
            logger.info(f"=> loaded successfully '{args.dataset}', split by {args.splitBy}, split {split}")
        #load ref ,find caption ,image,according to mask
    def load_captions(self):
        self.maskToAudioPath = {}  

        maskToSentence = {}
        maskToimageName = {}
        maskTosplit = {}
        maskToCategory = {}
        masktoRLE={}
        masktoArea={}
        maskToDecodedRLE={}
        list_mask=[]
        with open(self.ref_file, 'rb') as f:
            data = pickle.load(f)
            total_lines = len(data)  
            pbar = tqdm(data, desc="Loading captions", total=total_lines)  
            for line in pbar:
                split_line = line["split"]
                ann_id = line["ann_id"]
                audio_name=ann_id.replace("png","wav")
                img_name = line["file_name"]
                img_id = line["image_id"]
                category_id = line["category_id"]
                if split_line == self.split :
                    list_mask.append(ann_id)
                    self.maskToAudioPath[ann_id] = audio_name

                    maskToSentence[ann_id] = line["sentences"][0]["sent"]
                    maskToimageName[ann_id]=img_name
                    maskTosplit[ann_id]=split_line
                    maskToCategory[ann_id]=category_id

                # Only load the ann_id of the corresponding training set/validation set/test set.
            pbar.close()  
        with open(self.instances_path, 'r') as fi:
            data_json = json.load(fi)
            annotations = data_json['annotations']
            total_annotations = len(annotations)  
            pbar = tqdm(annotations, desc="Loading instance data", total=total_annotations)  
            for data in pbar:
                ann_id = data["id"]

                rle = data['segmentation']
                
                maskToDecodedRLE[ann_id] = rle

                masktoArea[ann_id] = data['area']
            pbar.close()  

        return self.maskToAudioPath,maskToSentence,maskToimageName,maskTosplit,maskToCategory,maskToDecodedRLE,masktoArea,list_mask
    def get_sent_embeddings(self):
        refToInput={}
        refToAttention={}
        refToExist={}
        total_mask_ids = len(self.list_mask)  
        pbar = tqdm(self.list_mask, desc="Generating sentence embeddings", total=total_mask_ids)  
        for mask_id in pbar:
            attention_mask = [0] * self.max_tokens
            padded_input_id = [0] * self.max_tokens
            ref=self.maskToSentence[mask_id]
            input_id=self.tokenizer.encode(text=ref, add_special_tokens=True)
            input_id = input_id[:self.max_tokens]

            padded_input_id[:len(input_id)] = input_id
            attention_mask[:len(input_id)] = [1] * len(input_id)

            input_id=torch.tensor(padded_input_id).unsqueeze(0)
            attention_mask=torch.tensor(attention_mask).unsqueeze(0)
            exist = torch.Tensor([True])
            refToInput[mask_id]=input_id
            refToAttention[mask_id]=attention_mask
            refToExist[mask_id]=exist
        pbar.close()  
        return refToInput,refToAttention,refToExist
    def get_exist(self, ref, sent_index):
        if "exist" in ref["sentences"][sent_index].keys():
            exist = torch.Tensor([ref["sentences"][sent_index]["exist"]])
        else:
            exist = torch.Tensor([True])
        return exist

    def __len__(self):
        return len(self.list_mask)

    def __getitem__(self, index):
        # print("-------------------------------")
        mask_id=self.list_mask[index]
        img_name = self.maskToimageName[mask_id]
        audio_name=self.maskToAudioPath[mask_id]
        # ref=self.maskToSentence[mask_id]
        # split=self.maskTosplit[mask_id]
        try:
            img = Image.open(os.path.join(self.image_path, img_name)).convert("RGB")
        except (OSError, ValueError) as e:  
            # print(f"Error loading image {img_name}: {e}")  
            
            if img_name.lower().endswith('.png'):  
                new_img_name = img_name[:-4] + '.jpg'  
                img = Image.open(os.path.join(self.image_path, new_img_name)).convert("RGB") 
            elif img_name.lower().endswith('.jpg'):  
                new_img_name = img_name[:-4] + '.png'  
                img = Image.open(os.path.join(self.image_path, new_img_name)).convert("RGB") 
            else:  
                print("Unsupported file type.")  
                return None  
        
        rle = self.maskToDecodedRLE[mask_id]
        m = mask.decode(rle)  

        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        ref_mask = m.astype(np.uint8)  # convert to np.uint8
        # compute area


        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1
        # convert it to a Pillow image
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")


        exist = torch.Tensor([True])
       
        
        
        if self.image_transforms is not None:
            # involves transform from PIL to tensor and mean and std normalization
            img, target = self.image_transforms(img, annot)
        else:
            target = annot
        if self.eval_mode:

            # tensor_embeddings=tensor_embeddings.unsqueeze(-1)
            # attention_mask=attention_mask.unsqueeze(-1)

            exist=exist.unsqueeze(1)
         
        audio_path = os.path.join(self.audio_path, audio_name)
        # print(audio_name)
        waveform, orig_sample_rate = torchaudio.load(audio_path)

        
        
        
        if waveform.shape[0] > 1:
            waveform = waveform[0, :].unsqueeze(0)
        
        
        if orig_sample_rate != self.sample_rate:
            
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        
        inputs = self.feature_extractor(
            waveform.squeeze(0),  
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=16000 * 10  
        )

        
        processed_waveform = inputs.input_values.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        return img, target, exist,processed_waveform, orig_sample_rate
