from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
# from bert.modeling_bert import BertModel
# import os
# from peft import LoraConfig, get_peft_model
# os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from transformers import HubertModel

class AudioProcessor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = HubertModel.from_pretrained("./facebook/hubert-base-ls960", local_files_only=True)
        self.sample_rate = 16000
        
        
        
        
        
        # peft_config = LoraConfig(
        #     inference_mode=False, 
        #     r=8, 
        #     lora_alpha=16, 
        #     lora_dropout=0.1,
        
        # )
        
        
        
        # self.encoder = get_peft_model(self.encoder, peft_config)
        
        
        # self.encoder.print_trainable_parameters()
        
        
        
        
        
                
        
        
        self.target_length = 513 * 320 
        # ==========================================================
        self.audio_max_len = args.audio_max_len  
        # self.freeze_audio=args.freeze_audio
        # print("self.freeze_audio",self.freeze_audio)
        
        # if args.freeze_audio:
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        # self.conv1d = nn.Conv1d(
        #     in_channels=768,
        #     out_channels=768,
        #     kernel_size=17,
        
        # )
        # self.adapter = nn.Sequential(
        #             nn.Linear(768, 768),
        #             nn.LayerNorm(768),
        #             nn.ReLU(),
        
        #             nn.Linear(768, 768)
        #         )
    
    def forward(self, waveform, sample_rate):

                
        current_length = waveform.shape[1]
        
        if current_length > self.target_length:
            
            processed_waveform = waveform[:, :self.target_length]
        elif current_length < self.target_length:
            
            pad_amount = self.target_length - current_length
            processed_waveform = F.pad(waveform, (0, pad_amount))
        else:
            
            processed_waveform = waveform
        # ==========================================================
        
        
        waveform=processed_waveform
        
        outputs = self.encoder(waveform)
        features = outputs.last_hidden_state  # [B, T, 768]
        # features = self.adapter(features)
            # print("features",features.shape)
         
        features = features.permute(0, 2, 1)  # [B, 768, T]
        
        

        # print("features",features.shape)
        
        # features = self.adapter(features)  # [B, T, 256]
        # features = features.permute(0, 2, 1)  # [B, 256, T]
        
        
        return features


def load_weights(model, load_path):
    dict_trained = torch.load(load_path)['model']
    dict_new = model.state_dict().copy()
    for key in dict_new.keys():
        if key in dict_trained.keys():
            dict_new[key] = dict_trained[key]
    model.load_state_dict(dict_new)
    del dict_new
    del dict_trained
    torch.cuda.empty_cache()
    print('load weights from {}'.format(load_path))
    return model


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
        self.audio_processor = AudioProcessor(args)
        
        
        self.audio_max_len = args.audio_max_len

    def forward(self, x, waveform, sample_rate):
        input_shape = x.shape[-2:]
        
        audio_feats = self.audio_processor(waveform, sample_rate)  # [B, 256, T]
        ##########################
         
        l_mask = torch.ones(audio_feats.size(0), audio_feats.size(2)).to(audio_feats.device)
        l_mask = l_mask.unsqueeze(dim=-1)  # [B, T, 1]
        
        
        input_shape = x.shape[-2:]

        features = self.backbone(x, audio_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4  = features   # e.g. x_c1:[B, 128, 120, 120], x_c2:[B, 256, 60, 60], x_c3:[B, 512, 30, 30], x_c4:[B, 1024, 15, 15]
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        return x


class LAVTOne(_LAVTOneSimpleDecode):  #change
    pass
