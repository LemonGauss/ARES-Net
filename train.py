import datetime
import os
import time
import torch
import torch.utils.data
import wandb
import cv2
import random
import transforms as T
import utils
import numpy as np
import gc
import operator
from functools import reduce
from transformers import BertModel
from lib import segmentation
from loss.loss import Loss



def seed_everything(seed=2401):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset(image_set, transform, args, eval):
    from data.all_dataset import All_Dataset
    ds = All_Dataset(args,
                      split=image_set,
                      max_tokens=30,
                      image_transforms=transform,
                      eval_mode=eval
                      )
    num_classes = 2

    return ds, num_classes


def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union


def get_transform(args):
    transforms = [
                  T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]
    return T.Compose(transforms)


def criterion(input, target, weight=0.1):
    return Loss(weight=weight)(input, target)


def evaluate(model, data_loader, bert_model, epoch):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "
    total_its = 0 # 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0 # 0, 0 
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    total_loss = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target,exist, waveforms, sample_rates = data

            pixels = cv2.countNonZero(target.data.numpy()[0]) / 230400.
            image, target, waveforms = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               waveforms.cuda(non_blocking=True)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                output = model(image, embedding, l_mask=attentions)
            else:
                output = model(image, waveforms, sample_rates )

            iou, I, U = IoU(output, target)
            loss = criterion(output, target)
            total_loss += loss.item()
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    if args.local_rank == 0:
        wandb.log({
            "val mIoU": mIoU,
            "val oiou": cum_I * 100. / cum_U,
            "val Loss": total_loss / total_its})


    return 100 * iou, 100 * cum_I / cum_U


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model,accumulation_steps=4):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    # for data in data_loader:
    for i, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        total_its += 1
        # image, target, sentences, attentions, exist = data
        # image, target, sentences, attentions = image.cuda(non_blocking=True),\
        #                                        target.cuda(non_blocking=True),\
        #                                        sentences.cuda(non_blocking=True),\
        #                                        attentions.cuda(non_blocking=True)
        image, target,exist, waveforms, sample_rates = data
        # sentences = sentences.squeeze(1)
        # attentions = attentions.squeeze(1)
        image, target, waveforms = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               waveforms.cuda(non_blocking=True)
        if bert_model is not None:

            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
            output = model(image, embedding, attentions)#, sentences_hidden_state)# [4,2,120,120]
        else:
            # output = model(image, sentences, attentions)#, sentences_hidden_state)
            # print(waveforms.shape)
            output = model(image, waveforms, sample_rates )#, sentences_hidden_state)
        # optimizer.zero_grad()
        loss = criterion(output, target)
        loss = loss / accumulation_steps
        loss.backward()
        # optimizer.step()
        # lr_scheduler.step()
        if (i + 1) % accumulation_steps == 0:
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if args.local_rank == 0:
        wandb.log({
            "Train Loss": train_loss / total_its,})


def main(args):
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args, eval=False)

    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args, eval=True)
    
    # print(dataset_test)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module  #ddp

    # for name, param in single_model.named_parameters():
    #     if 'blocks' in name:
    #         print("-----------------------blocks",name)
    #         param.requires_grad = False
    #     if 'text_encoder' in name:
    #             print("-----------------------text",name)
    #             param.requires_grad = False
    # print(model)
    if args.model != 'lavt_one':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        single_bert_model = bert_model
    else:
        bert_model = None
        single_bert_model = None

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        missing_keys, unexpected_keys = single_model.load_state_dict(checkpoint['model'], strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        if args.model != 'lavt_one':
            single_bert_model.load_state_dict(checkpoint['bert_model'])
#----------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    
    audio_train_mode = 'all'  

    
    if hasattr(single_model, 'audio_processor'):
        hubert_encoder = single_model.audio_processor.encoder
        
        
        for param in hubert_encoder.parameters():
            param.requires_grad = False
            
        audio_params_to_train = []
        
        if audio_train_mode == 'all':
            
            for param in hubert_encoder.parameters():
                param.requires_grad = True
                audio_params_to_train.append(param)
            print("Audio Config: Training ALL layers of Hubert.")
            
        elif audio_train_mode == 'last_3':
            
            
            
            
            
            layers = hubert_encoder.encoder.layers
            total_layers = len(layers)
            target_layers_indices = [total_layers - 1, total_layers - 2, total_layers - 3] 
            
            print(f"Audio Config: Training only Hubert layers: {target_layers_indices}")
            
            
            for i in target_layers_indices:
                for param in layers[i].parameters():
                    param.requires_grad = True
                    audio_params_to_train.append(param)
            
            
            
            if hasattr(hubert_encoder.encoder, 'layer_norm'):
                 for param in hubert_encoder.encoder.layer_norm.parameters():
                     param.requires_grad = True
                     audio_params_to_train.append(param)
                     
        else:
            print("Audio Config: Hubert is completely FROZEN.")

    else:
        
        audio_params_to_train = []
        print("No audio processor found in model.")
    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)
    for ename,ep in single_model.audio_processor.encoder.named_parameters():
        print(ename)
    # audio_params = [p for p in single_model.audio_processor.encoder.parameters() if p.requires_grad]
    if args.model != 'lavt_one':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            # {"params": reduce(operator.concat,
            #                   [[p for p in single_bert_model.module.encoder.layer[i].parameters()
            #                     if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            # {"params": reduce(operator.concat,
            #                   [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
            #                     if p.requires_grad] for i in range(10)])},
            {"params": audio_params_to_train},
        ]

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                                  lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)



    
    
    accumulation_steps = 4

    
    
    
    num_training_steps = int(len(data_loader) * args.epochs / accumulation_steps)
    
    
    def get_scheduler(optimizer, num_warmup_steps, num_training_steps, power=0.9):
        def lr_lambda(current_step):
            
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 1.0 - progress) ** power
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    
    num_warmup_steps = int(num_training_steps * 0.05) 
    print(f"Total optimization steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
    
    lr_scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1
    best_mIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    
    else:
        resume_epoch = -999

    # training loops
    if args.local_rank == 0:
        wandb.watch(model, log="all")


    for epoch in range(max(0, resume_epoch+1), args.epochs):
    # for epoch in range(0, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model, accumulation_steps=accumulation_steps)
        iou, overallIoU = evaluate(model, data_loader_test, bert_model, epoch)
        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        best = (best_oIoU < overallIoU)
        best_miou=(best_mIoU<iou)
        if single_bert_model is not None:
            dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}
        else:
            dict_to_save = {'model': single_model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}

        if best:
            print('Better epoch: {}\n'.format(epoch))
            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_oiou_{}.pth'.format(args.model_id)))
            best_oIoU = overallIoU
        if best_miou:
            print('Better miou epoch: {}\n'.format(epoch))
            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_miou_{}.pth'.format(args.model_id)))
            best_mIoU = iou
        if str(epoch)=="46":
            print('46 epoch: {}\n'.format(epoch))
            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_46_{}.pth'.format(args.model_id)))
        if epoch>=40:
            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_{}.pth'.format(epoch)))
        utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                'model_last.pth'))
        if args.local_rank == 0:
            wandb.save('model.h5')

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser

    seed_everything()
    parser = get_parser()
    args = parser.parse_args()

    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))    

    if args.local_rank == 0:
        wandb.init(project="rmsin_2080")
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
