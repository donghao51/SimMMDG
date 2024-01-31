from mmaction.apis import init_recognizer, inference_recognizer
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloader_SimMMDG import EPICDOMAIN
import torch.nn.functional as F
from losses import SupConLoss


def train_one_step(model, clip, labels, spectrogram, audio_cls_model):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    spectrogram = spectrogram.unsqueeze(1).cuda()

    with torch.no_grad():
        _, audio_feat, _ = audio_model(spectrogram)
        x_slow, x_fast = model.module.backbone.get_feature(clip)  
        v_feat = (x_slow.detach(), x_fast.detach()) 
        
    v_feat = model.module.backbone.get_predict(v_feat)
    v_predict, v_emd = model.module.cls_head(v_feat)

    audio_predict, audio_emd = audio_cls_model(audio_feat.detach())

    predict = mlp_cls(v_emd, audio_emd)
    loss = criterion(predict, labels)

    # Cross-modal Translation 
    a_emd_t = mlp_v2a(v_emd)
    v_emd_t = mlp_a2v(audio_emd)
    a_emd_t = a_emd_t/torch.norm(a_emd_t, dim=1, keepdim=True)
    v_emd_t = v_emd_t/torch.norm(v_emd_t, dim=1, keepdim=True)
    v2a_loss = torch.mean(torch.norm(a_emd_t-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
    a2v_loss = torch.mean(torch.norm(v_emd_t-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
    loss = loss + args.alpha_trans*(v2a_loss + a2v_loss)/2

    # Supervised Contrastive Learning
    v_dim = int(v_emd.shape[1] / 2)
    a_dim = int(audio_emd.shape[1] / 2)
    v_emd_proj = v_proj(v_emd[:, :v_dim])
    a_emd_proj = a_proj(audio_emd[:, :a_dim])
    emd_proj = torch.stack([v_emd_proj, a_emd_proj], dim=1)
    loss_contrast = criterion_contrast(emd_proj, labels)
    loss = loss + args.alpha_contrast*loss_contrast

    # Feature Splitting with Distance
    loss_e1 = -F.mse_loss(v_emd[:, :v_dim], v_emd[:, v_dim:])
    loss_e2 = -F.mse_loss(audio_emd[:, :a_dim], audio_emd[:, a_dim:])
    
    loss = loss + args.explore_loss_coeff * (loss_e1 + loss_e2)/2

    optim.zero_grad()
    loss.backward()
    optim.step()
    return predict, loss

def validate_one_step(model, clip, labels, spectrogram, audio_cls_model):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    spectrogram = spectrogram.unsqueeze(1).cuda()

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)  
        v_feat = (x_slow.detach(), x_fast.detach())  

        v_feat = model.module.backbone.get_predict(v_feat)
        v_predict, v_emd = model.module.cls_head(v_feat)
        _, audio_feat, _ = audio_model(spectrogram)
        audio_predict, audio_emd = audio_cls_model(audio_feat.detach())

        predict = mlp_cls(v_emd, audio_emd)

    loss = criterion(predict, labels)
    return predict, loss

class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(Encoder, self).__init__()
        self.enc_net = nn.Sequential(
          nn.Linear(input_dim, hidden),
          nn.ReLU(),
          nn.Dropout(p=0.5),
          nn.Linear(hidden, out_dim)
        )
        
    def forward(self, vfeat, afeat):
        feat = torch.cat((vfeat, afeat), dim=1)
        return self.enc_net(feat)

class EncoderTrans(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(EncoderTrans, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        feat = self.enc_net(feat)
        return feat

class ProjectHead(nn.Module):
    def __init__(self, input_dim=2816, hidden_dim=2048, out_dim=128):
        super(ProjectHead, self).__init__()
        self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        
    def forward(self, feat):
        feat = F.normalize(self.head(feat), dim=1)
        return feat


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-s','--source_domain', nargs='+', help='<Required> Set source_domain', required=True)
    parser.add_argument('-t','--target_domain', nargs='+', help='<Required> Set target_domain', required=True)
    parser.add_argument('--datapath', type=str, default='/path/to/EPIC-KITCHENS/',
                        help='datapath')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='lr')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch_size')
    parser.add_argument("--nepochs", type=int, default=15)
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--alpha_trans', type=float, default=0.1,
                        help='alpha_trans')
    parser.add_argument("--trans_hidden_num", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temp')
    parser.add_argument('--alpha_contrast', type=float, default=3.0,
                        help='alpha_contrast')
    parser.add_argument('--resumef', action='store_true')
    parser.add_argument('--explore_loss_coeff', type=float, default=0.7,
                        help='explore_loss_coeff')
    parser.add_argument("--BestEpoch", type=int, default=0)
    parser.add_argument('--BestAcc', type=float, default=0,
                        help='BestAcc')
    parser.add_argument('--BestTestAcc', type=float, default=0,
                        help='BestTestAcc')
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

    device = 'cuda:0' 
    device = torch.device(device)

    model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(2304, 8).cuda()
    cfg = model.cfg
    model = torch.nn.DataParallel(model)

    audio_args = get_arguments()
    audio_model = AVENet(audio_args)
    checkpoint = torch.load("pretrained_models/vggsound_avgpool.pth.tar")
    audio_model.load_state_dict(checkpoint['model_state_dict'])
    audio_model = audio_model.cuda()

    audio_cls_model = AudioAttGenModule()
    audio_cls_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    audio_cls_model.fc = nn.Linear(512, 8)
    audio_cls_model = audio_cls_model.cuda()
        
    mlp_v2a = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=512)
    mlp_a2v = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2304)
    mlp_v2a = mlp_v2a.cuda()
    mlp_a2v = mlp_a2v.cuda()

    mlp_cls = Encoder(input_dim=2304+512, out_dim=8)
    mlp_cls = mlp_cls.cuda()

    v_proj = ProjectHead(input_dim=1152, hidden_dim=args.hidden_dim, out_dim=args.out_dim)
    a_proj = ProjectHead(input_dim=256, hidden_dim=args.hidden_dim, out_dim=args.out_dim)
    v_proj = v_proj.cuda()
    a_proj = a_proj.cuda()

    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    base_path_model = "models/"
    if not os.path.exists(base_path_model):
        os.mkdir(base_path_model)

    log_name = "log%s2%s_video_audio"%(args.source_domain, args.target_domain)

    log_name = log_name + args.appen
    log_path = base_path + log_name + '.csv'
    print(log_path)

    criterion = nn.CrossEntropyLoss() 
    criterion = criterion.cuda()
    batch_size = args.bsz

    criterion_contrast = SupConLoss(temperature=args.temp)
    criterion_contrast = criterion_contrast.cuda()

    params = list(model.module.backbone.fast_path.layer4.parameters()) + list(
        model.module.backbone.slow_path.layer4.parameters()) +list(model.module.cls_head.parameters())+list(audio_cls_model.parameters())
    params = params + list(mlp_cls.parameters())
    params = params + list(mlp_v2a.parameters())+list(mlp_a2v.parameters())
    params = params + list(v_proj.parameters())+list(a_proj.parameters())

    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)

    BestLoss = float("inf")
    BestEpoch = args.BestEpoch
    BestAcc = args.BestAcc
    BestTestAcc = args.BestTestAcc

    if args.resumef:
        resume_file = base_path_model + log_name + '.pt'
        print("Resuming from ", resume_file)
        checkpoint = torch.load(resume_file)
        starting_epoch = checkpoint['epoch']+1
    
        BestLoss = checkpoint['BestLoss']
        BestEpoch = checkpoint['BestEpoch']
        BestAcc = checkpoint['BestAcc']
        BestTestAcc = checkpoint['BestTestAcc']

        model.load_state_dict(checkpoint['model_state_dict'])
        audio_model.load_state_dict(checkpoint['audio_model_state_dict'])
        audio_cls_model.load_state_dict(checkpoint['audio_cls_model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        mlp_v2a.load_state_dict(checkpoint['mlp_v2a_state_dict'])
        mlp_a2v.load_state_dict(checkpoint['mlp_a2v_state_dict'])
        mlp_cls.load_state_dict(checkpoint['mlp_cls_state_dict'])
        v_proj.load_state_dict(checkpoint['v_proj_state_dict'])
        a_proj.load_state_dict(checkpoint['a_proj_state_dict'])
    else:
        print("Training From Scratch ..." )
        starting_epoch = 0

    print("starting_epoch: ", starting_epoch)
    audio_model.eval()

    train_dataset = EPICDOMAIN(split='train', domain=args.source_domain, modality='rgb', cfg=cfg, use_audio=True, datapath=args.datapath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)

    val_dataset = EPICDOMAIN(split='test', domain=args.source_domain, modality='rgb', cfg=cfg, use_audio=True, datapath=args.datapath)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)

    test_dataset = EPICDOMAIN(split='test', domain=args.target_domain, modality='rgb', cfg=cfg, use_audio=True, datapath=args.datapath)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    with open(log_path, "a") as f:
        for epoch_i in range(starting_epoch, args.nepochs):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val', 'test']:
                acc = 0
                count = 0
                total_loss = 0
                print(split)
                model.train(split == 'train')
                audio_cls_model.train(split == 'train')
                mlp_cls.train(split == 'train')
                mlp_v2a.train(split == 'train')
                mlp_a2v.train(split == 'train')
                v_proj.train(split == 'train')
                a_proj.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (clip, spectrogram, labels)) in enumerate(dataloaders[split]):
                        if split=='train':
                            predict1, loss = train_one_step(model, clip, labels, spectrogram, audio_cls_model)
                        else:
                            predict1, loss = validate_one_step(model, clip, labels, spectrogram, audio_cls_model)

                        total_loss += loss.item() * batch_size
                        _, predict = torch.max(predict1.detach().cpu(), dim=1)

                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)
                        count += predict1.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                  loss.item(),
                                                                                                  acc / float(count)))
                        pbar.update()

                    if split == 'val':
                        currentvalAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:
                            BestLoss = total_loss / float(count)
                            BestEpoch = epoch_i
                            BestAcc = acc / float(count)
                            

                    if split == 'test':
                        currenttestAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:
                            BestTestAcc = currenttestAcc
                            if args.save_best:
                                save = {
                                    'epoch': epoch_i,
                                    'BestLoss': BestLoss,
                                    'BestEpoch': BestEpoch,
                                    'BestAcc': BestAcc,
                                    'BestTestAcc': BestTestAcc,
                                    'model_state_dict': model.state_dict(),
                                    'audio_model_state_dict': audio_model.state_dict(),
                                    'audio_cls_model_state_dict': audio_cls_model.state_dict(),
                                    'optimizer': optim.state_dict(),
                                }
                                save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                                save['mlp_cls_state_dict'] = mlp_cls.state_dict()
                                save['v_proj_state_dict'] = v_proj.state_dict()
                                save['a_proj_state_dict'] = a_proj.state_dict()

                                torch.save(save, base_path_model + log_name + '_best_%s.pt'%(str(epoch_i)))

                        if args.save_checkpoint:
                            save = {
                                'epoch': epoch_i,
                                'BestLoss': BestLoss,
                                'BestEpoch': BestEpoch,
                                'BestAcc': BestAcc,
                                'BestTestAcc': BestTestAcc,
                                'model_state_dict': model.state_dict(),
                                'audio_model_state_dict': audio_model.state_dict(),
                                'audio_cls_model_state_dict': audio_cls_model.state_dict(),
                                'optimizer': optim.state_dict(),
                            }
                            save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                            save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                            save['mlp_cls_state_dict'] = mlp_cls.state_dict()
                            save['v_proj_state_dict'] = v_proj.state_dict()
                            save['a_proj_state_dict'] = a_proj.state_dict()

                            torch.save(save, base_path_model + log_name + '.pt')
                        
                    f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count)))
                    f.flush()

                    print('acc on epoch ', epoch_i)
                    print("{},{},{}\n".format(epoch_i, split, acc / float(count)))
                    print('BestValAcc ', BestAcc)
                    print('BestTestAcc ', BestTestAcc)
                    
                    if split == 'test':
                        f.write("CurrentBestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc))
                        f.flush()

        f.write("BestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc))
        f.flush()

        print('BestValAcc ', BestAcc)
        print('BestTestAcc ', BestTestAcc)

    f.close()
