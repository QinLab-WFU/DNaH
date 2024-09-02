from utils.tools import *
# from model.network import *
from torch.autograd import Variable
# import os
from fvcore.nn import FlopCountAnalysis,parameter_count_table
import torch
import torch.optim as optim
import time
import numpy as np
from loguru import logger
#from model.nest import Nest
from model.vit import VisionTransformer, VIT_CONFIGS
# from torch.autograd import Variable
from ptflops import get_model_complexity_info
from apex import amp
from utils.Hash_loss import HashNetLoss
torch.multiprocessing.set_sharing_strategy('file_system')
from model.label_net import LabelNet
from relative_similarity import *
from centroids_generator import *
import torch.nn.functional as F
from utils.loss import *
from timm.models.nest import nest_base
import apex
from save_mat import Save_mat
from model.swin import build_model
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
from model.network import AlexNet
from NEST import Nest
import datetime
from NEST import nest_base
import jax
import flax
def get_config():
    config = {
        #"alpha": 0.1,
        # "alpha": 0.5,


        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 1e-4, "weight_decay": 1e-5}, "lr_type": "step"},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 1e-5}, "lr_type": "step"},
        # "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5, "weight_decay": 1e-5}, "lr_type": "step"},
        "info": "[NestNet_fusion_avgpool]",
        #"info": "[NestNet_fusion_maxpool]",
        # "info": "[Resnet101]",
        # "info":"AlexNet",
        # "info": "[NestNet]",
        #"info": "[ViT]",
        #"info": "[SWIN]",
        "step_continuation": 20,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 32, # O2 -> 128 O1 -> less
        #"datasets": "cifar10",
        # "datasets": "cifar10-1",
        # "datasets": "cifar10-2",
        #"datasets": "coco",
        #"datasets": "imagenet",
        #"datasets": "nuswide_21",
        "datasets": "mirflickr",
        # "datasets": "nuswide_21_m",
        #"datasets": "nuswide_81_m",
        "Label_dim" : 38, # nclass
        "epoch": 100,
        "test_map": 0,
        "save_path": "save/HashNet",
        "device": torch.device("cuda:0"),
        'test_device':torch.device("cuda:1"),
        "bit_list": [32],
        # "bit_list": [48],
        "pretrained_dir":"/home/admin01/桌面/DNaH/jx_nest_base-8bc41011.pth",
        #"pretrained_dir": "checkpoint/jx_nest_small-422eaded.pth",
        #"pretrained_dir":"/home/abc/下载/NesT_hashing/TDH-main/ViT-B_16.npz",
        #"pretrained_dir":"/home/abc/下载/NesT_hashing/TDH-main/swin_tiny_patch4_window7_224.pth",
        #"pretrained_dir":"ViT-B_32.npz",
        "img_size": 224,
        "patch_size": 4,
        "in_chans": 3,
        "num_work": 10,
        "model_type": "ViT-B_16",
        "top_img": 10
    }
    config = config_dataset(config)
    return config


def train_val(config, bit):
    # Prepare model
    configs = VIT_CONFIGS[config["model_type"]]
    #total_time=0
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["bit"] = bit
    #net=build_model(config)
    #net = config["net"](bit).to(device)
    # # 加入映射的hash位数（Add hash bits）
    #net = VisionTransformer(configs, config["img_size"],zero_head=True, num_classes=config["n_class"], vis=True,hash_bit=config["bit"])
    #net = Nest(config, num_levels=3, embed_dims=(128, 256, 512), num_heads=(4, 8, 16), depths=(2, 2, 14))
    #net=nest_base(pretrained=False)
    #net=nest_base()
    net=Nest(config, num_levels=3, embed_dims=(128, 256, 512), num_heads=(4, 8, 16), depths=(2, 2, 14))
    L_net = LabelNet(code_len=bit,label_dim=config["Label_dim"])
    # # net = Nest(config, num_levels=3, embed_dims=(96, 192, 384), num_heads=(3, 6, 12), depths=(2, 2, 14))
    # if config["pretrained_dir"] is not None:
    #     logger.info('Loading:', config["pretrained_dir"])
    #     state_dict = torch.load(config["pretrained_dir"])
    #     net.load_state_dict(state_dict, strict=False)
    #     logger.info('Pretrain weights loaded.')
    # if config["pretrained_dir"] is not None:
    #     logger.info('Loading:', config["pretrained_dir"])
    #     state_dict = torch.load(config["pretrained_dir"])
    #     net.load_state_dict(state_dict, strict=False)
    #     logger.info('Pretrain weights loaded.')
    # file = '/home/admin01/桌面/grad_cam/NEST/nest_b.flax'
    # with open(file, 'rb') as f:
    #     flax_data = f.read()
    # flax_param=flax.serialization.from_bytes(net,flax_data)
    # net=net.apply(flax_param)
    # print(11)
    # exit()
    net.to(config["device"])
    L_net.to(config["device"])

    # 计算模型计算力和参数量（Statistical model calculation and number of parameters）
    flops, num_params = get_model_complexity_info(net,(3,224,224), as_strings=True, print_per_layer_stat=False)
    logger.info("Total Parameter: \t%s" % num_params)
    logger.info("Total Flops: \t%s" % flops)
    tensor=(torch.rand(1,3,224,224)).to(config["device"])
    flops=FlopCountAnalysis(net,tensor)
    print("FLOPS:",flops.total())
    print(parameter_count_table(net))
    # exit()
    # optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-5)
    # L_optimizer = torch.optim.SGD(L_net.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
    L_optimizer = torch.optim.Adam(L_net.parameters(),lr=1e-5)
    # apex加速训练
    # help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
    # See details at https://nvidia.github.io/apex/amp.html"

    relative_similarity = RelativeSimilarity(nbit=bit, nclass=config["Label_dim"], batchsize=config["batch_size"])
    rela_optimizer = optim.Adam(relative_similarity.parameters(), lr = 1e-5)

    quan_loss = RelaHashLoss(multiclass=True)  # cifar10 _ False
    Best_mAP = 0
    [net, L_net], [optimizer, L_optimizer, rela_optimizer] = amp.initialize(models=[net, L_net],
                                                              optimizers=[optimizer, L_optimizer,rela_optimizer],
                                                              opt_level='O2',
                                                              num_losses=2)
    amp._amp_state.loss_scalers[0]._loss_scale = 5
    amp._amp_state.loss_scalers[1]._loss_scale = 1
    total_time = 0
    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        logger.info("%s[%2d/%2d][%s] bit:%d, datasets:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["datasets"]), end="")

        net.train()
        L_net.train()
        L_net.set_alpha(epoch)
        logger.info('Epoch [%d/%d], alpha for LabelNet: %.3f' % (epoch + 1, config["epoch"], L_net.alpha))
        train_loss = 0
        start_time = time.time()
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            _,_,label_output = L_net(label.to(torch.float64))
            ss_ = (label.to(torch.float64) @ label.to(torch.float64).t() > 0) * 2 - 1
            u  = net(image)
            u = F.normalize(u) # 必不可少 ， 如果你不想NAN
            logits = relative_similarity(u)
            q_loss = quan_loss(logits,label) # RelaHashLoss
            loss = torch.mean(torch.square((torch.matmul(u, label_output.t())) - ss_))
            train_loss += loss.item() + q_loss.item()
            with amp.scale_loss(q_loss , [optimizer,rela_optimizer] ,loss_id=0) as scaled_loss :
                scaled_loss.backward(retain_graph=True)
            with amp.scale_loss(loss , [optimizer,L_optimizer] ,loss_id=1) as scaled_loss:
                 scaled_loss.backward()
            optimizer.step()
            rela_optimizer.step()
            L_optimizer.step()
        total_time += time.time() - start_time
        # end_time = time.time()
        #print(total_time)
        # total_time += time.time()-start_time
        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b Traintime:%.4f" % (total_time))
        # print(train_loss,type(train_loss))
        # exit()
        # logger.info(
        # f">>>>>> [{epoch}/{config['datasets']}] loss: {train_loss.data}, lr: {'-'.join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))])}, time: {total_time}")
        logger.info("\b\b\b\b\b\b\b train_loss:%.4f" % (train_loss))
        logger.info("\b\b\b\b\b\b\b train_time:%.4f" % (total_time))
        #if (epoch + 1) % config["test_map"] == 0:
        if (epoch + 1) %1==0 or (epoch + 1) ==100 :
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            logger.info('Training time {}'.format(total_time_str))
            Best_mAP, index_img = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, 10)
            net.to(config["device"])
        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # logger.info('Training time {}'.format(total_time_str))
        # print('train time1', total_time)

if __name__ == "__main__":
    config = get_config()
    # 建立日志文件（Create log file）
    logger.add('llogs/{time}' + config["info"] + '2' + config["datasets"] + ' m '+str(0.9) + '.log', rotation='50 MB', level='DEBUG')

    logger.info(config)
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/alexnet/HashNet_{config['datasets']}_{bit}.json"
        train_val(config, bit)
