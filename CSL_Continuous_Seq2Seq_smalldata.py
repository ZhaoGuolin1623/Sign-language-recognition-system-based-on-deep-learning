
#针对小数据集测试

import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms


from dataset_video_smalldata import CSL_Continuous# CSL_Continuous_Char和CSL_Continuous函数里的read_images函数只能处理图片而不能处理视频


from models.Seq2Seq import Encoder, Decoder, Seq2Seq
from train import train_seq2seq
from validation import val_seq2seq


# Path setting
data_path = "./dataset/color_small_train"#训练数据集路径
dict_path = "./dictionary.txt"# 字典，手语里的词汇，这里词汇不多，为500个
corpus_path = "./corpus.txt"#训练语料库，就是手语的所有可能语句，这里只有100句。真实场景可能有很多，甚至无限多。
model_path = "Data/seq2seq_models" #训练好的模型权重文件存储位置，就是太上老君炼丹之后所得到的先但.pth文件存储位置
log_path = "log/seq2seq_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
sum_path = "runs/slr_seq2seq_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())


# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')
writer = SummaryWriter(sum_path)

# Use specific gpus
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
#########################################################################################################################
#同学们训练时，只需要改两个##号行之间的参数即可。其他的一般不动
num_sentences = 2#corups里有，视频是根据corups搞的 # self.signers = 50#数据集里及个人打视频，越多模型越好。
signers = 1#为了测试，就用1条视频去测试，
repetition = 5#重复数目，每次喂batchsize个数据，里面可以有重复的，这里recetition加重了这种重复性。
epochs = 5#针对这个小数据集，大约50就能准确率100%,就是整个数据集共循环多少次，一般辣酱次数越多月精确。
batch_size = 8#一次喂多少条数据进去
########################################################################################################################

learning_rate = 1e-4
weight_decay = 1e-5
sample_size = 128
sample_duration = 48
enc_hid_dim = 512
emb_dim = 256
dec_hid_dim = 512
dropout = 0.5
clip = 1
log_interval = 100

if __name__ == '__main__':
    # 加载数据
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = CSL_Continuous(data_path=data_path, dict_path=dict_path,
        corpus_path=corpus_path, frames=sample_duration, train=True, transform=transform)
    val_set = CSL_Continuous(data_path=data_path, dict_path=dict_path,
        corpus_path=corpus_path, frames=sample_duration, train=False, transform=transform)
    # train_set = CSL_Continuous_Char(data_path=data_path, corpus_path=corpus_path,
    #     frames=sample_duration, train=True, transform=transform)
    # val_set = CSL_Continuous_Char(data_path=data_path, corpus_path=corpus_path,
    #     frames=sample_duration, train=False, transform=transform)
    logger.info("Dataset samples: {}".format(len(train_set)+len(val_set)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # Create Model
    encoder = Encoder(lstm_hidden_size=enc_hid_dim, arch="resnet18").to(device)
    decoder = Decoder(output_dim=train_set.output_dim, emb_dim=emb_dim, enc_hid_dim=enc_hid_dim, dec_hid_dim=dec_hid_dim, dropout=dropout).to(device)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Start training
    logger.info("Training Started".center(60, '#'))
    for epoch in range(epochs):
        # Train the model
        train_seq2seq(model, criterion, optimizer, clip, train_loader, device, epoch, logger, log_interval, writer)

        # Validate the model
        val_seq2seq(model, criterion, val_loader, device, epoch, logger, writer)

        # Save model
        torch.save(model.state_dict(), os.path.join(model_path, "slr_seq2seq_epoch{:03d}.pth".format(epoch+1)))#这句话放在epoch的循环体里面，好处就是中间断了，也可以保存没有训练完成的epoch的模型
        logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    logger.info("Training Finished".center(60, '#'))
