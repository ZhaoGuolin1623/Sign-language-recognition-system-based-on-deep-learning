
#修改以适合连续数据集，视频，只测试视频，不测试孤立的数据
#可以选择网络或者训练时的模型，也可以选择选练后参数权重的模型

import torch
from sklearn.metrics import accuracy_score
from models.Seq2Seq import Encoder, Decoder, Seq2Seq
import torch.nn as nn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tools import wer
import logging

# Path setting
data_path = './dataset/color_small_test/' #测试时局即路径， 因为collect_number_of_subfolers的num_dir(target_path)的目录target_pat最后一个字符需要是/
dict_path = './dictionary.txt'
corpus_path = './corpus.txt'

log_path = "log/test_seq2seq_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
sum_path = "runs/test_slr_seq2seq_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# 同时为了测试test，有一些test自有的按数需要单独设置
########################################################################################################################################
#训练时，只需要改两个##号行之间的参数即可。其他的一般不动
videos_per_folder = 10 #这个需要录好视频直接放在测试数据集里，查好数目即可。
test_batch_size =1  #测试时的该参数不需要一定和训练时一样，虽然最好一样，但是因为二者数据集大小不一样，当test数据集比train数据集小很多时，可能需要调整
#########################################################################################################################################




#为了保持测试模型和训练模型的参数完全一致，同时增加程序的可扩展性，这里用import把训练时的模型高嵯来，到时只需要修改训练时的模型参数就可以了，会自动同步。
from CSL_Continuous_Seq2Seq_smalldata import enc_hid_dim
from CSL_Continuous_Seq2Seq_smalldata import emb_dim
from CSL_Continuous_Seq2Seq_smalldata import dec_hid_dim
from CSL_Continuous_Seq2Seq_smalldata import dropout
###################################################


# 字典逆向查找（reverse lookup），根据值 value 从字典中查找并返回其 key， 参考https://blog.csdn.net/weixin_58123489/article/details/124053683
def lookup(look, get_value_list):

    # words = []
    words = list()
    look = look


    for get_value in get_value_list:

        if get_value in look.values():
                word = list(look.keys())[list(look.values()).index(get_value)]
                if word not in ['<pad>', '<sos>', '<eos>']:
                   words.append(word)

    return words



# #因为模型不一样，所以这里的test函数没有用，直接用的是validation.py里面的val_seq2seq函数进行改编。
# def test(model, criterion, dataloader, device, epoch, logger, writer):
if __name__ == '__main__':
    import os
    import argparse
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from dataset_video_smalldata_test import CSL_Continuous
    from models.Conv3D import resnet18, resnet34, resnet50, r2plus1d_18

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default= data_path,
        type=str, help='Data path for testing')
    parser.add_argument('--label_path', default=dict_path,
        type=str, help='Label path for testing')
    # parser.add_argument('--model', default='3dresnet18',
    #     type=str, help='Choose a model for testing')#model_path = "Data/seq2seq_models"
    # parser.add_argument('--model_path', default='3dresnet18.pth',
    #     type=str, help='Model state dict path')

    #替换video的那个模型
    parser.add_argument('--model', default='Seq2Seq',
        type=str, help='Choose a model for testing')#model_path = "Data/seq2seq_models"
    parser.add_argument('--model_path', default='./Data/seq2seq_models/slr_seq2seq_epoch050.pth',
                        type=str, help='Model state dict path')

    parser.add_argument('--num_classes', default=500,
        type=int, help='Number of classes for testing')
    parser.add_argument('--batch_size', default=test_batch_size,
        type=int, help='Batch size for testing')#这个地方要根据测试数据集情况合理设置，不然影响结果，比如，只有一个样本时，设置为32那精确度就是100%，而设置为1是就是58.3%左右。
    parser.add_argument('--sample_size', default=128,
        type=int, help='Sample size for testing')
    parser.add_argument('--sample_duration', default=16,
        type=int, help='Sample duration for testing')
    parser.add_argument('--no_cuda', action='store_true',
        help='If true, dont use cuda')
    parser.add_argument('--cuda_devices', default='2',
        type=str, help='Cuda visible devices')
    args = parser.parse_args()

    # Path setting
    data_path = args.data_path
    label_path = args.label_path
    model_path = args.model_path
    # Use specific gpus
    os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_devices
    # Device setting
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Hyperparams
    num_classes = args.num_classes
    batch_size = args.batch_size
    sample_size = args.sample_size
    sample_duration = args.sample_duration

    # Start testing
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    #要根据新的数据集处理函数CSL_Continuous修改
    test_set = CSL_Continuous(data_path=data_path, dict_path=dict_path, corpus_path=corpus_path, frames=sample_duration, train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)



    # Create model
    if args.model == '3dresnet18':
        model = resnet18(pretrained=True, progress=True, sample_size=sample_size,
            sample_duration=sample_duration, num_classes=num_classes).to(device)
    elif args.model == '3dresnet34':
        model = resnet34(pretrained=True, progress=True, sample_size=sample_size,
            sample_duration=sample_duration, num_classes=num_classes).to(device)
    elif args.model == '3dresnet50':
        model = resnet50(pretrained=True, progress=True, sample_size=sample_size,
            sample_duration=sample_duration, num_classes=num_classes).to(device)
    elif args.model == 'r2plus1d':
        model = r2plus1d_18(pretrained=True, num_classes=num_classes).to(device)
    # 加上一行替换，因为一般会加载本地训练的模型文件路径，所以这里一半也用不上，对程序影响不大，不过要是想用原始的训练时的model，而不是训练后更新权重的pth文件模型，这个就起作用了。但是参数使用
    # 和上面几个网络下载函数不太一样，这里就不用了，直接加载训练后的模型即可。
    elif args.model == 'Seq2Seq':
        encoder = Encoder(lstm_hidden_size=enc_hid_dim, arch="resnet18").to(device)#需要先将Encoder和Decoder实例化，然后传入，为和训练时的模型保持一致，这里需要从CSL_Continuous_Seq2Seq_smalldata.py里超参数拷贝过来放到程序的前面。
        # decoder = Decoder(output_dim=train_set.output_dim, emb_dim=emb_dim, enc_hid_dim=enc_hid_dim, dec_hid_dim=dec_hid_dim, dropout=dropout).to(device)
        # 原来函数写的“output_dim=train_set.output_dim”，这里直接修改为503，就是字典的行数。测试模型要和训练模型完全一致。
        decoder = Decoder(output_dim=503, emb_dim=emb_dim, enc_hid_dim=enc_hid_dim, dec_hid_dim=dec_hid_dim, dropout=dropout).to(device)
        model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)

    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # Load model
    #这里实现的很有技巧，因为load这句话在上面网络加在模型语句的下面，所以如果不写入实际的本地模型，那就加在的是网络模型地址，还是网络模型，网络模型下载之后，这里又重新加载一遍，下载后的模型一版来说存在.cache文件下下，

    model.load_state_dict(torch.load(model_path))#这一句最重要，把训练权重加载进来了。


    # 用validation.py里的来替代。详细的注释请看train.py相应的注释，那里精确率的计算和这里异曲同工，train_seq2seq，比较难的就是对s2q模型输出的理解，但输出最终抛弃掉一些就是【数据条目，字典里字个数】
    # def val_seq2seq(model, criterion, dataloader, device, epoch, logger, writer):
    # Log to file & tensorboard writer
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger('SLR')
    logger.info('Logging to file...')
    writer = SummaryWriter(sum_path)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []


    with torch.no_grad():
        for batch_idx, (imgs, target) in enumerate(test_loader):
            # imgs = imgs.to(device)#images的参数【*，*，*，*，*】，【batchsize，通道数，帧数，像素长，像素宽】

            # ba字典和语料库字典返回，用于反向操作，好把可读的字搞出来，好显示。，所以在dataset里返回的是 return images, self.dict, self.corpus，相应的也得改一下。
            dict = imgs[1]
            corpus = imgs[2]
            # lookup = lookup(dict, get_value)
            imgs = imgs[0].to(device)  # images的参数【*，*，*，*，*】，【batchsize，通道数，帧数，像素长，像素宽】


            target = target.to(device)#该视频在corups里对应的话，是话的编码

            # forward(no teacher forcing)
            outputs = model(imgs, target, 0)#valid为什么还需要传入target呢？还需要反向传播吗？除了验证准确率外，还做了设呢吗影响参数的事情？参看train文件里的注释，这里是s2q文件的需要。
            # outputs = model(imgs)

            # target: (batch_size, trg len)
            # outputs: (trg_len, batch_size, output_dim)
            # skip sos
            output_dim = outputs.shape[-1]
            outputs = outputs[1:].view(-1, output_dim)# 【1：】表示第一个元素去掉，可能是设呢吗开始标记符号，看一下前面的代码就知道了，估计是，懒得看了。
            target = target.permute(1, 0)[1:].reshape(-1)#target.permute(1,0)是转置，比如转置前形状是【1，7】，转置之后形状是【7，1】，并不是元素的次序反过来。
                                                         # [1:] 意思是去掉列表中第一个元素（下标为0），.为什么这样做呢？ 可能涉及到编码，可能是开始符号？ reshape(-1)是搞成一维度向量，一串，如果batchsize=1，那就没什么变化


            # compute the loss
            loss = criterion(outputs, target)
            losses.append(loss.item())

            # compute the accuracy
            prediction = torch.max(outputs, 1)[1]  # 数据形式是[数据条数n，503】，     torch.max(outputs, 1)是沿着第二维度503这个找出最大值，就是找出最大可能的字典里的字。但是返回两个数据，第一个是数据，
                                                   # 第二个是索引，这里的索引就是数字的位置，这里索引就是前面的词的编码，索引是我们感兴趣的，所以取第二维度，用[1]表示。  # https://blog.csdn.net/Caesar6666/article/details/121900138
                                                   # 这样预测的话，那映射不是100个句子，而是具体的字，那就有扩展性了，但是准吗？这里想办法把字体显示出来，就知道了。

            # lookup(dict, get_value)
            words_predict = lookup(dict, prediction)
            print (words_predict)

            score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())  # numpy.squeeze() 这个函数的作用是去掉矩阵里维度为1的维度，a.data.cpu().numpy()把tensor转换成numpy的格式http://t.zoukankan.com/limingqi-p-11729572.html
            all_trg.extend(target)
            all_pred.extend(prediction)

            # 计算 损失值
            # prediction: ((trg_len-1)*batch_size)
            # target: ((trg_len-1)*batch_size)
            batch_size = imgs.shape[0]
            prediction = prediction.view(-1, batch_size).permute(1, 0).tolist()
            target = target.view(-1, batch_size).permute(1, 0).tolist()
            wers = []
            for i in range(batch_size):
                # add mask(remove padding, eos, sos)
                prediction[i] = [item for item in prediction[i] if item not in [0, 1, 2]]
                target[i] = [item for item in target[i] if item not in [0, 1, 2]]
                wers.append(wer(target[i], prediction[i]))
            all_wer.extend(wers)

    # 计算平均损失值和准确率
    validation_loss = sum(losses) / len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    test_small_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    validation_wer = sum(all_wer) / len(all_wer)
    print("Test Acc: {:.2f}%".format(test_small_acc*100))
    # Log
    writer.add_scalars('Loss', {'test_small': validation_loss})
    writer.add_scalars('Accuracy', {'test_small': test_small_acc})
    writer.add_scalars('WER', {'test_small': validation_wer})
    logger.info("Average test_small Loss: {:.6f} | Acc: {:.2f}% | WER: {:.2f}%".format(validation_loss, test_small_acc * 100, validation_wer))


