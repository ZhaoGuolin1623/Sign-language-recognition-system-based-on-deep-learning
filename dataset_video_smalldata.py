import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2



class CSL_Continuous(Dataset):
    def __init__(self, data_path, dict_path, corpus_path, frames=12, train=True, transform=None):
        super(CSL_Continuous, self).__init__()
        # 3个路径
        self.data_path = data_path
        self.dict_path = dict_path
        self.corpus_path = corpus_path
        # 帧数在读取图像时用到
        self.frames = frames#init的默认值是12，但是在main函数里传入的值是48，
        '''train_set = CSL_Continuous(data_path=data_path, dict_path=dict_path,
          corpus_path=corpus_path, frames=sample_duration, train=True, transform=transform)
          模式，变换'''
        self.train = train
        self.transform = transform
        # 其他参数
        self.num_sentences = 2#corups里有，视频是根据corups搞的
        # self.signers = 50#数据集里及个人打视频，越多模型越好。
        self.signers = 1#为了测试，就用1条视频去测试，
        self.repetition = 4#重复数目，每次喂batchsize个数据，里面可以有重复的，这里recetition加重了这种重复性。原始数据集是5，这里为了测试，拿出一个剩下4


        # 根据任务不同划分训练集测试集的大小，0.8*50*5=200，训练集每个句子对应200个样本
        # if self.train:
        #     self.videos_per_folder = int(0.8 * self.signers * self.repetition)
        # else:
        #     self.videos_per_folder = int(0.2 * self.signers * self.repetition)
        if self.train:
            self.videos_per_folder = int(0.75 * self.signers * self.repetition)
        else:
            self.videos_per_folder = int(0.25 * self.signers * self.repetition)

        self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        self.output_dim = 3
        try:
            dict_file = open(self.dict_path, 'r', encoding='utf-8')
            for line in dict_file.readlines():
                line = line.strip().split('\t')

                if '（' in line[1] and '）' in line[1]:
                    for delimeter in ['（', '）', '、']:
                        line[1] = line[1].replace(delimeter, " ")
                    words = line[1].split()
                else:
                    words = [line[1]]
                for word in words:
                    self.dict[word] = self.output_dim
                self.output_dim += 1
        except Exception as e:
            raise


        self.data_folder = []
        try:
            # 列出data_path下所有文件，obs_path包括所有item的路径
            obs_path = [os.path.join(self.data_path, item) for item in os.listdir(self.data_path)]
            self.data_folder = sorted([item for item in obs_path if os.path.isdir(item)])
        except Exception as e:
            raise
        # print(self.data_folder[1])
        # 就是000000-000099的目录，这里是\\，加了索引就变成了\

        # corpus
        self.corpus = {}
        self.unknown = set()
        try:
            corpus_file = open(self.corpus_path, 'r', encoding='utf-8')
            for line in corpus_file.readlines():
                line = line.strip().split()
                sentence = line[1]
                raw_sentence = (line[1]+'.')[:-1]
                paired = [False for i in range(len(line[1]))]
                # print(id(raw_sentence), id(line[1]), id(sentence))
                # pair long words with higher priority
                for token in sorted(self.dict, key=len, reverse=True):
                    index = raw_sentence.find(token)
                    # print(index, line[1])
                    if index != -1 and not paired[index]:
                        line[1] = line[1].replace(token, " "+token+" ")
                        # mark as paired
                        for i in range(len(token)):
                            paired[index+i] = True
                # add sos
                tokens = [self.dict['<sos>']]
                for token in line[1].split():
                    if token in self.dict:
                        tokens.append(self.dict[token])
                    else:
                        self.unknown.add(token)
                # add eos
                tokens.append(self.dict['<eos>'])
                self.corpus[line[0]] = tokens
        except Exception as e:
            raise

        # add padding
        length = [len(tokens) for key, tokens in self.corpus.items()]
        self.max_length = max(length)
        # print(max(length))
        for key, tokens in self.corpus.items():
            if len(tokens) < self.max_length:
                tokens.extend([self.dict['<pad>']]*(self.max_length-len(tokens)))
        # print(self.corpus)
        # print(self.unknown)

    def read_images(self, folder_path):
        # 在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃
        # assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder: " + str(folder_path)

        images = [] # list
        capture = cv2.VideoCapture(folder_path)#  VideoCapture()中参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开如cap = cv2.VideoCapture(“../test.avi”)

        fps = capture.get(cv2.CAP_PROP_FPS)#获取视频fps，
        fps_all = capture.get(cv2.CAP_PROP_FRAME_COUNT)#  获取视频总帧数。

        # 取整数部分
        timeF = int(fps_all/self.frames)
        n = 1

        # 对一个视频文件进行操作
        while capture.isOpened():#摄像头或者视频是否打开，windows下要用绝对路径
            ret, frame = capture.read()#cap.read()按帧读取视频，ret,frame是获cap.read()方法的两个返回值。
                                        # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，
                                        # 它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
                                        #frame是视频文件的一个帧，有可能全是0，但其他帧可能不是0，这里是正常的，并不代表整个视频是0。
            if ret is False:
                break
            # 每隔timeF帧进行存储操作
            if (n % timeF == 0):
                image = frame # frame是PIL
                image = Image.fromarray(image) # np array
                if self.transform is not None:
                    image = self.transform(image) # tensor
                images.append(image)
            n = n + 1
            # cv2.waitKey(1)
        capture.release()
        print('读取视频完成')
        # print("采样间隔：", timeF)

        lenB = len(images)
        # 将列表随机去除一部分元素，剩下的顺序不变

        for o in range(0, int(lenB-self.frames)):
            # 删除一个长度内随机索引对应的元素，不包括len(images)即不会超出索引
            del images[np.random.randint(0, len(images))]
            # images.pop(np.random.randint(0, len(images)))
        lenF = len(images)

        # 沿着一个新维度对输入张量序列进行连接，序列中所有的张量都应该为相同形状
        images = torch.stack(images, dim=0)
        # 原本是帧，通道，h，w，需要换成可供3D CNN使用的形状
        images = images.permute(1, 0, 2, 3)

        # print("数据类型：", images.dtype)
        # print("图像形状：", images.shape)
        # print("总帧数：%d, 采样后帧数：%d, 抽帧后帧数：%d" % (fps_all, lenB, lenF))

        return images

    def __len__(self):
        # 100*200=20000
        return self.num_sentences * self.videos_per_folder#决定了后面batchsize个索引的取值集，

    def __getitem__(self, idx):
        # 根据索引确定访问的文件夹，1000为第5个文件夹，就是obs_path中的某个
        # 新思路，索引就是样本，哪个样本就是哪个文件夹，在索引前面补充0至6位
        s = "%06d" % int(idx/self.videos_per_folder)
        top_folder = os.path.join(self.data_path, s)
        # top_folder = self.data_folder[int(idx/self.videos_per_folder)]
        # top_folder 'D:/Download/CSL_Continuous/color\\000005'
        # os.listdir 用于返回指定的文件夹包含的文件或文件夹的名字的列表

        # selected_folders就是文件夹内全部视频的路径
        selected_folders = [os.path.join(top_folder, item) for item in os.listdir(top_folder)]
        # sorted可以对所有可迭代的对象进行排序操作，但是结果表明此列表不可迭代
        # selected_folders = sorted([item for item in selected_folders_s if os.path.isdir(item)])
        #上面说不可迭代，这里我就让可以迭代
        selected_folders = sorted([item for item in selected_folders])

        # print(selected_folders)
        # 根据索引选定一个视频文件
        if self.train:
            selected_folder = selected_folders[idx%self.videos_per_folder]#idx从总的文件数目里选取，总的文件数目根据读取的文件数客观决定，这里是20000个，然后有多少个batch就有多少个idx，idx的选取规则由sample具体定，是随机的，
                                                                          #选取是从训练集选取，不包括验证集，这里的videos_per_folder也是每个文件夹里含有的验证集数据个数，因为idx也是以它为计算机基础的，所以去余之后还会i
                                                                          #保证idx%self.videos_per_folder为所在文件夹的顺序，同时也是文件加名字，代表第“idx%self.videos_per_folder”句话。
        else:
            selected_folder = selected_folders[idx%self.videos_per_folder + int(0.75*self.signers*self.repetition)]#videos_per_folder此时是每个文件夹里验证集的元素个数。
        # 给定文件夹（索引类别）进行读取，其中250个视频（否）
        images = self.read_images(selected_folder)

        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.corpus['{:06d}'.format(int(idx/self.videos_per_folder))])
        tokens = torch.LongTensor(self.corpus['{:06d}'.format(int(idx/self.videos_per_folder))])
        len_label = len(tokens)

        dict_file = open(self.dict_path, 'r', encoding='utf-8')
        len_voc = len(dict_file.readlines()) + 2
                                                    # dictionary
                                                    # self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
                                                    # self.output_dim = 3

        # print("标签长度：%d 词典长度: %d" % (len_label, len_voc))

        # return images, tokens, len_label, len_voc
        return images, tokens
