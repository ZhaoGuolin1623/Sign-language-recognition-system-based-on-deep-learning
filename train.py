import torch
from sklearn.metrics import accuracy_score
from tools import wer

def train_epoch(model, criterion, optimizer, dataloader, device, epoch, logger, log_interval, writer):
    model.train()
    losses = []
    all_label = []
    all_pred = []

    for batch_idx, data in enumerate(dataloader):
        # get the inputs and labels
        inputs, labels = data['data'].to(device), data['label'].to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        if isinstance(outputs, list):
            outputs = outputs[0]

        # compute the loss
        loss = criterion(outputs, labels.squeeze())
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]
        all_label.extend(labels.squeeze())
        all_pred.extend(prediction)
        score = accuracy_score(labels.squeeze().cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())

        # backward & optimize
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), score*100))

    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    # Log
    writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
    logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch+1, training_loss, training_acc*100))


def train_seq2seq(model, criterion, optimizer, clip, dataloader, device, epoch, logger, log_interval, writer):
    model.train()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    for batch_idx, (imgs, target) in enumerate(dataloader):
        imgs = imgs.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(imgs, target)#outputs是s2q模型的输出，和target不一样，含有隐含层元素。

        # target: (batch_size, trg len)
        # outputs: (trg_len, batch_size, output_dim)
        # skip sos
        #大致来看，outputs和target是首先都是把向量变为一维向量，然后criterion(outputs, target)交叉商函数搞一下。

        output_dim = outputs.shape[-1]
        outputs = outputs[1:].view(-1, output_dim)#这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成4列，那不确定的地方就可以写成-1
                                                    # 参考https://blog.csdn.net/scut_salmon/article/details/82391320

        target = target.permute(1,0)[1:].reshape(-1)#target.permute(1,0)是转置，[1:] 意思是去掉列表中第一个元素（下标为0），.reshape(-1)是搞成一维度向量，一串

        # compute the loss
        loss = criterion(outputs, target)
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]#数据形式是[数据条数n，503】，torch.max(outputs, 1)是沿着第二维度503这个找出最大值，就是找出最大可能的字典里的字。但是返回两个数据条数n长度数据，第一个是数据，
                                             # 第二个是索引，索引是我们感兴趣的，所以取第二维度，
        score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
                                                                #numpy.squeeze() 这个函数的作用是去掉矩阵里维度为1的维度，a.data.cpu().numpy()把tensor转换成numpy的格式
        all_trg.extend(target)
        all_pred.extend(prediction)

        # compute wer
        # prediction: ((trg_len-1)*batch_size)
        # target: ((trg_len-1)*batch_size)
        batch_size = imgs.shape[0]#前面parser部分定义batch_size=8，这里表示实际的，如果数据条目小于8，那么batch_size=实际条目，而不等于8
        prediction = prediction.view(-1, batch_size).permute(1,0).tolist()#.view(-1, batch_size)greshape成batch_size列，.permute(1,0)行列调换
        target = target.view(-1, batch_size).permute(1,0).tolist()
        wers = []
        for i in range(batch_size):
            # add mask(remove padding, sos, eos)
            prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
            target[i] = [item for item in target[i] if item not in [0,1,2]]
            wers.append(wer(target[i], prediction[i]))
        all_wer.extend(wers)

        # backward & optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}% | WER {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), score*100, sum(wers)/len(wers)))

    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    training_wer = sum(all_wer)/len(all_wer)
    # Log
    writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
    writer.add_scalars('WER', {'train': training_wer}, epoch+1)
    logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER {:.2f}%".format(epoch+1, training_loss, training_acc*100, training_wer))
