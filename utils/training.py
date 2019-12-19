import torch
from torch.autograd import Variable
import numpy as np
import os
from utils.utils import calculate_val_real_mae


def train(train_loader, valid_loader, model, criterion, optimizer, epochs, learning_rate, stage_training, use_gpu,
          batch_size, display_interval, best_mae, best_epoch, stage_1_model, stage_2_model, model_path):
    running_loss = 0.0
    model_path = os.path.join(model_path, 'best_models')
    if stage_training == '1':
        for k, param in enumerate(model.modules()): #model.modules): #model.module.features.parameters()
            if param.features:
                for i, param_inner in enumerate(param.features.parameters()):
                    param_inner.requires_grad = False
                break
        model.train()
        for epoch in range(epochs):
            for i, (input1, real_label, app_label, input2) in enumerate(train_loader):
                if use_gpu:
                    try:
                        real_label = real_label.float()
                        app_label = app_label.float()
                        input1, input2 = input1.float(), input2.float()
                        input1, real_label, app_label, input2 = Variable(input1.cuda()), Variable(real_label.cuda()), Variable(app_label.cuda()), Variable(input2.cuda())
                    except Exception as e:
                        print(e)
                else:
                    input1, real_label, app_label, input2 = Variable(input1), Variable(real_label), Variable(app_label), Variable(input2)

                output1, output2 = model(input1, input2)
                loss1 = criterion(app_label, torch.squeeze(output1, 1))
                loss2 = criterion(real_label, torch.squeeze(output2, 1))

                loss = loss1 + loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                try:
                    running_loss += loss.data / batch_size
                except Exception as e:
                    print(e)

                if (i+1) % display_interval == 0:
                    msg = 'epoch:{}, batch: {}, lr: {}, loss: {:.6f}'.format(epoch+1, (i+1), learning_rate, running_loss / (i+1))
                    print(msg)

            best_mae, statu = valid(valid_loader, model, best_mae, best_epoch, epoch, criterion, use_gpu, batch_size, model_path, stage_1_model)
            if statu == 'stop':
                break
    elif stage_training == '2':
        checkpoint = torch.load(os.path.join(model_path, stage_1_model))
        model.load_state_dict(checkpoint)
        model.train()
        for epoch in range(epochs):
            for i, (input1, real_label, app_label, input2) in enumerate(train_loader):
                if use_gpu:
                    try:
                        real_label = real_label.float()
                        app_label = app_label.float()
                        input1, input2 = input1.float(), input2.float()
                        input1, real_label, app_label, input2 = Variable(input1.cuda()), Variable(real_label.cuda()), Variable(app_label.cuda()), Variable(input2.cuda())
                    except Exception as e:
                        print(e)
                else:
                    input1, real_label, app_label, input2 = Variable(input1), Variable(real_label), Variable(app_label), Variable(input2)

                output1, output2 = model(input1, input2)

                loss1 = criterion(app_label, torch.squeeze(output1, 1))
                loss2 = criterion(real_label, torch.squeeze(output2, 1))

                loss = loss1 + loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                try:
                    running_loss += loss.data / batch_size
                except Exception as e:
                    print(e)

                if (i + 1) % display_interval == 0:
                    msg = 'epoch:{}, batch: {}, lr: {}, loss: {:.6f}'.format(epoch + 1, (i + 1), learning_rate,
                                                                             running_loss / (i + 1))
                    print(msg)
            best_mae, status = valid(valid_loader, model, best_mae, best_epoch, epoch, criterion, use_gpu, batch_size, model_path, stage_2_model)
            if status == 'stop':
                break


def valid(valid_loader, model, best_mae, best_epoch, cur_epoch, criterion, use_gpu, batch_size, model_path, best_model):
    running_loss = 0.0
    model.eval()
    output_app = []
    output_real = []
    label_app = []
    label_real = []
    print("valid is starting......")
    for i, (input1, real_label, app_label, input2) in enumerate(valid_loader):
        if use_gpu:
            try:
                real_label = real_label.float()
                app_label = app_label.float()
                input1, input2 = input1.float(), input2.float()
                input1, real_label, app_label, input2 = Variable(input1.cuda()), Variable(real_label.cuda()), Variable(app_label.cuda()), Variable(input2.cuda())
            except Exception as e:
                print(e)
        else:
            input1, real_label, app_label, input2 = Variable(input1), Variable(real_label), Variable(
                app_label), Variable(input2)

        output1, output2 = model(input1, input2)

        loss1 = criterion(app_label, torch.squeeze(output1))
        loss2 = criterion(real_label, torch.squeeze(output2))

        loss = loss1 + loss2
        try:
            running_loss += loss.data
            if use_gpu:
                output1_cpu = output1.cpu().detach().numpy()
                output2_cpu = output2.cpu().detach().numpy()
                real_label = real_label.cpu().detach().numpy()
                app_label = app_label.cpu().detach().numpy()
            else:
                output1_cpu = output1
                output2_cpu = output2
                real_label = real_label
                app_label = app_label
            output_app.extend(output1_cpu)
            output_real.extend(output2_cpu)
            label_app.extend(real_label)
            label_real.extend(app_label)
        except Exception as e:
            print(e)

    output_real_arr = np.asarray(output_real)
    label_real_arr = np.asarray(label_real)

    val_mae = calculate_val_real_mae(label_real_arr, output_real_arr, batch_size)
    print("The current MAE for valid date is: {}".format(val_mae))
    if cur_epoch == 0:
        best_mae = val_mae
    elif val_mae < best_mae:
        best_mae = val_mae
        best_epoch = cur_epoch
        print("saving best valid model...")
        torch.save(model.state_dict(), os.path.join(model_path, best_model))
        print("new best valid mae: {:.6f}, current epoch is: {}".format(best_mae, best_epoch))
        print("Best model path is:{}".format(os.path.join(model_path, best_model)))
    else:
        pass
    #     stop_epoch = best_epoch + 100
    #     if cur_epoch == stop_epoch:
    #         return best_mae, 'stop'
    # print("The best mae is: {}".format(best_mae))
    # print(best_epoch)
    return best_mae, 'continue'



