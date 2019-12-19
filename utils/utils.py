import os
import numpy as np
import torch
from torch.autograd import Variable


def predict(model, input1, input2, use_gpu):
    model.eval()
    if isinstance(input1, np.ndarray):
        input1 = torch.from_numpy(input1)
    if isinstance(input2, np.ndarray):
        input2 = torch.from_numpy(input2)
    input1 = input1.unsqueeze_(0)
    input2 = input2.unsqueeze_(0)
    if use_gpu:
        try:
            input1, input2 = input1.float(), input2.float()
            input1, input2 = Variable(input1.cuda()), Variable(input2.cuda())
        except Exception as e:
            print(e)
    else:
        input1, input2 = input1.float(), input2.float()
        input1, input2 = Variable(input1), Variable(input2)

    out1, out2 = model(input1, input2)
    return out1, out2


#evalute the result(MAE)
def evalute(gt, pre, n_factor):
    gt = np.array(gt)
    pre = np.array(pre)
    gt = gt.astype(np.float64)
    predictions = pre.astype(np.float64)

    gt = gt * float(n_factor)
    predictions = predictions * float(n_factor)

    error = []
    for i in range(len(gt)):
        error.append(abs(gt[i]-predictions[i]))
    mae = np.array(error).mean()
    print("Mean Absolute Error:", mae)

    return mae


def print_init_msg(data_dir, data_h5, output_dir, model_filename, train_model,
                   stage_training, learning_rate, batch_size, epochs, learning_rate_stage1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'best_models')):
        os.makedirs(os.path.join(output_dir, 'best_models'))

    print("data_dir: {}, data_h5: {}, output_dir: {}".format(data_dir, data_h5, output_dir))
    print("train_model: {}, stage_training: {}".format(train_model, stage_training))

    if stage_training == '1':
        print('learning_rate (stage_1):', learning_rate_stage1)
    elif stage_training == '2':
        print('learning_rate (stage_1 and 2):', learning_rate_stage1)

    print("learning_rate: {}, batch_size: {}, epochs: {}".format(learning_rate, batch_size, epochs))

    if stage_training == '1':
        filename_st1 = model_filename + '_stage_' + stage_training + '_st1-lr' + str(learning_rate) + '.pth'
        filename_st2 = 'null'
        print("Final filename (st1) = ", filename_st1 + '.pth')
    else:
        if stage_training == '2':
            filename_st2 = model_filename + '_stage_' + stage_training + '_st2-lr_' + str(learning_rate) + '.pth'
            filename_st1 = model_filename + '_stage_' + str(1) + '_st1-lr_' + str(learning_rate) + '.pth'
            print('Final (current) filename (st2) = ', filename_st2)
            print('Previout stage filename (st1) = ', filename_st1)
        else:
            print('WARNING: lr of stage 1 and 2 must be the same in this case (to-do: optimize code here)')
            filename_st2 = model_filename + '_stage_' + stage_training + '_st1-2-lr_' + str(
                learning_rate_stage1) + '_st3-lr_' + str(learning_rate) + '.pth'
            filename_st1 = model_filename + '_stage_' + str(2) + '_st1-lr_' + str(
                learning_rate_stage1) + '_st2-lr_' + str(learning_rate_stage1) + '.pth'
            print('Final (current) filename (st2) = ', filename_st2)
            print('Previout stage filename (st1) = ', filename_st1)
    return filename_st1, filename_st2


def calculate_val_real_mae(label, output, batch_sz):
    mae_value = 0.0
    for i in range(len(label)):
        mae_value += abs(label[i] - output[i])
    return mae_value[0] / batch_sz
