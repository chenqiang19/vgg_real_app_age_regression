import sys
import os
import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torchvision import models, transforms
import torch.nn.parallel
from torch.utils.data import DataLoader
from network import vgg16_real_app

from data_process import load_data
from utils import training, utils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_model():
    vgg16 = models.vgg16_bn(pretrained=True)
    vgg16_age = vgg16_real_app.vgg16_bn()

    pretrained_dict = vgg16.state_dict()
    model_dict = vgg16_age.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    vgg16_age.load_state_dict(model_dict)

    return vgg16_age


def main(data_h5, output_dir, train_model, stage_training, filename_st1, filename_st2,
         learning_rate, batch_size, epochs, optim_value, use_gpu):
    #if train_model is True start training
    if train_model:
        model = load_model()
        train_set, train_real_labels, train_app_labels = load_data.load_data_set(data_h5, 'train')
        valid_set, valid_real_labels, valid_app_labels = load_data.load_data_set(data_h5, 'valid')

        train_all_extra_labels = load_data.load_data_extra_set(data_h5, 'train')
        valid_all_extra_labels = load_data.load_data_extra_set(data_h5, 'valid')

        train_data = load_data.dataSet(train_set, train_real_labels, train_app_labels, train_all_extra_labels)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)

        valid_data = load_data.dataSet(valid_set, valid_real_labels, valid_app_labels, valid_all_extra_labels)
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size//2, num_workers=4)
        criterion = nn.MSELoss()

        if use_gpu:
            cudnn.benchmark = True
            torch.cuda.set_device(0)
            model.cuda()
            criterion.cuda()

        if optim_value == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        elif optim_value == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=[0.9, 0.999], eps=1e-08, weight_decay=0.0, amsgrad=False)
        else:
            print("optim paramter is not set.")

        display_interval = 1

        best_valid_auc = 0.0
        best_valid_epoch = 0
        training.train(train_loader, valid_loader, model, criterion, optimizer, epochs, learning_rate,
                              stage_training, use_gpu, batch_size, display_interval, best_valid_auc,
                              best_valid_epoch, filename_st1, filename_st2, output_dir)

    else:
        print("Load pre-trained model (and predicting ages.)")
        print(">> Warning-model generated from stage:", stage_training)
        print(">> Results shown in the paper are based on 2 stages training")
        model = vgg16_real_app.vgg16_bn()
        if stage_training == '1':
            checkpoint = torch.load(os.path.join(output_dir, 'best_models', filename_st1))
        else:
            checkpoint = torch.load(os.path.join(output_dir, 'best_models', filename_st2))

        test_set, test_real_labels, test_app_labels = load_data.load_data_set(data_h5, 'test')
        test_all_extra_labels = load_data.load_data_extra_set(data_h5, 'test')
        cudnn.benchmark = True
        model.load_state_dict(checkpoint)
        model.cuda()
        pred_out1 = []
        pred_out2 = []
        real_label = []
        app_label = []

        test_sets = np.transpose(test_set, (0, 3, 1, 2))
        for data, real, app, extra in zip(test_sets, test_real_labels, test_app_labels, test_all_extra_labels):
            output1, output2 = utils.predict(model, data, extra, use_gpu)
            pred_out1.extend(output1.cpu().detach().numpy()[0])
            pred_out2.extend(output2.cpu().detach().numpy()[0])
            real_label.append(float(real))
            app_label.append(float(app))
        mae_app = utils.evalute(app_label, pred_out1, 100)
        mae_real = utils.evalute(real_label, pred_out2, 100)
        print("The test mae_app is: {}, mae_real is: {}.".format(mae_app, mae_real))


if __name__ == "__main__":
    data_dir = sys.argv[1]
    if sys.argv[2] == 'True':
        train_model = True
    else:
        train_model = False

    stage_training = sys.argv[3]
    learning_rate = float(sys.argv[4])
    batch_size = int(sys.argv[5])
    epochs = int(sys.argv[6])

    learning_rate_stage1 = float(sys.argv[7])

    optim_set = sys.argv[8]

    use_gpu = sys.argv[9]
    data_h5 = os.path.join(data_dir, 'data_h5')
    output_dir = os.path.join(data_dir, 'output/')
    model_filename = 'vgg16_app_real_age_fg2019'

    new_filename_st1, new_filename_st2 = utils.print_init_msg(data_dir, data_h5, output_dir, model_filename, train_model,
                                                        stage_training, learning_rate, batch_size, epochs,
                                                        learning_rate_stage1)

    main(data_h5, output_dir, train_model, stage_training, new_filename_st1, new_filename_st2, learning_rate,
         batch_size, epochs, optim_set, use_gpu)

