
# -*- coding: utf-8 -*-
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim

import JackFramework as jf
import sys
# import UserModelImplementation.user_define as user_def
from .model import ColorModel, DepthModel, GeneratorModel, NLayerDiscriminator
import torch.nn.functional as F
import torch.nn as nn
from .submodel import UNet 
import numpy as np
import ops

class BodyReconstructionInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for BodyReconstructionInterface"""

    MODEL_COLOR_ID = 0  # only color_net
    MODEL_DEPTH_ID = 1  # only depth net
    MODEL_GENERATOR_ID = 0
    MODEL_COLOR_DISC_ID = 2 # only discriminator for color
    MODEL_DEPTH_DISC_ID = 3 # only discriminator for depth
    COLOR_LABEL_ID = 0  # with 2 label
    DEPTH_LABLE_ID = 1
    COLOR_ID = 0
    DEPTH_ID = 1
    UV_ID = 2

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.__criterion = nn.L1Loss()
        self.__CrossEntropy = nn.CrossEntropyLoss()
        self.__mse = nn.MSELoss()
        self.output_G_color = None
        self.output_G_normal = None
        self.model_color = None
        self.model_depth = None
        self.disc_color = None
        self.disc_depth = None

    @staticmethod
    def lr_lambda(epoch: int) -> float:
        max_warm_up_epoch = 10
        convert_epoch = 50
        off_set = 1
        lr_factor = 1.0

        factor = ((epoch + off_set) / max_warm_up_epoch) if epoch < max_warm_up_epoch \
            else lr_factor if (epoch >= max_warm_up_epoch and epoch < convert_epoch) \
            else lr_factor * 0.25
        return factor

    def get_model(self) -> list:
        args = self.__args
        #self.__lr = args.lr
        # return model
        ngf = 64
        self.model_color = ColorModel(ngf=ngf, mask=args.mask)
        self.model_depth = DepthModel(ngf=ngf, mask=args.mask)
        #self.generator = GeneratorModel(in_channel=7, ngf=ngf, mask=args.mask)
        self.disc_color = NLayerDiscriminator(input_nc=3, ndf=32, n_layers=3)
        self.disc_depth = NLayerDiscriminator(input_nc=3, ndf=32, n_layers=3)
        return [self.model_color, self.model_depth, self.disc_color, self.disc_depth]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        opt_color = optim.Adam(model[0].parameters(), lr=lr)
        if args.lr_scheduler:
            sch_color = optim.lr_scheduler.LambdaLR(opt_color, lr_lambda=self.lr_lambda)
        else:
            sch_color = None
        
        opt_depth = optim.Adam(model[1].parameters(), lr=lr)
        if args.lr_scheduler:
            sch_depth = optim.lr_scheduler.LambdaLR(opt_depth, lr_lambda=self.lr_lambda)
        else:
            sch_depth = None
       
        opt_color_disc = optim.Adam(model[2].parameters(), lr=0.00001*lr)    
        if args.lr_scheduler:
            sch_color_disc = optim.lr_scheduler.LambdaLR(opt_color_disc, lr_lambda=self.lr_lambda)
        else:
            sch_color_disc = None

        opt_depth_disc = optim.Adam(model[3].parameters(), lr=0.00001*lr)    
        if args.lr_scheduler:
            sch_depth_disc = optim.lr_scheduler.LambdaLR(opt_depth_disc, lr_lambda=self.lr_lambda)
        else:
            sch_depth_disc = None

        return [opt_color, opt_depth, opt_color_disc, opt_depth_disc], \
                [sch_color, sch_depth, sch_color_disc, sch_depth_disc]
        

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        if self.MODEL_COLOR_ID == sch_id:
            sch.step()
        if self.MODEL_DEPTH_ID == sch_id:
            sch.step()
        if self.MODEL_COLOR_DISC_ID == sch_id:
            sch.step()
        if self.MODEL_DEPTH_DISC_ID == sch_id:
            sch.step()

    def inference(self, model: object, input_data: list, model_id: int) -> list:
        args = self.__args
        '''
        if self.MODEL_GENERATOR_ID == model_id:
            fake_images = model(input_data[self.COLOR_ID], 
                                            input_data[self.DEPTH_ID], 
                                            input_data[self.UV_ID])
            normal_pre = ops.depth_to_normal(fake_images[1])
            if args.mode == "train":
                assert self.output_G_color is None and self.output_G_depth is None
                self.output_G_color = fake_images[0].detach()
                self.output_G_depth= fake_images[1].detach()
                with torch.no_grad():
                    fake_prob_color = self.disc_color(self.output_G_color)
                    fake_prob_depth = self.disc_depth(self.output_G_depth)
                    #if args.mask:
                    #    return [fake_images[0], fake_images[1], normal_pre, fake_prob_color, fake_prob_depth, fake_images[2]]
                    return [fake_images[0], fake_images[1], normal_pre, fake_prob_color, fake_prob_depth] 
            
            return [fake_images[0], fake_images[1], normal_pre, input_data[self.COLOR_ID], input_data[self.DEPTH_ID]]
            '''
        if self.MODEL_COLOR_ID == model_id:

            
            if args.mode == "train":  
                fake_images = model(input_data[self.COLOR_ID], 
                                            input_data[6], 
                                            input_data[self.UV_ID])
                assert self.output_G_color is None
                self.output_G_color = fake_images[0].detach()
                with torch.no_grad():
                    fake_prob_color = self.disc_color(self.output_G_color) 
                    if args.mask:
                        #print(fake_images[0].shape, fake_prob_color.shape)
                        return [fake_images[0], fake_prob_color, fake_images[1]] 
                        
                    return [fake_images[0], fake_prob_color] 
            else:
                fake_images = model(input_data[self.COLOR_ID], 
                                            input_data[self.DEPTH_ID], 
                                            input_data[self.UV_ID])
                return [fake_images[0], input_data[self.COLOR_ID]]

        if self.MODEL_DEPTH_ID == model_id:
            if args.mode == "train":
                fake_images = model(input_data[5], 
                                            input_data[self.DEPTH_ID], 
                                            input_data[self.UV_ID])
                normal_pre = ops.depth_to_normal(fake_images[0])
                assert self.output_G_normal is None
                self.output_G_normal = normal_pre.detach().cuda()
                with torch.no_grad():
                    fake_prob = self.disc_depth(self.output_G_normal)
                     
                    if args.mask:
                        return [fake_images[0], normal_pre, fake_prob, fake_images[1]]
                    return [fake_images[0], normal_pre, fake_prob]
            else:
                depth_front = model(input_data[self.COLOR_ID], 
                                            input_data[self.DEPTH_ID], 
                                            input_data[self.UV_ID])
                normal_pre = ops.depth_to_normal(depth_front[0])
                return [depth_front[0], normal_pre, input_data[self.DEPTH_ID]]
        
        if args.mode == "train":
            if self.MODEL_COLOR_DISC_ID == model_id:
                assert self.output_G_color is not None
                disc_color_fack = model(self.output_G_color)
                disc_color_true = model(input_data[3])
                self.output_G_color = None
                return [disc_color_fack, disc_color_true]

        if args.mode == "train":
            if self.MODEL_DEPTH_DISC_ID == model_id:
                assert self.output_G_normal is not None
                disc_depth_fack = model(self.output_G_normal)               
                normal_gt = ops.depth_to_normal(input_data[4])
                disc_depth_true = model(normal_gt.detach())
                self.output_G_normal = None
                return [disc_depth_fack, disc_depth_true]




    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc 
        # args = self.__args
        acc_0 = None
        acc_1 = None
        if self.MODEL_COLOR_ID == model_id:
            acc_0 = jf.acc.BaseAccuracy.rmse_score(output_data[0], label_data[0])
            #acc_1 = jf.acc.BaseAccuracy.rmse_score(output_data[1], label_data[1])
            return[acc_0]
            #return [acc_0, acc_1]
        if self.MODEL_DEPTH_ID == model_id:
            acc_1 = jf.acc.BaseAccuracy.rmse_score(output_data[0], label_data[1])
            return [acc_1]
        if self.MODEL_COLOR_DISC_ID == model_id:
            acc_fack = torch.mean(torch.sigmoid(output_data[0]))
            acc_real = torch.mean(torch.sigmoid(output_data[1]))
            return [acc_fack, acc_real]
        if self.MODEL_DEPTH_DISC_ID == model_id:
            acc_fack = torch.mean(torch.sigmoid(output_data[0]))
            acc_real = torch.mean(torch.sigmoid(output_data[1]))
            return [acc_fack, acc_real]

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        # args = self.__args
        #total_loss = None
        #mask = (label_data[0]>0)
        #print(label_data[1].shape)
        #mask = np.expand_dims(color_gt.cpu(), axis=0)
        #print(mask.shape)

        args = self.__args
        '''
        if self.MODEL_GENERATOR_ID == model_id:
            loss_color_gan = nn.functional.binary_cross_entropy(torch.sigmoid(output_data[3]), 
                                                                torch.ones_like(output_data[3]).cuda())
            loss_color = torch.mean(torch.abs(output_data[self.COLOR_ID]-label_data[self.COLOR_LABEL_ID]))
            loss_depth = torch.mean(torch.abs(output_data[self.DEPTH_ID]-label_data[self.DEPTH_LABLE_ID]))
            loss_depth_gan = nn.functional.binary_cross_entropy(torch.sigmoid(output_data[4]), 
                                                                torch.ones_like(output_data[4]).cuda())
            normal_gt = ops.depth_to_normal(label_data[self.DEPTH_LABLE_ID])
            loss_normal = torch.mean(torch.abs(output_data[2]-normal_gt))
            loss_total = 100 * (loss_color + loss_depth + loss_normal) + loss_color_gan + loss_depth_gan
            return [loss_total, loss_color, loss_depth]
        '''
        if self.MODEL_COLOR_ID == model_id:
            loss_color_gan = nn.functional.binary_cross_entropy(torch.sigmoid(output_data[1]), 
                                                                torch.ones_like(output_data[1]).cuda())
            loss_color = torch.mean(torch.abs(output_data[0]-label_data[0]))
            loss_mask = torch.mean(torch.abs(output_data[2]-label_data[2])) if args.mask else 0
            loss_total = 15 * loss_color + 0.25*loss_mask + loss_color_gan 
            return [loss_total, loss_color]
        if self.MODEL_DEPTH_ID == model_id:
            loss_depth = torch.mean(torch.abs(output_data[0]-label_data[1]))
            
            loss_normal_gan = nn.functional.binary_cross_entropy(torch.sigmoid(output_data[2]), 
                                                                torch.ones_like(output_data[2]).cuda())
            #loss_depth_gan = nn.functional.binary_cross_entropy(torch.sigmoid(output_data[1]), 
            #                                                    torch.ones_like(output_data[1]).cuda())
            normal_gt = ops.depth_to_normal(label_data[1])
            loss_normal = torch.mean(torch.abs(output_data[1]-normal_gt))
            #total loss
            loss_mask = torch.mean(torch.abs(output_data[3]-label_data[3])) if args.mask else 0
            total_depth_loss = 10*(loss_depth + loss_normal) + 0.5*loss_mask + loss_normal_gan
            #total_depth_loss = 20*loss_depth + loss_normal + 0.5*loss_mask + loss_depth_gan
            return [total_depth_loss, loss_depth]
        if self.MODEL_COLOR_DISC_ID == model_id:
            #b, _, h, w = output_data[0].shape
            #print(output_data[0].shape)
            #label_fake = torch.zeros([b, h, w]).cuda().long()
            #label_real = torch.ones([b, h, w]).cuda().long()
            #loss_disc_color_fake = self.__CrossEntropy(output_data[0], label_fake)
            #loss_disc_color_real = self.__CrossEntropy(output_data[1], label_real)
            #loss_disc_color_real = self.__CrossEntropy(output_data[1], 
            #                                torch.ones_like(output_data[1]).type(torch.cuda.LongTensor))
            out_fake = torch.sigmoid(output_data[0])
            out_real = torch.sigmoid(output_data[1])
            all0 = torch.zeros_like(out_fake).cuda()
            all1 = torch.ones_like(out_real).cuda()
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_disc = ad_fake_loss + ad_true_loss
            
            return[loss_disc]
        if self.MODEL_DEPTH_DISC_ID == model_id:
            out_fake = torch.sigmoid(output_data[0])
            out_real = torch.sigmoid(output_data[1])
            all0 = torch.zeros_like(out_fake).cuda()
            all1 = torch.ones_like(out_real).cuda()
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_disc = ad_fake_loss + ad_true_loss
            return[loss_disc]


    # Optional
    def pretreatment(self, epoch: int, rank: object) -> None:
        # do something before training epoch
        pass
             
    # Optional
    def postprocess(self, epoch: int, rank: object,
                    ave_tower_loss: list, ave_tower_acc: list) -> None:
        # do something after training epoch
        pass

    # Optional
    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        return None
