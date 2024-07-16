# -*- coding: utf-8 -*-
import argparse
import JackFramework as jf
import UserModelImplementation.user_define as user_def

# model and dataloader
from UserModelImplementation import Models
from UserModelImplementation import Dataloaders


class UserInterface(jf.UserTemplate.NetWorkInferenceTemplate):
    """docstring for UserInterface"""

    def __init__(self) -> object:
        super().__init__()

    def inference(self, args: object) -> object:
        dataloader = Dataloaders.dataloaders_zoo(args, args.dataset)
        model = Models.model_zoo(args, args.modelName)
        return model, dataloader

    def user_parser(self, parser: object) -> object:
        # return parser
        parser.add_argument('--lr_scheduler', type=UserInterface.__str2bool,
                            default=user_def.LR_SCHEDULER,
                            help='use or not use lr scheduler')
        parser.add_argument('--save_mesh', type=UserInterface.__str2bool,
                            default=user_def.SAVE_MESH,
                            help='save or not save mesh')
        parser.add_argument('--mask', type=UserInterface.__str2bool,
                            default=user_def.MASK_OPT,
                            help='save or not save mesh')
        return parser

    @staticmethod
    def __str2bool(arg: str) -> bool:
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
