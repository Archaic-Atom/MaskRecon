# -*- coding: utf-8 -*-
import JackFramework as jf

from .dataloader import BodyDataloader


def dataloaders_zoo(args: object, name: str) -> object:
    for case in jf.Switch(name):
        if case('thuman2.0'):
            jf.log.info("Enter the your dataloader")
            dataloader = BodyDataloader(args)
            break
        if case(''):
            dataloader = None
            jf.log.error("The dataloader's name is error!!!")
    return dataloader
