# -*- coding: utf-8 -*
import JackFramework as jf

from .BodyReconstruction.inference import BodyReconstructionInterface


def model_zoo(args: object, name: str) -> object:
    for case in jf.Switch('BodyReconstruction'):
        if case('BodyReconstruction'):
            jf.log.info("Enter the BodyReconstruction model")
            model = BodyReconstructionInterface(args)
            break
        if case(''):
            model = None
            jf.log.error("The model's name is error!!!")
    return model
