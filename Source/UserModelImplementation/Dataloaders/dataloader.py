# -*- coding: utf-8 -*-
from dataclasses import dataclass
from pickle import TRUE
import time
import JackFramework as jf
# import UserModelImplementation.user_define as user_def
from .data_set import BodyReconstructionDataset
from .data_saver import DataSaver

# traing
ID_COLOR = 0
ID_DEPTH = 1
ID_UV = 2
ID_COLOR_GT =3
ID_DEPTH_GT =4
#testing

class BodyDataloader(jf.UserTemplate.DataHandlerTemplate):
    """docstring for DataHandlerTemplate"""

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.__result_str = jf.ResultStr()
        self.__train_dataset = None
        self.__val_dataset = None
        self.__saver = DataSaver(args)
        self.__start_time = 0
        #self.model0_res = None
        
        

    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        args = self.__args
        self.__train_dataset = BodyReconstructionDataset(args, args.trainListPath, is_training)
        return self.__train_dataset

    def get_val_dataset(self, path: str) -> object:
        # return dataset
        args = self.__args
        self.__val_dataset = BodyReconstructionDataset(args, args.valListPath, False)
        return self.__val_dataset

    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        self.__start_time = time.time()
        if is_training:
            # return input_data_list, label_data_list
             return [batch_data[ID_COLOR], batch_data[ID_DEPTH], batch_data[ID_UV], batch_data[ID_COLOR_GT], batch_data[ID_DEPTH_GT]], \
                  [batch_data[ID_COLOR_GT], batch_data[ID_DEPTH_GT]]
            # return input_data, supplement
        return [batch_data[ID_COLOR], batch_data[ID_DEPTH], batch_data[ID_UV]], []

    def show_train_result(self, epoch: int, loss:
                          list, acc: list,
                          duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        #info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, True)
        info_str = self.__result_str.training_list_intermediate_result(epoch, loss, acc)
        jf.log.info(info_str)

    def show_val_result(self, epoch: int, loss:
                        list, acc: list,
                        duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, False)
        jf.log.info(info_str)

    def save_result(self, output_data: list, supplement: list,
                    img_id: int, model_id: int) -> None:
        assert self.__train_dataset is not None
        args = self.__args
        # save method  
        if model_id == 0:
            #self.model0_res = output_data[0].detach().cpu().numpy()
            self.__saver.save_output_color(output_data[0].detach().cpu().numpy(),
                                    img_id, args.dataset, supplement)

            self.__saver.save_output_depth(output_data[1].detach().cpu().numpy(),
                                    output_data[2].detach().cpu().numpy(),
                                    img_id, args.dataset, supplement)
            if args.save_mesh == True:
                self.__saver.save_output_mesh_2(output_data[0].detach().cpu().numpy(),
                                    output_data[1].detach().cpu().numpy(),
                                    output_data[3].detach().cpu().numpy(),
                                    output_data[4].detach().cpu().numpy(),
                                   img_id, args.dataset, supplement)

            
        
    

    def show_intermediate_result(self, epoch: int,
                                 loss: list, acc: list) -> str:
        assert len(loss) == len(acc)  # same model number
        data_str = self.__result_str.training_list_intermediate_result(epoch, loss, acc)
        return data_str
