 # -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
#import JackFramework as jf


class DataSaver(object):
    """docstring for DataSaver"""
    _DEPTH_UNIT = 1000.0
    _DEPTH_DIVIDING = 255.0

    def __init__(self, args: object) -> object:
        super().__init__()
        self.__args = args

    def save_output_color(self, color_pre: np.array, 
                    img_id: int, dataset_name: str,
                    supplement: list) -> None:
        #print(color_pre.shape)
        batch_size, _, _, _, = color_pre.shape
        #names = supplement[0]
        for i in range(batch_size):
            temp_color = color_pre[i,:,:,:]
            #print(temp_depth.shape)
            name = batch_size * img_id + i

            self._save_output_color(temp_color, name)

    
    def save_output_depth(self, depth_pre: np.array, 
                    normal_pre: np.array,
                    img_id: int, dataset_name: str,
                    supplement: list) -> None:
        #print(color_pre.shape)
        batch_size, _, _, _, = normal_pre.shape
        #names = supplement[0]
        for i in range(batch_size):
            temp_depth = depth_pre[i,:,:,:]
            temp_normal = normal_pre[i,:,:,:]
            #print(temp_depth.shape)
            name = batch_size * img_id + i

            self._save_output_depth(temp_depth, name)
            self._save_output_normal(temp_normal, name)
            #self._save_output_mesh(temp_depth, temp_color, name)

    def save_output_mesh(self, color_pre: np.array, 
                    depth_pre: np.array,
                    img_id: int, dataset_name: str,
                    supplement: list) -> None:
        #print(color_pre.shape)
        batch_size, _, _, _, = color_pre.shape
        #names = supplement[0]
        for i in range(batch_size):
            temp_color = color_pre[i,:,:,:]
            temp_depth = depth_pre[i,:,:,:]          
            #print(temp_depth.shape)
            name = batch_size * img_id + i

            self._save_output_mesh(temp_depth, temp_color, name)

    def _save_output_depth(self, img: np.array, num: int) -> None:
        args = self.__args      
        path = self._generate_output_img_path(args.resultImgDir, num, "%04d_depth_front")        
        img =  img.transpose(1, 2, 0)
        img = (img * float(DataSaver._DEPTH_UNIT)).astype(np.uint16)
        img = np.where(img<DataSaver._DEPTH_UNIT,img,0)
        #print("depth_img")
        #print(img.shape)
        cv2.imwrite(path, img)


    def _save_output_color(self, img: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num, "%04d_color_front") 
        #img = np.squeeze(img)
        img = img.transpose(1, 2, 0)
        #img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        #cv2.imwrite(path, img)
        img = (img  * float(DataSaver._DEPTH_DIVIDING)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)

    def _save_output_mesh(self, depth: np.array, color: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num, "%04d_mesh_front") 
        depth = np.squeeze(depth)
        depth = (depth * float(DataSaver._DEPTH_UNIT))
        self._remove_points(depth)
        color = color.transpose(1, 2, 0)
        color = (color  * float(DataSaver._DEPTH_DIVIDING)).astype(np.uint8)
        depth = cv2.flip(depth,1)
        color = cv2.flip(color,1)

        low_thres = 100 
        mask = depth > low_thres
        mask = mask.astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        eroded = cv2.erode(mask, kernel)
        edge = (mask - eroded).astype(np.bool)
        depth[edge] = depth[edge] * 2 
        self._depth2mesh(depth, mask, color, path)


    def _remove_points(self, fp):
        f0 = (fp>1000)
        f1 = fp<10
        fp[f0] = 0.0
        fp[f1] = 0.0

    
    def _write_matrix_txt(self,a,filename):
        mat = np.matrix(a)
        with open(filename,'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.5f')

    def _save_output_normal(self, img: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num, "%04d_normal_front") 
        normal=np.array(img*255)
        #np.savetxt("normal.txt",normal[0,:,:], fmt='%.8f')
        normal = cv2.cvtColor(np.transpose(normal, [1, 2, 0]), cv2.COLOR_BGR2RGB)         
        cv2.imwrite(path, normal.astype(np.uint8))

    # Function borrowed from https://github.com/sfu-gruvi-3dv/deep_human
    def _depth2mesh(self, depth, mask, color, filename):
        h = depth.shape[0]
        w = depth.shape[1]
        #depth = depth.reshape(h,w,1)
        depth = depth/1000
        f = open(filename + ".obj", "w")
        for i in range(h):
            for j in range(w):
                f.write('v '+str(float(2.0*i/h))+' '+str(float(2.0*j/w))+' '+str(float(depth[i,j]))\
                    +' '+str(float(color[i,j,0]))+' '+str(float(color[i,j,1]))+' '+str(float(color[i,j,2]))+'\n')

        threshold = 0.07

        for i in range(h-1):
            for j in range(w-1):
                if i < 2 or j < 2:
                    continue
                localpatch= np.copy(depth[i-1:i+2,j-1:j+2])
                dy_u = localpatch[0,:] - localpatch[1,:]
                dx_l = localpatch[:,0] - localpatch[:,1]
                dy_d = localpatch[0,:] - localpatch[-1,:]
                dx_r = localpatch[:,0] - localpatch[:,-1]
                dy_u = np.abs(dy_u)
                dx_l = np.abs(dx_l)
                dy_d = np.abs(dy_d)
                dx_r = np.abs(dx_r)
                if np.max(dy_u)<threshold and np.max(dx_l) < threshold and np.max(dy_d) < threshold and np.max(dx_r) < threshold and mask[i,j]:
                    f.write('f '+str(int(j+i*w+1))+' '+str(int(j+i*w+1+1))+' '+str(int((i + 1)*w+j+1))+'\n')
                    f.write('f '+str(int((i+1)*w+j+1+1))+' '+str(int((i+1)*w+j+1))+' '+str(int(i * w + j + 1 + 1)) + '\n')
        f.close()


    @staticmethod
    def _generate_output_img_path(dir_path: str, num: str,
                                  filename_format: str = "%04d_10",
                                  img_type: str = ".png"):
        return dir_path + filename_format % num + img_type
    
    def _generate_output_mesh_path(dir_path: str, num: str,
                                  filename_format: str = "%04d_10",
                                  img_type: str = ".obj"):
        return dir_path + filename_format % num + img_type

