 # -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
#import JackFramework as jf
import torch
import trimesh


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
             
            #self._remove_outliers(temp_depth)

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

    # this part of code comes from normalgan
    def save_output_mesh_2(self, color_pre: np.array, 
                    depth_pre: np.array,
                    color_front: np.array,
                    depth_front: np.array,
                    img_id: int, dataset_name: str,
                    supplement: list) -> None:
        #print(color_pre.shape)
        batch_size, _, _, _, = color_pre.shape
        #names = supplement[0]
        for i in range(batch_size):
            temp_color_front = color_front[i,:,:,:]
            temp_depth_front = depth_front[i,:,:,:]
            temp_color_back = color_pre[i,:,:,:]
            temp_depth_back = depth_pre[i,:,:,:]
                  
            name = batch_size * img_id + i
            self._merge_output_mesh(temp_color_front, temp_depth_front, temp_color_back, temp_depth_back, name)
            

    def _merge_output_mesh(self, color_f: np.array, 
                            depth_f: np.array, 
                            color_b: np.array, 
                            depth_b: np.array, 
                            num: int) -> None:
        args = self.__args
        crop_w = args.imgWidth
        crop_h = args.imgHeight
        #path = self._generate_output_mesh_path(args.resultImgDir, num, "%04d_mesh") 
        path_obj = args.resultImgDir + "%04d_mesh" % num + ".obj"
        path_ply = args.resultImgDir + "%04d_mesh" % num + ".ply"
        #color_f = color_f.transpose(1, 2, 0)
        #color_b = color_b.transpose(1, 2, 0)
        #depth_f = depth_f.transpose(1, 2, 0)
        #depth_b = depth_b.transpose(1, 2, 0)
        color_f = color_f  * float(DataSaver._DEPTH_DIVIDING)
        color_b = color_b  * float(DataSaver._DEPTH_DIVIDING)
        depth_f = depth_f * float(DataSaver._DEPTH_UNIT)
        depth_b = depth_b * float(DataSaver._DEPTH_UNIT)
        #depth_b = np.where(depth_b<DataSaver._DEPTH_UNIT * 0.99,depth_b,0)


        # buff data
        #depth_f = depth_f * float(3.0)
        #depth_b = depth_b + 1000

        # real data
        

        #zed data
        #depth_f = depth_f * float(3.0) + 50
        #depth_b = depth_b * float(1.5)+ 850 

        # another real data
        #depth_f = depth_f * float(10)
        #depth_b = depth_b + 500

        #tang dataset
        #depth_f = depth_f * float(4.0)
        #depth_b = np.where(depth_b>0,depth_b *float(2.0)  + 1150, 0)

        # normalgan dataset
        #depth_f= depth_f*2
        # for image 0
        #depth_b = depth_b+490
        #for image 1 and 2
        #depth_b = depth_b/1.5 + 500


        # xu teacher
        #depth_f = depth_f * float(1.5) +50
        #depth_b = depth_b * float(1.8)

 
        #self._remove_outliers_1(depth_f)
        self._remove_outliers_1(depth_b) 
        self._remove_outliers_2(depth_b)
        #self._remove_outliers_2(depth_f)


        #_, crop_h, crop_w = depth_f.shape

        Y, X = np.meshgrid(np.arange(crop_h), np.arange(crop_w))

        x_cord = X * 2
        y_cord = Y * 2
        x_cord = x_cord.reshape(1, crop_h, crop_w)
        y_cord = y_cord.reshape(1, crop_h, crop_w)
        fp_idx = np.zeros([crop_h, crop_w], dtype=np.int)
        bp_idx = np.ones_like(fp_idx) * (crop_h * crop_w)
        for hh in range(crop_h):
            for ww in range(crop_w):
                fp_idx[hh, ww] = hh * crop_w + ww
                bp_idx[hh, ww] += hh * crop_w + ww
        # convert the images to 3D mesh
        fpct = np.concatenate((x_cord, y_cord, depth_f, color_f), axis=0)
        bpct = np.concatenate((x_cord, y_cord, depth_b, color_b), axis=0)
        #print("fpct:", fpct.shape)
        # dilate for the edge point interpolation
        #fpct = self._dilate(fpct, 1)
        #bpct = self._dilate(bpct, 1)
        #fpct = self._erode(fpct, 1)
        #bpct = self._erode(bpct, 1)
        #print("fpct:", fpct.shape)
        fpc = np.transpose(fpct, [1, 2, 0])
        bpc = np.transpose(bpct, [1, 2, 0])

        self._remove_points(bpc)
        self._remove_outliers(bpc)
         
        self._remove_points_1(fpc, bpc)
        
        # get the edge region for the edge point interpolation
        low_thres = 100.0 
        mask_pc = fpc[:, :, 2] > low_thres
        mask_pc = mask_pc.astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        eroded = cv2.erode(mask_pc, kernel)
        edge = (mask_pc - eroded).astype(np.bool)
        # interpolate 2 points for each edge point pairs
        fpc[edge, 2:6] = (fpc[edge, 2:6] * 4 + bpc[edge, 2:6] * 1) / 5
        bpc[edge, 2:6] = (fpc[edge, 2:6] * 2 + bpc[edge, 2:6] * 3) / 5
        fpc = fpc.reshape(-1, 6)
        bpc = bpc.reshape(-1, 6)
        if (np.sum(mask_pc) < 100):
            print('noimage')
        fix_p = 1555
        f_faces = self._getfrontFaces(mask_pc, fp_idx)
        b_faces = self._getbackFaces(mask_pc, bp_idx)
        edge_faces = self._getEdgeFaces(mask_pc, fp_idx, bp_idx)
        faces = np.vstack((f_faces, b_faces, edge_faces))
        points = np.concatenate((fpc, bpc), axis=0)
        points[:, 0:3] = -(points[:, 0:3] - np.array([[crop_w / 2 , (crop_h - 5)  - 700, fix_p]])) / 1000.0
        #points[:, 0] = -points[:, 0]
        vertices = points[:, 0:3]
        colors = points[:, 3:6].astype(np.uint8)
        
        #colors = points[:, 3:6].astype(int)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
        self._ply_from_array_color(mesh.vertices, mesh.visual.vertex_colors, 
                                    mesh.faces, path_ply)
        #self._save_obj_mesh_with_color(path_obj, mesh.vertices, mesh.faces, mesh.visual.vertex_colors)
        points[:, 0] = -points[:, 0]
        vertices = points[:, 0:3]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
        self._save_obj_mesh_without_color(path_obj, mesh.vertices, mesh.faces)
        
        '''
        #处理噪声 加强背面颜色
        self._meshcleaning(path)
        out_mesh = trimesh.load(path).split()[0]
        #print("out_mesh", out_mesh.shape)
        color = out_mesh.visual.vertex_colors
        verts = out_mesh.vertices
        print(verts.shape)
        #faces = out_mesh.faces
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib_tensor = torch.Tensor(projection_matrix).unsqueeze(0).float()
        #print(calib_tensor[:1].shape)
        out_mesh.visual.vertex_colors = np.array(self._esti_color(np.array(color,dtype='uint16'), \
            self._orthogonal(torch.from_numpy(verts.T).unsqueeze(0).float().cuda(), calib_tensor[:1])),dtype='uint8') #更新颜色
        out_mesh.export(path)
        '''
        
    def _meshcleaning(self, obj_path):
        '''
        只留下最大的连通图 去噪音
        '''
        print(f"Processing mesh cleaning: {obj_path}")

        mesh = trimesh.load(obj_path)
        cc = mesh.split()    

        out_mesh = cc[0]
        bbox = out_mesh.bounds
        height = bbox[1,0] - bbox[0,0]
        #找高度最大的连通体，舍弃噪音
        for c in cc:
            bbox = c.bounds
            if height < bbox[1,0] - bbox[0,0]:
                height = bbox[1,0] - bbox[0,0]
                out_mesh = c
        
        out_mesh.export(obj_path)  

    def _orthogonal(self, points, calib, transform=None):
        '''
        使用正交投影将点投影到屏幕空间
        args:
            points: [B, 3, N] 3d points in world coordinates
            calib: [B, 3, 4] projection matrix
            transform: [B, 2, 3] screen space transformation
        return:
            [B, 3, N] 3d coordinates in screen space
        '''
        rot = calib[:, :3, :3]  #[B, 3, 3]
        trans = calib[:, :3, 3:4]  #[B, 3, 1]
        print(points.shape, calib.shape)
        pts = torch.baddbmm(trans.cuda(), rot.cuda(), points.cuda()) #[B, 3, N]
        if transform is not None:
            scale = transform[:2, :2]
            shift = transform[:2, 2:3]
            pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
        return pts

    def _esti_color(self, color, xyz_tensor):
        '''
        重新渲染颜色，每一个z<0的点去找y最近的左右两个点的颜色
        '''
        xyz = xyz_tensor.cpu().numpy()[0].T
        _ = list(range(xyz.shape[0]))
        xyz_1 = [[*xyz[i].tolist(),_[i]] for i in range(xyz.shape[0]) if xyz[i][2]<0]
        x_2 = np.array([[*xyz[i].tolist()[:2],_[i]] for i in range(xyz.shape[0]) if xyz[i][2]>=0 and xyz[i][2]<0.001])
        
        def find_closest(point,x_l):
            x = point[0]
            y = point[1]


            try:
                left = [int(i[2]) for i in sorted(x_l[x_l[:,0]-x<0], key=lambda x:(-abs(x[1]-y),x[0]),reverse=True)[:10]]
            except:
                left = [int(i[2]) for i in sorted(x_l[x_l[:,0]-x<0], key=lambda x:(-abs(x[1]-y),x[0]),reverse=True)]
            try:
                right = [int(i[2]) for i in sorted(x_l[x_l[:,0]-x>=0], key=lambda x:(abs(x[1]-y),x[0]))[:10]]
            except:
                right = [int(i[2]) for i in sorted(x_l[x_l[:,0]-x>=0], key=lambda x:(abs(x[1]-y),x[0]))]
            left = None if len(left)==0 else left
            right = None if len(right)==0 else right
            return [left, right]
        
        for i in range(len(xyz_1)):
            point = xyz_1[i]
            left, right = find_closest(point, x_2)
            #print(left, right)
            if right != None and left != None:
                temp_color = (sum(color[left])+sum(color[right]))/(len(left) + len(right))
            elif right == None and left != None:
                temp_color = sum(color[left])/len(left)
            elif right != None and left == None:
                temp_color = sum(color[right])/len(right)
            else:
                raise Exception('找不到最近颜色。')
            color[point[-1]] = temp_color
        return color

    def _save_output_depth(self, img: np.array, num: int) -> None:
        args = self.__args      
        path = self._generate_output_img_path(args.resultImgDir, num, "%04d_depth_pre")        
        img =  img.transpose(1, 2, 0)
        img = (img * float(DataSaver._DEPTH_UNIT)).astype(np.uint16)
        img = np.where(img<DataSaver._DEPTH_UNIT,img,0)
        
        #print("depth_img")
        #print(img.shape)
        cv2.imwrite(path, img)


    def _save_output_color(self, img: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num, "%04d_color_pre") 
        #img = np.squeeze(img)
        img = img.transpose(1, 2, 0)
        #img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        #cv2.imwrite(path, img)
        img = (img  * float(DataSaver._DEPTH_DIVIDING)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)

    def _save_output_mesh(self, depth: np.array, color: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num, "%04d_mesh_pre") 
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
        f0 = fp[:,:,2]>800
        f1 = fp[:,:,2]<50
        fp[f0] = 0.0
        fp[f1] = 0.0
        return fp


    def _remove_outliers(self, data):
        mask = data[:,:,2]>0
        data0 = data[mask]
        mean = np.mean(data0[:,2])
        std = np.std(data0[:,2])
        max = np.max(data0[:,2])
        min = np.min(data0[:,2])      
        llim = 100
        ulim = mean + 3*std
        mask_ulim = data[:,:,2] > ulim
        #mask_llim = data < llim
        data[mask_ulim] = 0
        #data[mask_llim] = 0

        #print(mean, std, max, min, ulim, llim)
        return data

    def _remove_outliers_2(self, data):
        _, h, w = data.shape
        data0 = data[0,:,:]
        threshold = 30
        for i in range(1,h-1):
            for j in range(1,w-1):
                if (data0[i,j] > 0):
                    if (abs(data0[i,j]-data0[i-1,j])>threshold) and \
                        (abs(data0[i,j]-data0[i+1,j])>threshold):
                        data[:,i,j] = 0
                        #if (data0[i-1,j]>0 and data0[i+1,j]>0):
                        #    data[:,i,j] = (data0[i-1,j] + data0[i+1,j])/2.0
                        #else:
                        #    data[:,i,j] = (data0[i-1,j] + data0[i+1,j])

                        

        
    def _remove_outliers_1(self, data):
        _, h, w = data.shape
        data0 = data[0,:,:]
        threshold = 20
        for i in range(1,h-1):
            for j in range(1,w-1):
                if ((data0[i,j]>0) and (data0[i-1,j]>0) and (abs(data0[i,j]-data0[i-1,j])>threshold)) \
                    or ((data0[i,j]>0) and (data0[i,j-1]>0) and (abs(data0[i,j]-data0[i,j-1])>threshold)):
                    data[:,i,j] = 0
        #dx = abs((data0[:, 2:h,1:w-1]-data0[:, 0:h-2,1:w-1]))
        #dx_1 = abs((data0[:, 2:h,1:w-1]-data0[:, 0:h-2,1:w-1]))
        #dy = abs((data0[:, 1:h-1,2:w]-data0[:, 1:h-1,0:w-2])) 
        #threshold = 100
        #data = np.where(dx < threshold, data, 0)  
        #data = np.where(dy < threshold, data, 0)  
        #data[dx>100] = 0
        

    
    def _write_matrix_txt(self,a,filename):
        mat = np.matrix(a)
        with open(filename,'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.5f')

    def _save_output_normal(self, img: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num, "%04d_normal_pre") 
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
                                  img_type: str = ".ply"):
        return dir_path + filename_format % num + img_type

    def _dilate(self, depth, pix):
        # depth: [1, H,W]
        newdepth = np.array(depth)
        for i in range(pix):
            d1 = newdepth[:, 1:, :]
            d2 = newdepth[:, :-1, :]
            d3 = newdepth[:, :, 1:]
            d4 = newdepth[:, :, :-1]
            newdepth[:, :-1, :] = np.where(newdepth[0:1, :-1, :] > 0, newdepth[:, :-1, :], d1)
            newdepth[:, 1:, :] = np.where(newdepth[0:1, 1:, :] > 0, newdepth[:, 1:, :], d2)
            newdepth[:, :, :-1] = np.where(newdepth[0:1, :, :-1] > 0, newdepth[:, :, :-1], d3)
            newdepth[:, :, 1:] = np.where(newdepth[0:1, :, 1:] > 0, newdepth[:, :, 1:], d4)
            depth = newdepth
        return newdepth

    def _erode(self, depth, pix):
        # depth: [B, C, H, W]
        newdepth = np.array(depth)
        for i in range(pix):
            d1 = depth[:, 1:, :]
            d2 = depth[:, :-1, :]
            d3 = depth[:, :, 1:]
            d4 = depth[:, :, :-1]
            newdepth[:, :-1, :] = np.where(newdepth[:, :-1, :] > 0, d1, newdepth[:, :-1, :])
            newdepth[:, 1:, :] = np.where(newdepth[:, 1:, :] > 0, d2, newdepth[:, 1:, :])
            newdepth[:, :, :-1] = np.where(newdepth[:, :, :-1] > 0, d3, newdepth[:, :, :-1])
            newdepth[:, :, 1:] = np.where(newdepth[:, :, 1:] > 0, d4, newdepth[:, :, 1:])
            depth = newdepth
        return newdepth

    def _getfrontFaces(self, mask, p_idx):
        p_valid_idx = p_idx * mask
        p00_idx = p_valid_idx[:-1, :-1].reshape(-1, 1)
        p10_idx = p_valid_idx[1:, :-1].reshape(-1, 1)
        p11_idx = p_valid_idx[1:, 1:].reshape(-1, 1)
        p01_idx = p_valid_idx[:-1, 1:].reshape(-1, 1)
        all_faces = np.vstack((np.hstack((p00_idx, p10_idx, p01_idx)), np.hstack((p01_idx, p10_idx, p11_idx)),
                            np.hstack((p00_idx, p10_idx, p11_idx)), np.hstack((p00_idx, p11_idx, p01_idx))))
        fp_faces = all_faces[np.where(all_faces[:, 0] * all_faces[:, 1] * all_faces[:, 2] > 0)]
        return fp_faces

    def _getbackFaces(self, mask, p_idx):
        p_valid_idx = p_idx * mask
        p00_idx = p_valid_idx[:-1, :-1].reshape(-1, 1)
        p10_idx = p_valid_idx[1:, :-1].reshape(-1, 1)
        p11_idx = p_valid_idx[1:, 1:].reshape(-1, 1)
        p01_idx = p_valid_idx[:-1, 1:].reshape(-1, 1)
        all_faces = np.vstack((np.hstack((p00_idx, p01_idx, p10_idx)), np.hstack((p01_idx, p11_idx, p10_idx)),
                            np.hstack((p00_idx, p11_idx, p10_idx)), np.hstack((p00_idx, p01_idx, p11_idx))))
        fp_faces = all_faces[np.where(all_faces[:, 0] * all_faces[:, 1] * all_faces[:, 2] > 0)]
        return fp_faces

    def _getEdgeFaces(self, mask, fp_idx, bp_idx):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        all_boundary_faces_idx = []
        for i in range(len(contours)):
            edges = contours[i][:, 0, :]
            nextedges = np.vstack((edges[1:], edges[0]))
            fp_edge_idx = fp_idx[edges[:, 1], edges[:, 0]].reshape(-1, 1)
            bp_edge_idx = bp_idx[edges[:, 1], edges[:, 0]].reshape(-1, 1)
            bp_nextedge_idx = bp_idx[nextedges[:, 1], nextedges[:, 0]].reshape(-1, 1)
            fp_nextedge_idx = fp_idx[nextedges[:, 1], nextedges[:, 0]].reshape(-1, 1)
            boundary_faces_idx = np.vstack((np.hstack((fp_edge_idx, bp_edge_idx, bp_nextedge_idx)),
                                            np.hstack((fp_edge_idx, bp_nextedge_idx, fp_nextedge_idx))))
            if i == 0:
                all_boundary_faces_idx = boundary_faces_idx
            else:
                all_boundary_faces_idx = np.vstack((all_boundary_faces_idx, boundary_faces_idx))
        return all_boundary_faces_idx

    def _save_obj_mesh_with_color(self, mesh_path, verts, faces, colors):
        file = open(mesh_path, 'w')

        for idx, v in enumerate(verts):
            c = colors[idx]
            file.write('v %.4f %.4f %.4f %d %d %d\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
        for f in faces:
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
        file.close()   


    def _save_obj_mesh_without_color(self, mesh_path, verts, faces):
        file = open(mesh_path, 'w')

        for idx, v in enumerate(verts):
            file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        for f in faces:
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
        file.close()  

    def _ply_from_array_color(self, points, colors, faces, output_file):

        num_points = len(points)
        num_triangles = len(faces)

        header = '''ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    element face {}
    property list uchar int vertex_indices
    end_header\n'''.format(num_points, num_triangles)

        with open(output_file,'w') as f:
            f.writelines(header)
            index = 0
            for item in points:
                f.write("{0:0.6f} {1:0.6f} {2:0.6f} {3} {4} {5}\n".format(item[0], item[1], item[2],
                                                        colors[index, 0], colors[index, 1], colors[index, 2]))
                index = index + 1

            for item in faces:
                number = len(item)
                row = "{0}".format(number)
                for elem in item:
                    row += " {0} ".format(elem)
                row += "\n"
                f.write(row)

    
    def _remove_points_1(self, fp, bp):
        f0 = fp[:, :, 2] - bp[:, :, 2]
        f0 = f0 > 0
        fp[f0, 2] = 0.0
        bp[f0, 2] = 0.0  
