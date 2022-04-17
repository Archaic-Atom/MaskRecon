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
            print("name:", type(name))
            self._merge_output_mesh(temp_color_front, temp_depth_front, temp_color_back, temp_depth_back, name)

    def _merge_output_mesh(self, color_f: np.array, 
                            depth_f: np.array, 
                            color_b: np.array, 
                            depth_b: np.array, 
                            num: int) -> None:
        args = self.__args
        #path = self._generate_output_mesh_path(args.resultImgDir, num, "%04d_mesh") 
        path = args.resultImgDir + "%04d_mesh" % num + ".ply"

        #color_f = color_f.transpose(1, 2, 0)
        #color_b = color_b.transpose(1, 2, 0)
        #depth_f = depth_f.transpose(1, 2, 0)
        #depth_b = depth_b.transpose(1, 2, 0)
        color_f = color_f  * float(DataSaver._DEPTH_DIVIDING)
        color_b = color_b  * float(DataSaver._DEPTH_DIVIDING)
        depth_f = depth_f * float(DataSaver._DEPTH_UNIT)
        depth_b = depth_b * float(DataSaver._DEPTH_UNIT)

        #depth_b = cv2.flip(depth_b,1)
        #color_b = cv2.flip(color_b,1)

        #depth_b = -depth_b + 1000

        _, crop_h, crop_w = depth_f.shape
        #Y, X = torch.meshgrid(torch.tensor(range(crop_h)), torch.tensor(range(crop_w)))
        #X = X.unsqueeze(0).unsqueeze(0).float()  # (B,H,W)
        #Y = Y.unsqueeze(0).unsqueeze(0).float()
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
        print(x_cord.shape, y_cord.shape, depth_f.shape, color_f.shape)
        fpct = np.concatenate((x_cord, y_cord, depth_f, color_f), axis=0)
        bpct = np.concatenate((x_cord, y_cord, depth_b, color_b), axis=0)
        print("fpct:", fpct.shape)
        # dilate for the edge point interpolation
        fpct = self._dilate(fpct, 1)
        bpct = self._dilate(bpct, 1)
        print("fpct:", fpct.shape)
        #fpc = fpct.permute(1, 2, 0)
        #bpc = bpct.permute(1, 2, 0)
        fpc = np.transpose(fpct, [1, 2, 0])
        bpc = np.transpose(bpct, [1, 2, 0])
        print("bpc:", fpc.shape)
        self._remove_points(fpc)
        self._remove_points(bpc)
        self._remove_points_1(fpc, bpc)
        
        # get the edge region for the edge point interpolation
        low_thres = 100.0 
        mask_pc = fpc[:, :, 2] > low_thres
        mask_pc = mask_pc.astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        eroded = cv2.erode(mask_pc, kernel)
        edge = (mask_pc - eroded).astype(np.bool)
        # interpolate 2 points for each edge point pairs
        fpc[edge, 2:6] = (fpc[edge, 2:6] * 2 + bpc[edge, 2:6] * 1) / 3
        bpc[edge, 2:6] = (fpc[edge, 2:6] * 1 + bpc[edge, 2:6] * 2) / 3
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
        points[:, 0] = -points[:, 0]
        vertices = points[:, 0:3]
        colors = points[:, 3:6].astype(np.uint8)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
        self._ply_from_array_color(mesh.vertices, mesh.visual.vertex_colors, 
                                    mesh.faces, path)
        
        

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
        f0 = fp[:,:,2]>650
        f1 = fp[:,:,2]<10
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
