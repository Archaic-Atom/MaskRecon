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
        batch_size, _, _, _, = color_pre.shape
        for i in range(batch_size):
            temp_color = color_pre[i,:,:,:]
            name = batch_size * img_id + i

            self._save_output_color(temp_color, name)

    
    def save_output_depth(self, depth_pre: np.array, 
                    normal_pre: np.array,
                    img_id: int, dataset_name: str,
                    supplement: list) -> None:
        batch_size, _, _, _, = normal_pre.shape
        for i in range(batch_size):
            temp_depth = depth_pre[i,:,:,:]
            temp_normal = normal_pre[i,:,:,:]
            name = batch_size * img_id + i

            self._save_output_depth(temp_depth, name)
            self._save_output_normal(temp_normal, name)


    def save_output_mesh(self, color_pre: np.array, 
                    depth_pre: np.array,
                    color_front: np.array,
                    depth_front: np.array,
                    img_id: int, dataset_name: str,
                    supplement: list) -> None:
        batch_size, _, _, _, = color_pre.shape

        for i in range(batch_size):
            temp_color_front = color_front[i,:,:,:]
            temp_depth_front = depth_front[i,:,:,:]
            temp_color_back = color_pre[i,:,:,:]
            temp_depth_back = depth_pre[i,:,:,:]
                  
            name = batch_size * img_id + i
            self._merge_output_mesh(temp_color_front, temp_depth_front, temp_color_back, temp_depth_back, name)
            
    # this part of codes comes from normalgan
    def _merge_output_mesh(self, color_f: np.array, 
                            depth_f: np.array, 
                            color_b: np.array, 
                            depth_b: np.array, 
                            num: int) -> None:
        args = self.__args
        crop_w = args.imgWidth
        crop_h = args.imgHeight

        path_ply = args.resultImgDir + "%04d_mesh" % num + ".ply"

        color_f = color_f  * float(DataSaver._DEPTH_DIVIDING)
        color_b = color_b  * float(DataSaver._DEPTH_DIVIDING)
        depth_f = depth_f * float(DataSaver._DEPTH_UNIT)
        depth_b = depth_b * float(DataSaver._DEPTH_UNIT)
        
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

        vertices = points[:, 0:3]
        colors = points[:, 3:6].astype(np.uint8)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
        self._ply_from_array_color(mesh.vertices, mesh.visual.vertex_colors, 
                                    mesh.faces, path_ply)



    def _save_output_depth(self, img: np.array, num: int) -> None:
        args = self.__args      
        path = self._generate_output_img_path(args.resultImgDir, num, "%04d_depth_pre")        
        img =  img.transpose(1, 2, 0)
        img = (img * float(DataSaver._DEPTH_UNIT)).astype(np.uint16)
        img = np.where(img<DataSaver._DEPTH_UNIT,img,0)

        cv2.imwrite(path, img)


    def _save_output_color(self, img: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num, "%04d_color_pre") 
        img = img.transpose(1, 2, 0)
        img = (img  * float(DataSaver._DEPTH_DIVIDING)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)



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
        ulim = mean + 3*std
        mask_ulim = data[:,:,2] > ulim
        data[mask_ulim] = 0
        return data    
    
    def _write_matrix_txt(self,a,filename):
        mat = np.matrix(a)
        with open(filename,'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.5f')

    def _save_output_normal(self, img: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num, "%04d_normal_pre") 
        normal=np.array(img*255)
        normal = cv2.cvtColor(np.transpose(normal, [1, 2, 0]), cv2.COLOR_BGR2RGB)         
        cv2.imwrite(path, normal.astype(np.uint8))



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
