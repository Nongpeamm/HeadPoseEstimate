import sys
import numpy as np
import os
from insightfaceDetector.scrfd import SCRFD
from insightfaceRecognition.utils.face_align import norm_crop
from insightfaceRecognition.utils.utils import load_pickle, compute_sim,save_pickle
from insightfaceRecognition.arcface_onnx import ArcFaceONNX


class Face_class:
    def __init__(self, model_rec_path=r"D:\Github\HeadPoseEstimate\insightfaceRecognition\model\w600k_r50.onnx", det_thresh=0.5, ctx_id=0, det_size=(640, 640), ref_face=3):
        # Load model
        self.model_det = SCRFD()
        self.model_det.prepare(
            ctx_id, input_size=det_size, det_thresh=det_thresh)
        self.model_rec = ArcFaceONNX(model_file=model_rec_path)
        self.model_rec.prepare(ctx_id=ctx_id)
        self.list_face, self.ref_face = [], ref_face
        frame_dummy = np.zeros((112, 112, 3), np.uint8)
        _ = self.model_rec.get_feat(frame_dummy)
        self.thres_center = 0.25
        self.thres_m = 0.225

    def detect(self, img, max_num=10, metric='default'):
        # bboxes, landmarks
        return self.model_det.detect(img=img, max_num=max_num, metric=metric)

    def embed(self, img, landmark):
        _img = norm_crop(img, landmark=landmark)
        embedding = self.model_rec.get_feat(_img).flatten()
        return embedding

    def norm_crop(self, img, landmark):
        return norm_crop(img, landmark=landmark)

    def run_detect_em(self, img, max_num=10, metric='default'):
        bboxes, landmarks = self.model_det.detect(
            img=img, max_num=max_num, metric=metric)
        output = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4].astype('int')
            det_score = bboxes[i, 4]
            landmark = landmarks[i]
            
            output.append(
                {"bbox": bbox,"confidence": det_score, "embedding": self.embed(img, landmark)})
        return output

    def recognition_face(self, embedding, recognition_thresh=0.4):
        id_ref = -1
        distance = float('-inf')
        for face_save in self.list_face:
            sim = compute_sim([face_save["embedding"]], [embedding])
            if sim > recognition_thresh and sim > distance:
                id_ref = face_save["id"]
                distance = sim
        return id_ref, distance

    def register_face(self, id, embedding):
        index = next((index for (index, i) in enumerate(
            self.list_face) if i["id"] == id), None)
        if index is not None:
            if len(self.list_face[index]["embedding"]) > self.ref_face:
                self.list_face[index]["embedding"].pop(0)
            self.list_face[index]["embedding"].append(embedding)

    def load_face_dict(self, path):
        try:
            self.list_face = load_pickle(path)
        except Exception as err:
            print(err)

    def load_face_door(self, path):
        try:
            dict_face = load_pickle(path)
            for i in range(len(dict_face["encode"])):
                self.list_face.append(
                    {"id": dict_face["name"][i], "embedding": dict_face["encode"][i]})
        except Exception as err:
            print(err)

    def list_face_len(self):
        return len(self.list_face)

    def save_face(self,path,name):
        save_pickle(path,self.list_face,name)

        
    @property
    def get_list_face(self):
        return self.list_face
