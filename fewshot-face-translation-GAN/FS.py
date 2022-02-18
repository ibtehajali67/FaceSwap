import warnings
from models import FaceTranslationGANInferenceModel
from face_toolbox_keras.models.verifier.face_verifier import FaceVerifier
from face_toolbox_keras.models.parser import face_parser
from face_toolbox_keras.models.detector import face_detector
from face_toolbox_keras.models.detector.iris_detector import IrisDetector
import numpy as np
from utils import utils
from matplotlib import pyplot as plt
import cv2


class FaceSwap():
    def __init__(self):
        warnings.filterwarnings("ignore")
        self.model = FaceTranslationGANInferenceModel()
        self.fv = FaceVerifier(classes=512)
        self.fp = face_parser.FaceParser()
        self.fd = face_detector.FaceAlignmentDetector()
        self.idet = IrisDetector()

    def Prediction(self,fn_src,fn_tar):
        src, mask, aligned_im, (x0, y0, x1, y1), landmarks,M = utils.get_src_inputs(fn_src, self.fd, self.fp, self.idet)
        tar, emb_tar = utils.get_tar_inputs(fn_tar, self.fd, self.fv)
        out = self.model.inference(src, mask, tar, emb_tar)
        result_face = np.squeeze(((out[0] + 1) * 255 / 2).astype(np.uint8))
        result_img = utils.post_process_result(fn_src, self.fd, result_face, aligned_im, src, x0, y0, x1, y1, landmarks,M)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        return result_img


