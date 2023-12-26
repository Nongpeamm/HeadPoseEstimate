from load_model import YoloDetect
class DetectFasion:
    def __init__(self, person_model, fasion_model):
        self.person_model = person_model
        self.fasion_model = fasion_model

    def person_detect(self, image):
        return YoloDetect(image, self.person_model, classes=[0], conf=0.4, want_track_id=True) # return person images

    def fashion_detect(self, image): # image is person image
        #if image is array of person images
        if isinstance(image, list):
            for person_img in image:
                YoloDetect(person_img, self.fasion_model, conf=0.7)
        else:
            YoloDetect(image, self.fasion_model, conf=0.7)
    

class DetectOnFace:
    def __init__(self, face_model, gender_model):
        self.face_model = face_model
        self.gender_model = gender_model

    def face_detect(self, image):
        return self.face_model(image)
    
    def gender_detect(self, image):
        return self.gender_model(image)