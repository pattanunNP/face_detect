import os,json
from BaseModel.FaceNet import FaceNet
from Utils.Prepare import  Prepare
import Config as ENV
import  pendulum
import json,time
from annoy import  AnnoyIndex
from tqdm import  tqdm

class FaceRecognition:

    def __init__(self):
        model = FaceNet()
        self.Prepare = Prepare()
        self.encoder = model.loadModel(ENV.Facenet)
        self.model = AnnoyIndex(128, "euclidean")
        self.model.load('./database.tree')
        # self.model = model.load(ENV.Model)
        database = open('./database.json', encoding="utf-8")
        # print(database)
        self.database = json.load(database)
    
    def encode_face_data(self, face):
     
        # face = self.Prepare.preprocess_input(path)
        embedding_face = self.encoder.predict(face)[0, :]
        # print(embedding_face.shape)
        return embedding_face
    
    def encode_face_data2(self, path):
     
        face = self.Prepare.preprocess_input(path)
        embedding_face = self.encoder.predict(face)[0, :]
        # print(embedding_face.shape)
        return embedding_face

    def add_face(self,path):
        dataset = []
        reprensentative = []
        datablock   = {}
        tree = AnnoyIndex(128, "euclidean")
        for i,fname in tqdm(enumerate(os.listdir(path)), total=len(os.listdir(path))):
            if fname.endswith('.jpg') or fname.endswith('png') or fname.endswith('.jpeg'):
                print(fname)
                timestamp = pendulum.now('Asia/Bangkok')
                real_name = fname.replace('_', ' ').split('.')[0]
                embedding = FaceRecognition().encode_face_data2(f"{path+fname}").tolist()
                vector = embedding
                tree.add_item(i,vector)
                datablock = {
                    "index":i,
                    "student_id":"",
                    "name": f"{real_name}",
                    "nickname":"",
                    "class": "6/3",
                    "image_path":f"{path+fname}",
                    "added_time": f"{timestamp}",
                }
            dataset.append(datablock)
        tree.build(3)
        tree.save('database.tree')
        with open('database.json', 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        print("build done")

    def match(self, face_encoded, n_of_similarity=1, include_distances=True):
        tic = time.clock()
        idx = self.model.get_nns_by_vector(face_encoded, n_of_similarity, include_distances=True)
        toc = time.clock()
        used_time = toc - tic
        return idx

    def getface(self, idx,theshold=8):
        result = {}
        for matched, dist in zip(idx[0], idx[1]):
            print(matched,dist)
            if dist <= theshold:
                result = self.database[matched]
            else:
                result = {
                        "index":-1,
                        "student_id":"unknown",
                        "name": f"unknown",
                        "nickname":"unknown",
                        "class": "unknown",
                        "image_path":f"unknown",
                        "added_time": f"unknown"
                    }
        return result

        



        
