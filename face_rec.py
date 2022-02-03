from re import I
import face_recognition
import numpy as np
from sklearn import svm
import os
import pandas as pd
import cv2
import sys


def extractFrames(clf,vflag, pathIn=""):
    videos = os.listdir(pathIn)
    #print(videos)
    dic_faces={}
    pi=pathIn
    pathOut="./Frames"
    #po=pathOut

    for i in videos:
        print(i)
        
        pathIn=pathIn+"/"+i
        if int(vflag) != 0:
            cap = cv2.VideoCapture(pathIn)
        else:
            cap = cv2.VideoCapture(0)
        count = 0
 
        while (cap.isOpened()):
            ret, frame = cap.read()
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                                                            
            
 
            if ret == True:
                #print('Read %d frame: ' % count, ret)
                pathF=os.path.join(pathOut, "frame{:d}.jpg".format(count))
                cv2.imwrite(pathF, frame)  # save frame as JPEG file
                dic_faces,face_loc=test(pathF,clf,count,dic_faces)
                count += 1
            else:
                break
            color=(0,255,0)
            thickness=3

            for i in face_loc:
                #print(i)
                start_point=(i[3],i[0])
                end_point=(i[1],i[2])
                frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            
    # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        pathIn=pi
        
    df=pd.DataFrame()
    df['frames']=list(dic_faces.keys())
    df['Person']=list(dic_faces.values())
    print(df)
    df.to_csv('faces.csv')
        


def train(train_path):
    encoding=[]
    names=[]
    train_dir = os.listdir(train_path)

    for person in train_dir:
        
        pix = os.listdir(train_path + person)
  
        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(
                train_path + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)
  
            # If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image 
                # with corresponding label (name) to the training data
                encoding.append(face_enc)
                names.append(person)
            else:
                print(person + "/" + person_img + " can't be used for training")
    clf = svm.SVC(gamma ='scale',C=1.5,probability=True)
    clf.fit(encoding, names)

    return clf

def test(test_image,clf,count,dic_faces):
    test_image1 = face_recognition.load_image_file(test_image)
  
    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image1)
    no = len(face_locations)
    print(count)
    print("Number of faces detected: ", no)
    classes=list(clf.classes_)
    # Predict all the faces in the test image using the trained classifier
    print("Found:")
    names=[]
    countF=1
    pathF="./Allface/"
    for i in range(no):
        pathc=pathF+str(count)+"_"+str(countF)+".jpg"
        #print(face_locations)
        face_img=test_image1[face_locations[i][0]:face_locations[i][2] , face_locations[i][3]:face_locations[i][1]]
        cv2.imwrite(pathc,face_img)
        test_image_enc = face_recognition.face_encodings(test_image1)[i]
        name = clf.predict_proba([test_image_enc])
        mvalue=max(list(name[0]))
        
        if mvalue<0.5:
            fname="unknown"
        else:
            fname=classes[np.argmax(name[0])]
        if fname not in names:
            names.append(fname)
        countF+=1
    print(names)
    dic_faces[count]=",".join(names)
    return dic_faces,face_locations

  
def main(flag):
    #args = docopt.docopt(__doc__)
    #train_dir = args["--train_dir"]
    #pathIn = args["--test_video"]
    #pathOut = args["--"]
    train_dir="./Faces/"
    pathIn="./video/"
    clf=train(train_dir)
    extractFrames(clf,flag,pathIn)
    
  
if __name__=="__main__":
    flag=sys.argv[1]
    
    main(flag)