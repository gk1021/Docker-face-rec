import cv2
import os
import docopt
 
def extractFrames(pathIn, pathOut):
    videos = os.listdir(pathIn)
    #print(videos)
    pi=pathIn
    po=pathOut
    for i in videos:
        print(i)
        
        pathIn=pathIn+"/"+i
        name=i.split(".mp4")

        
        pathOut=pathOut+"/"+name[0]

        os.mkdir(pathOut)
 
        cap = cv2.VideoCapture(pathIn)
        count = 0
 
        while (cap.isOpened()):
 
            # Capture frame-by-frame
            ret, frame = cap.read()
 
            if ret == True:
                print('Read %d frame: ' % count, ret)
                cv2.imwrite(os.path.join(pathOut, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
                count += 1
            else:
                break
 
    # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        pathIn=pi
        pathOut=po
     
 
def main():
    args = docopt.docopt(__doc__)
    pathIn = args["--Video"]
    pathOut = args["--Frames"]
    extractFrames(pathIn, pathOut)
 
if __name__=="__main__":
    main()