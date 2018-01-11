import cv2
import numpy as np
import sqlite3
import datetime

facereg = cv2.createLBPHFaceRecognizer()
facereg.load('facetrainner/facetrainners.yml')
cascadeofpath = "haarcascade_frontalface_default.xml"
faceofCascade = cv2.CascadeClassifier(cascadeofpath);

photo = cv2.imread('shao3.jpg')
fontshow = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
photogray=cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
facedect =faceofCascade.detectMultiScale(photogray, 1.1, 5)
for (x, y, w, h) in facedect:
    cv2.rectangle(photo, (x, y), (x + w, y + h), (255, 0, 0), 2)
    grayset = photogray[y:y + h, x:x + w]
    colorset = photo[y:y + h, x:x + w]
    Id, conf = facereg.predict(photogray[y:y+h,x:x+w])
    if(conf<90):
        if(Id==1):
            Id="GW41290951"
            datab = sqlite3.connect('signtable.db')
            datab.row_factory = sqlite3.Row
            datab.execute('drop table if exists signtable')
            datab.execute('create table signtable(n1 signtable, m1 int)')
            datab.execute('insert into signtable (n1, m1) values (?, ?)', ('shaowenyuan', 41290951))
            datab.commit()
            cur = datab.execute('select * from signtable')
            timenow = datetime.datetime.now()
            for name in cur:
                print(name['n1'], name['m1'])
                print "Sign Time: " + timenow.strftime('%Y.%m.%d-%H:%M:%S')
    else:
        Id="Unknown"
    cv2.cv.PutText(cv2.cv.fromarray(photo),str(Id), (x,y+h),fontshow, 255)

cv2.imshow('photo', photo)
cv2.waitKey(0)
cv2.destroyAllWindows()