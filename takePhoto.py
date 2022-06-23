import cv2
import pdb

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
imgNo = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        break
    elif k%256 == 32:
        imgName = "test" + str(imgNo) + ".jpg"
        cv2.imwrite(imgName, frame)
        print("{} written!".format(imgName))
        imgNo += 1
		
	

cam.release()
cv2.destroyAllWindows()
pdb.set_trace()