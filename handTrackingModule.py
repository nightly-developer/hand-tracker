
import mediapipe as mp
import cv2

class HandTracker():
  def __inid__ (self,mode=False,maxhands=2,detectionCon=0.5,trackCon=0.5):
    self.mode = mode 
    self.maxhands = maxhands 
    self.detectionCon = detectionCon 
    self.trackCon = trackCon 

    self.mpHands = mp.solutions.hands
    self.hands = self.mpHands.Hands(self.mode,self.maxhands,
                                    self.detectionCon,self.trackCon)
    self.mpDraw = mp.solutions.drawing_utils
  
  def trackHands(self,img,draw=True):
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = self.hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
      for handLMS in results.multi_hand_landmarks:
        if draw:
          self.mpDraw.draw_landmarks(img,handLMS,self.mpHands.HAND_CONNECTIONS)
    return img



def main():
  pTime = 0
  cTime = 0
  cap = cv2.VideoCapture(0)
  tracker = HandTracker()
  while True:
    success, img = cap.read()
    img = tracker.trackHands(img)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3)
  cv2.imshow("Image",img)
  key = cv2.waitKey(1)
  if key == 27:
    cv2.destroyAllWindows()

main()
