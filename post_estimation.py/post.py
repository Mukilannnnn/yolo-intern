from trackers.Tracker import Tracker
import cv2

tracker = Tracker('models/best.pt')
image = cv2.imread(r'C:\Users\umuki\OneDrive\Desktop\intern\cb\trainig\Players,-referee,-gates-hockey-10\Players,-referee,-gates-hockey-10\train\images\-18-12-2022-13-13_000_jpg.rf.4a58209a473048856179bc0d49f2e6b2.jpg')
detections = tracker.detect_objects(image)
print(detections)
