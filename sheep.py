import cv2
from ultralytics import YOLO
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

model = YOLO('resources/weights/yolov8m-sheep.pt')
unique_id = set()

class CameraStream:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.Subscriber("/uav1/camera1/image_raw", Image, self.callback)

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        results = model.track(frame, tracker="bytetrack.yaml", persist=True)
        img = results[0].plot()

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            unique_id.update(ids)

        cv2.putText(img, f'Sheep Count: {len(unique_id)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera Stream', img)
        cv2.waitKey(3)

def main():
    rospy.init_node('sheep_detector', anonymous=True)
    CameraStream()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
