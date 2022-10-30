import cv2
import torch
import time


def load_model():
    print("[INFO] loading model...")
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def get_video_stream():
    print("[INFO] starting video stream...")
    return cv2.VideoCapture(0) # 0 means read from local camera.


class ObjectDetection:
    
    def __init__(self):
        self.model = load_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __call__(self):
        player = get_video_stream()
        assert player.isOpened()
        time.sleep(2.0)
        
        ret, frame = player.read()
        
        while ret:
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            
            if frame is None:
                ret, frame = player.read()
                continue
            
            cv2.imshow("Frame", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            ret, frame = player.read()
    
        cv2.destroyAllWindows()
        player.stop()

    def score_frame(self, frame):
        self.model.to(self.device)
        results = self.model(frame)
        labels = results.xyxyn[0][:, -1].numpy()
        cord = results.xyxyn[0][:, :-1].numpy()
        
        return labels, cord

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            # If score is less than 0.2 we avoid making a prediction.
            if row[4] < 0.2: 
                continue
            x1 = int(row[0]*x_shape)
            y1 = int(row[1]*y_shape)
            x2 = int(row[2]*x_shape)
            y2 = int(row[3]*y_shape)
            bgr = (0, 255, 0) # color of the box
            classes = self.model.names # Get the name of label index
            label_font = cv2.FONT_HERSHEY_SIMPLEX #Font for the label.
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2) #Plot the boxes
            cv2.putText(frame,
                        classes[labels[i]],
                        (x1, y1),
                        label_font, 0.9, bgr, 2) #Put a label over box.
            
        return frame


det = ObjectDetection()
det()
