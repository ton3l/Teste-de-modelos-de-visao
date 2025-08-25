from threading import Thread
import cv2

class Watcher:
    def __init__(self, stream_url):
        self.stream = cv2.VideoCapture(stream_url)
        self.ok, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def stop(self):
        self.stopped = True

    def get_frame(self):
        return self.frame
    
    def update(self):
        while not self.stopped:
            self.ok, self.frame = self.stream.read()