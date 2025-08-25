from services.Watcher import Watcher
from services.Reader import Reader
import dotenv
import time

STREAM_URL = dotenv.get_key(dotenv.find_dotenv(), 'STREAM_URL')

class Monitoring:
    def __init__(self, stream_url):
        self.watcher = Watcher(stream_url).start()
        self.reader = Reader()
        self.delay = 1

    def start(self):
        try:
            self.__main_loop()
        finally:
            self.watcher.stop()

    def __main_loop(self):
        while True:
            time.sleep(self.delay)
            frame = self.watcher.get_frame()
            result = self.reader.get_frame_text(frame, save_image=True)

            print(result)

if __name__ == "__main__":
    monitoring = Monitoring(STREAM_URL)
    monitoring.start()