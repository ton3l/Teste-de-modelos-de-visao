from ultralytics import YOLO
import cv2

video = cv2.VideoCapture('')

model = YOLO('balancaReader.pt')

while True:
    confirm, frame = video.read()
    if not confirm:
        print('Erro ao capturar o frame.')
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes
        result.save('resultado_deteccao.jpg')
        print('\n', result.to_json(), '\n')
