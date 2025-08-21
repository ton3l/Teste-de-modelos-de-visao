from ultralytics import YOLO
from PIL import Image

model = YOLO('balancaReader.pt')
image_path = 'original.png'

print('------------------------------------------')
print('Reading image:', image_path)
results = model(image_path)
print('Finish reading')
print('------------------------------------------')

# Processa os resultados da detecção.
for result in results:
    boxes = result.boxes  # Boxes object for bounding boxes
    for box in boxes:
        # Obtém as coordenadas da caixa delimitadora no formato xyxy (x1, y1, x2, y2).
        xyxy = box.xyxy
        # Obtém a probabilidade da detecção.
        conf = box.conf
        # Obtém o índice da classe detectada.
        cls = box.cls

        print(f"Box: {xyxy}, Confiança: {conf}, Classe: {cls}")

    # Plota os resultados na imagem.
    plot_img = result.plot()
    image = Image.fromarray(plot_img)
    image.show()

    # Salva a imagem com as detecções (opcional).
    image.save('resultado_deteccao.jpg')
