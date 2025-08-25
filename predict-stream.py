import cv2
from ultralytics import YOLO
import time # Usaremos para controlar o FPS

def main():
    # --- CONFIGURAÇÃO ---
    model = YOLO('balancaReader.pt')
    
    # DEFINA O SEU FPS ALVO AQUI
    TARGET_FPS = 10

    # Calcula o tempo que cada quadro deve levar
    TARGET_FRAME_DURATION = 1 / TARGET_FPS

    # Altere a fonte do vídeo conforme sua necessidade
    fonte_do_video = '' 
    
    # --------------------

    cap = cv2.VideoCapture(fonte_do_video)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir a fonte de vídeo: {fonte_do_video}")
        return

    while True:
        # NOVO: Marca o tempo de início do processamento do quadro
        start_time = time.time()

        success, frame = cap.read()
        if not success:
            print("Fim do stream de vídeo ou erro na captura.")
            break

        results = model.predict(frame, stream=True, verbose=False, conf=0.3, iou=0.5) 

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                class_name = model.names[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{class_name} {conf}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Leitura de Stream - YOLOv8 (FPS Limitado)", frame)

        # --- LÓGICA DE CONTROLE DE FPS ---
        # Calcula quanto tempo o processamento do quadro levou
        processing_time = time.time() - start_time
        
        # Calcula o tempo que precisamos esperar para atingir o FPS alvo
        wait_time = TARGET_FRAME_DURATION - processing_time

        # Converte o tempo de espera para milissegundos para o cv2.waitKey
        wait_ms = int(wait_time * 1000)

        # Se o processamento foi mais rápido que o alvo, esperamos.
        # Se foi mais lento, não esperamos nada (wait_ms será <= 0).
        if wait_ms < 1:
            wait_ms = 1 # waitKey não pode ter valor 0, o mínimo é 1.
        
        # Espera o tempo calculado E verifica se a tecla 'q' foi pressionada
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break
        # --- FIM DA LÓGICA DE CONTROLE ---

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()