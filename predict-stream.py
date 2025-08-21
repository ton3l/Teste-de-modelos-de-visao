import cv2
from ultralytics import YOLO
import time # Para calcular o FPS

def main():
    # --- CONFIGURAÇÃO ---
    # Carrega o seu modelo treinado
    model = YOLO('balancaReader.pt')

    # Altere a fonte do vídeo conforme sua necessidade
    # Opção 1: Webcam
    # Use 0 para a primeira webcam, 1 para a segunda, etc.
    # fonte_do_video = 0 
    
    # Opção 2: Arquivo de Vídeo
    # fonte_do_video = 'caminho/para/seu_video.mp4'

    # Opção 3: Stream de Câmera IP (RTSP)
    fonte_do_video = 'rtsp://admin:SEG101085$$a@10.0.0.157:554/cam/realmonitor?channel=1&subtype=1'

    # --------------------

    # Inicia a captura do vídeo
    cap = cv2.VideoCapture(fonte_do_video)

    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir a fonte de vídeo: {fonte_do_video}")
        return

    # Variáveis para cálculo de FPS
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        # 1. Captura um quadro (frame) do vídeo
        success, frame = cap.read()

        if not success:
            print("Fim do stream de vídeo ou erro na captura.")
            break

        # 2. Faz a predição no quadro atual
        # O argumento stream=True otimiza o uso de memória para vídeos
        results = model(frame, stream=True, verbose=False, device='cpu')

        # 3. Processa os resultados e desenha na imagem
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Coordenadas
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Classe e Confiança
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Desenha o retângulo
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Escreve o texto
                label = f'{class_name} {conf}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 4. Calcula e exibe o FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # 5. Mostra o quadro processado em uma janela
        cv2.imshow("Leitura de Stream - YOLOv8", frame)

        # 6. Condição para sair do loop (pressione 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos ao finalizar
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()