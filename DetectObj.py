from ultralytics import YOLO
import cv2
import time
import numpy as np

class ObjectDetector:
    def __init__(self, model_path, conf_thresh=0.5):
        """
        Inisialisasi detector
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_thresh
        self.colors = self._generate_colors()
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def _generate_colors(self):
        """
        Generate warna random untuk setiap kelas
        """
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(100, 3)).tolist()
        return colors

    def _calculate_fps(self):
        """
        Hitung FPS
        """
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps

    def process_frame(self, frame):
        """
        Proses frame dan tampilkan hasil deteksi
        """
        # Jalankan deteksi
        results = self.model(frame, conf=self.conf_threshold)
        
        # Hitung FPS
        fps = self._calculate_fps()
        
        # Visualisasi hasil
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Koordinat bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Confidence score
                conf = float(box.conf[0])
                
                # Class ID dan nama
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                
                # Pilih warna untuk kelas ini
                color = self.colors[cls_id % len(self.colors)]
                
                # Gambar box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Gambar label dengan background
                label = f'{cls_name} {conf:.2f}'
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                cv2.rectangle(frame, (x1, y1-label_height-10), 
                            (x1+label_width, y1), color, -1)
                cv2.putText(frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        # Tampilkan FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

    def run_webcam(self):
        """
        Jalankan deteksi pada webcam
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Gagal membaca frame")
                    break

                # Proses frame
                processed_frame = self.process_frame(frame)
                
                # Tampilkan hasil
                cv2.imshow("YOLOv8 Detection", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def process_image(self, image_path):
        """
        Proses satu gambar
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("Tidak dapat membaca gambar")
            
        processed_frame = self.process_frame(frame)
        return processed_frame

# Penggunaan
if __name__ == "__main__":
    # Inisialisasi detector
    detector = ObjectDetector(
        model_path='path/to/your/best.pt',  # Ganti dengan path model Anda
        conf_thresh=0.5
    )
    
    # Pilih mode yang diinginkan:
    
    # 1. Mode Webcam
    detector.run_webcam()
    
    # 2. Mode Image (uncomment untuk menggunakan)
    # image = detector.process_image('path/to/your/image.jpg')
    # cv2.imshow("Result", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
