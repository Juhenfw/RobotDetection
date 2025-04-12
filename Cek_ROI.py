import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load model YOLOv8
model = YOLO("yolov8n.pt")  # Model paling ringan untuk performa real-time

# Setup kamera dan dapatkan resolusi
cap = cv2.VideoCapture(0)
success, test_frame = cap.read()
if not success:
    print("Tidak bisa mengakses kamera!")
    exit()

# Dapatkan dimensi frame dari webcam
frame_width = test_frame.shape[1]
frame_height = test_frame.shape[0]
print(f"Resolusi webcam: {frame_width}x{frame_height}")

# Sesuaikan checkpoint area berdasarkan resolusi webcam
checkpoints = {
    "Area 1": [int(frame_width*0.7), 0, frame_width, int(frame_height*0.3)],
    "Area 2": [int(frame_width*0.3), int(frame_height*0.3), int(frame_width*0.7), int(frame_height*0.7)],
    "Area 3": [0, int(frame_height*0.7), int(frame_width*0.3), frame_height]
}

# Inisialisasi warna untuk setiap area (B, G, R)
colors = {
    "Area 1": (0, 255, 0),    # Hijau
    "Area 2": (0, 255, 255),  # Kuning
    "Area 3": (255, 0, 0)     # Biru
}

# Untuk mencatat objek yang terdeteksi di setiap checkpoint
object_in_checkpoint = {area: set() for area in checkpoints}

# Kelas objek yang ingin dideteksi (kosongkan untuk mendeteksi semua objek)
# Contoh: target_classes = ["person", "car", "bicycle"]
target_classes = []  # Kosong berarti deteksi semua objek

# Function untuk mengecek apakah bbox objek berada di dalam checkpoint
def is_in_area(bbox, area):
    x1, y1, x2, y2 = bbox
    ax1, ay1, ax2, ay2 = area
    
    # Cek apakah tengah bounding box berada di dalam area
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    return (ax1 <= center_x <= ax2) and (ay1 <= center_y <= ay2)

# Fungsi untuk menggambar checkpoint pada frame
def draw_checkpoints(frame):
    for name, area in checkpoints.items():
        x1, y1, x2, y2 = area
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[name], 2)
        cv2.putText(frame, name, (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[name], 2)
    
    return frame

# Fungsi untuk mengonfigurasi checkpoint dengan klik mouse
def setup_checkpoints():
    global checkpoints
    setup_checkpoints = {}
    drawing = False
    point1 = (0, 0)
    area_count = 1
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, point1, area_count
        if event == cv2.EVENT_LBUTTONDOWN:
            point1 = (x, y)
            drawing = True
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            area_name = f"Area {area_count}"
            setup_checkpoints[area_name] = [
                min(point1[0], x),
                min(point1[1], y),
                max(point1[0], x),
                max(point1[1], y)
            ]
            area_count += 1
            drawing = False
    
    cv2.namedWindow("Setup Checkpoints")
    cv2.setMouseCallback("Setup Checkpoints", mouse_callback)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Tampilkan checkpoint yang sudah dibuat
        for name, area in setup_checkpoints.items():
            x1, y1, x2, y2 = area
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Tampilkan petunjuk
        cv2.putText(frame, "Klik dan drag untuk membuat area. 'q' untuk selesai, 'r' untuk reset", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Gambar area yang sedang dibuat
        if drawing:
            x, y = cv2.getMousePosition()
            cv2.rectangle(frame, point1, (x, y), (255, 0, 0), 2)
        
        cv2.imshow("Setup Checkpoints", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            setup_checkpoints = {}
            area_count = 1
    
    cv2.destroyWindow("Setup Checkpoints")
    
    if setup_checkpoints:
        return setup_checkpoints
    else:
        return checkpoints

# Tanyakan pengguna apakah ingin mengatur checkpoint melalui GUI
print("Atur checkpoint melalui GUI? (y/n)")
use_gui = input().lower() == 'y'

if use_gui:
    checkpoints = setup_checkpoints()

# Counter untuk FPS
start_time = time.time()
frame_count = 0

# Untuk mencatat kapan terakhir objek terdeteksi di checkpoint (mencegah spam notifikasi)
last_detection_time = {area: {} for area in checkpoints}
cooldown = 2  # Waktu cooldown dalam detik

# Loop utama
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Gagal membaca dari kamera.")
        break
    
    # Hitung FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    
    # Reset penghitung FPS setiap 5 detik untuk akurasi
    if elapsed_time > 5:
        start_time = time.time()
        frame_count = 0
    
    # Gambar checkpoint areas
    frame = draw_checkpoints(frame)
    
    # Tampilkan FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Jalankan deteksi YOLOv8
    results = model(frame, conf=0.4)  # Naikkan confidence threshold untuk performa lebih baik
    
    # Reset deteksi untuk frame ini
    current_detections = {area: set() for area in checkpoints}
    
    # Proses hasil deteksi
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            # Ekstrak bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Dapatkan nama kelas
            class_name = model.names[cls]
            
            # Filter kelas jika target_classes tidak kosong
            if target_classes and class_name not in target_classes:
                continue
            
            # Gambar bounding box untuk semua objek yang terdeteksi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # Buat ID unik untuk objek
            object_id = f"{class_name}_{x1}_{y1}"
            
            # Cek apakah objek berada di dalam checkpoint
            for area_name, area_coords in checkpoints.items():
                if is_in_area([x1, y1, x2, y2], area_coords):
                    current_detections[area_name].add(object_id)
                    
                    # Cek apakah objek baru terdeteksi atau sudah cukup lama sejak notifikasi terakhir
                    current_time = time.time()
                    last_time = last_detection_time[area_name].get(class_name, 0)
                    
                    if (object_id not in object_in_checkpoint[area_name] or 
                        (current_time - last_time) > cooldown):
                        print(f"Objek {class_name} memasuki {area_name} pada {time.strftime('%H:%M:%S')}")
                        last_detection_time[area_name][class_name] = current_time
                    
                    # Highlight objek yang ada di checkpoint
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colors[area_name], 3)
    
    # Update tabel deteksi
    object_in_checkpoint = current_detections
    
    # Tampilkan jumlah objek di setiap area
    y_offset = 60
    for area_name, detections in current_detections.items():
        count = len(detections)
        cv2.putText(frame, f"{area_name}: {count} objek", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[area_name], 2)
        y_offset += 30
    
    # Tampilkan frame
    cv2.imshow("Deteksi Objek dengan Checkpoint", frame)
    
    # Kontrol keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset checkpoint
        checkpoints = setup_checkpoints()
        object_in_checkpoint = {area: set() for area in checkpoints}
        last_detection_time = {area: {} for area in checkpoints}

cap.release()
cv2.destroyAllWindows()
