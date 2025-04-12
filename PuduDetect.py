import cv2
from ultralytics import YOLO
import time
import os

# ========== TETAPKAN PATH ANDA DI SINI ==========
# Path ke file video yang akan diproses
VIDEO_PATH = "C:/Users/YourName/Videos/your_video.mp4"  # GANTI DENGAN PATH VIDEO ANDA
# Path ke model YOLOv8
MODEL_PATH = "best.pt"  # Jika file best.pt di folder yang sama dengan script ini
# Confidence threshold
CONFIDENCE_THRESHOLD = 0.25
# Apakah akan menyimpan output video
SAVE_OUTPUT = True
# ================================================

def test_model_on_video():
    # Cek apakah file video ada
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: File video tidak ditemukan: {VIDEO_PATH}")
        return
    
    # Load YOLOv8 model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Open the video file
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {VIDEO_PATH}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Initialize video writer if saving output
    output_path = "output_" + os.path.basename(VIDEO_PATH)
    out = None
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    # Process the video
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
        
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the processing speed
        processing_fps = frame_count / (time.time() - start_time)
        cv2.putText(annotated_frame, f"Processing: {processing_fps:.1f} FPS", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write the frame to the output video
        if SAVE_OUTPUT and out is not None:
            out.write(annotated_frame)
        
        # Display the frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\nProcessing completed!")
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average processing speed: {frame_count/total_time:.2f} FPS")

if __name__ == "__main__":
    test_model_on_video()
