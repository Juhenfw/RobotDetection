import cv2
from ultralytics import YOLO
import argparse
import time

def test_model_on_video(video_path, model_path, conf_thresh=0.25, save_output=True):
    # Load YOLOv8 model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Open the video file
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Initialize video writer if saving output
    output_path = "output_" + video_path.split("/")[-1] if "/" in video_path else "output_" + video_path
    out = None
    if save_output:
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
        results = model(frame, conf=conf_thresh)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the processing speed
        processing_fps = frame_count / (time.time() - start_time)
        cv2.putText(annotated_frame, f"Processing: {processing_fps:.1f} FPS", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write the frame to the output video
        if save_output and out is not None:
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
    parser = argparse.ArgumentParser(description="Test YOLOv8 model on video file")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLOv8 model (default: best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--no-save", action="store_true", help="Don't save output video")
    
    args = parser.parse_args()
    
    test_model_on_video(
        video_path=args.video,
        model_path=args.model,
        conf_thresh=args.conf,
        save_output=not args.no_save
    )
