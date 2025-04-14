import torch  
from ultralytics import YOLO  
import multiprocessing  
import os  

def verify_dataset_structure(dataset_path):
    """Verify YOLO dataset directory structure"""  
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']  
    missing_dirs = []  
    for dir_path in required_dirs:  
        full_path = os.path.join(dataset_path, dir_path)  
        if not os.path.exists(full_path):  
            missing_dirs.append(full_path)  
    
    if missing_dirs:  
        print("Missing directories:")  
        for missing_dir in missing_dirs:  
            print(f"- {missing_dir}")  
        return False  
    return True  

if __name__ == '__main__':  
    # Multiprocessing support for Windows  
    multiprocessing.freeze_support()  

    # Dataset path  
    dataset_path = "D:/DATA/FOR_DL/objectdetection/magang2025/PuduRobot"  

    # Verify dataset structure  
    if not verify_dataset_structure(dataset_path):  
        print("Dataset structure is incomplete. Please check your directories.")
        exit(1)  

    # CUDA availability  
    print("CUDA Available:", torch.cuda.is_available())  

    # Device selection  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    print(f"Training on: {device}")  

    try:  
        # Load YOLO model  
        model = YOLO("yolov8n.pt")  

        # Training configuration  
        model.train( 
            data=os.path.join(dataset_path, "D:/DATA/FOR_DL/objectdetection/magang2025/PuduRobot/data.yaml"),  # Path to dataset.yaml  
            epochs=50,  # Increased epochs  
            imgsz=640,  
            batch=32,  # Adjusted batch size  
            device=device,  
            workers=4,  # Increased workers for faster data loading  
            patience=20,  # Adjusted patience for early stopping  
            plots=True,  
            save=True,  
            verbose=True  
        )

        # Save trained model  
        model.save("trainedPuduRobot_yolov8n.pt")  

        # Validate model  
        results = model.val()  
        print(results)  

    except Exception as e:  
        print(f"An error occurred during training: {e}")