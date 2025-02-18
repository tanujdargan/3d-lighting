import cv2
import torch
import numpy as np
import open3d as o3d
import threading
import queue

# Set up GPU acceleration if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the MiDaS model for depth estimation.
model_type = "DPT_Large"  # For faster inference, you might try "MiDaS_small"
model = torch.hub.load("intel-isl/MiDaS", model_type)
model.to(device)
model.eval()

# Load the proper transforms for preprocessing.
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Set up OpenCV video capture.
# Replace 'video.mp4' with 0 for a webcam input.
cap = cv2.VideoCapture('video.mp4')

# Create a queue (maxsize=1) to hold the latest frame.
frame_queue = queue.Queue(maxsize=1)

# Thread function for continuous frame capture.
def capture_frames(cap, frame_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Ensure only the newest frame is in the queue.
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)
    cap.release()

# Start the capture thread.
capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue))
capture_thread.daemon = True
capture_thread.start()

# Set up Open3D visualizer for the 3D point cloud.
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Reconstruction", width=640, height=480)
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

while True:
    try:
        frame = frame_queue.get(timeout=1)
    except queue.Empty:
        continue

    # Convert frame color from BGR (OpenCV default) to RGB.
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply the transform and send image tensor to the GPU.
    input_batch = transform(img).to(device)

    # Inference on depth map.
    with torch.no_grad():
        prediction = model(input_batch)
        # Resize prediction back to the original image dimensions.
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Debug prints to see statistics for the depth map.
    print("Depth map stats -- min: {:.2f}, max: {:.2f}, mean: {:.2f}".format(
        np.min(depth_map), np.max(depth_map), np.mean(depth_map)
    ))

    # Optional: visualize the raw depth map with a colormap.
    depth_vis = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    depth_vis = np.uint8(depth_vis)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # --- 3D Reconstruction ---
    h, w = depth_map.shape
    focal_length = w  # Approximation; adjust as needed.
    cx, cy = w / 2, h / 2
    i_range, j_range = np.indices((h, w))
    
    # Compute raw 3D coordinates using a simple pinhole camera model.
    z = depth_map
    x = (j_range - cx) * z / focal_length
    y = (i_range - cy) * z / focal_length
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = cv2.resize(img, (w, h)).reshape(-1, 3) / 255.0

    # Filter out invalid depth values:
    # Here we only accept points with depth greater than a threshold (e.g., > 0 or >0.5)
    # Adjust the threshold based on the model and scene.
    valid_mask = points[:, 2] > 0.5  # Change this threshold if needed.
    valid_points = points[valid_mask]
    valid_colors = colors[valid_mask]

    if valid_points.size == 0:
        print("No valid 3D points this frame; skipping update.")
    else:
        # Update the Open3D point cloud.
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        pcd.colors = o3d.utility.Vector3dVector(valid_colors)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    # Display the current video frame and depth map.
    cv2.imshow("Video Input", frame)
    cv2.imshow("Depth Map", depth_vis)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vis.destroy_window()