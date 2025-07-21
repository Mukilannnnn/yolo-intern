import cv2

# Function to read video and capture frames
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return []
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    print(f"Number of frames captured: {len(frames)}")  # Debugging line to check frame count
    cap.release()  # Release the capture object when done
    return frames

# Function to save frames as a video
def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        print("No frames to save.")  # If no frames were passed
        return  # Early exit if no frames are available
    
    # Access the last frame's dimensions for width and height
    last_frame = output_video_frames[-1]
    frame_height, frame_width = last_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (frame_width, frame_height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()  # Release the video writer when done

# Main function to execute the video read and save operations
def main():
    video_path = r'C:\Users\umuki\OneDrive\Desktop\intern\cb\input_videos\Untitled video - Made with Clipchamp.mp4'  # Specify the correct path to your video
    output_video_path = 'output_videos/output_video.avi'  # Specify the output video path
    
    # Read frames from the video
    video_frames = read_video(video_path)
    
    if video_frames:  # Only save if frames are captured
        save_video(video_frames, output_video_path)
    else:
        print("No frames to process.")

# Run the main function
if __name__ == "__main__":
    main()
