import os
import cv2
import numpy as np
from utils import read_video, save_video
from trackers.Tracker import Tracker
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def main():
    os.makedirs('output_videos', exist_ok=True)

    # Read video frames
    video_frames = read_video(r'C:\Users\umuki\OneDrive\Desktop\intern\cb\input_videos\Untitled video - Made with Clipchamp.mp4')
    print(f"Number of frames captured: {len(video_frames)}")

    tracker = Tracker('models/best.pt')

    # Get object tracks
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )
    print("Tracks:", tracks)

    output_video_frames = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for frame_idx, frame in enumerate(video_frames):
            annotated_frame = frame.copy()

            # Get tracks for the current frame
            for track in tracks.get(frame_idx, []):
                x1, y1, x2, y2 = track['bbox']
                class_id = track['class_id']

                print(f"Frame {frame_idx}: bbox=({x1},{y1},{x2},{y2}), class_id={class_id}")

                if x2 <= x1 or y2 <= y1:
                    print(f"Invalid bbox for frame {frame_idx}: {track['bbox']}")
                    continue

                cropped_frame = frame[y1:y2, x1:x2]
                if cropped_frame.size == 0:
                    print(f"Empty crop for bbox {track['bbox']} in Frame {frame_idx}")
                    continue

                cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                result = pose.process(cropped_frame_rgb)

                if result.pose_landmarks:
                    print(f"Pose detected in Frame {frame_idx}")
                    mp_drawing.draw_landmarks(
                        cropped_frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                else:
                    print(f"No pose detected in Frame {frame_idx}")

                annotated_frame[y1:y2, x1:x2] = cropped_frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame, f"Class {class_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                )

            output_video_frames.append(annotated_frame)

    save_video(output_video_frames, 'output_videos/output_video_with_pose.avi')
    print("Annotated video saved to 'output_videos/output_video_with_pose.avi'")

if __name__ == '__main__':
    main()
