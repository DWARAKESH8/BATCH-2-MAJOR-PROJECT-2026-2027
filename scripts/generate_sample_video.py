import cv2
import numpy as np
from pathlib import Path

def create_sample_crowd_video(output_path: str = "sample_crowd.mp4", duration_sec: int = 5, fps: int = 25):
    width, height = 640, 480
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    np.random.seed(42)
    num_people = 35
    # Initialize positions and velocities for people
    xs = np.random.uniform(50, width - 50, num_people)
    ys = np.random.uniform(50, height - 50, num_people)
    vxs = np.random.uniform(-3, 3, num_people)
    vys = np.random.uniform(-3, 3, num_people)

    for f in range(total_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 220 # light background
        # Add background grid/zone
        cv2.rectangle(frame, (int(0.3 * width), int(0.2 * height)), (int(0.7 * width), int(0.85 * height)), (200, 200, 240), 2)
        cv2.putText(frame, "Monitored Bottleneck Zone", (int(0.3 * width) + 10, int(0.2 * height) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 100, 100), 1)

        # Update positions
        xs += vxs
        ys += vys

        # Bounce off walls
        for i in range(num_people):
            if xs[i] < 30 or xs[i] > width - 30:
                vxs[i] *= -1
            if ys[i] < 30 or ys[i] > height - 30:
                vys[i] *= -1

            # Draw person representation (head circle + torso box)
            x, y = int(xs[i]), int(ys[i])
            cv2.circle(frame, (x, y - 15), 8, (50, 50, 200), -1) # Head
            cv2.rectangle(frame, (x - 10, y - 5), (x + 10, y + 25), (200, 50, 50), -1) # Torso

        out.write(frame)

    out.release()
    print(f"Sample crowd video created: {output_path} ({total_frames} frames)")

if __name__ == "__main__":
    create_sample_crowd_video()
