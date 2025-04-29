import pyrealsense2 as rs
import numpy as np
import cv2
import os

def preprocess_bag(input_bag, output_dir):
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, input_bag, repeat_playback=False)

    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Get actual intrinsics from the first frame
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    
    # Save intrinsics to file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(f"{output_dir}/intrinsics.txt", 
               np.array([intr.fx, intr.fy, intr.ppx, intr.ppy]))

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    frame_count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            if not frames:
                break

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Get raw data
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Apply filters to depth
            filtered_depth = spatial.process(depth_frame)
            filtered_depth = temporal.process(filtered_depth)
            filtered_depth = hole_filling.process(filtered_depth)
            depth_image = np.asanyarray(filtered_depth.get_data())

            # Mild color enhancement
            color_image_enhanced = cv2.convertScaleAbs(color_image, alpha=1.1, beta=5)
            
            # Save raw depth as uint16 PNG (lossless)
            cv2.imwrite(f"{output_dir}/color_{frame_count:06d}.png", color_image_enhanced)
            cv2.imwrite(f"{output_dir}/depth_{frame_count:06d}.png", depth_image)
            frame_count += 1

    finally:
        pipeline.stop()
        print(f"Preprocessed {frame_count} frames saved to {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_bag", help="Path to input .bag file")
    parser.add_argument("output_dir", help="Directory to save preprocessed frames")
    args = parser.parse_args()
    preprocess_bag(args.input_bag, args.output_dir)