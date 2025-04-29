import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import argparse
import os
from pathlib import Path
import random
import quaternion

class OrientationTracker:
    def __init__(self):
        self.orientation = np.quaternion(1, 0, 0, 0)
        self.gyro_bias = np.zeros(3)
        
    def update(self, gyro, dt):
        gyro_corrected = gyro - self.gyro_bias
        gyro_q = np.quaternion(0, *gyro_corrected)
        self.orientation += 0.5 * self.orientation * gyro_q * dt
        self.orientation = self.orientation.normalized()
        return self.orientation

class EnhancedSLAM:
    def __init__(self, intrinsics):
        self.intrinsics = intrinsics
        self.orientation_tracker = OrientationTracker()
        self.global_pcd = o3d.geometry.PointCloud()
        self.current_pose = np.eye(4)
        
        # Enhanced feature detection
        self.orb = cv2.ORB_create(2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_frame = None
        
        # Preprocessing parameters
        self.depth_scale = 1000.0  # mm to meters
        self.min_depth = 0.3       # 30cm minimum depth
        self.max_depth = 10.0      # 10m maximum depth
        
        # Mapping parameters
        self.voxel_size = 0.01     # 1cm voxel size
        self.icp_threshold = 0.1   # 10cm ICP threshold
        
        # Visualization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=1200, height=800)
        self.vis.add_geometry(self.global_pcd)

    def estimate_translation(self, color, depth):
        """Robust translation estimation using feature matching and depth"""
        if self.prev_frame is None:
            self.prev_frame = (color, depth)
            return None
            
        prev_color, prev_depth = self.prev_frame
        
        # Feature detection
        prev_gray = cv2.cvtColor(prev_color, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        prev_kp, prev_des = self.orb.detectAndCompute(prev_gray, None)
        curr_kp, curr_des = self.orb.detectAndCompute(curr_gray, None)
        
        # Early exit if not enough features
        if prev_des is None or curr_des is None or len(prev_des) < 50 or len(curr_des) < 50:
            self.prev_frame = (color, depth)
            return None
        
        # Feature matching with ratio test
        matches = self.matcher.match(prev_des, curr_des)
        matches = sorted(matches, key=lambda x: x.distance)[:200]  # Use best 200 matches
        
        # Get 3D-2D correspondences
        pts3d = []
        pts2d = []
        for m in matches:
            u, v = map(int, prev_kp[m.queryIdx].pt)
            z = prev_depth[v, u]
            if self.min_depth < z < self.max_depth:  # Valid depth
                # Convert to 3D in previous camera frame
                x = (u - self.intrinsics[2]) * z / self.intrinsics[0]
                y = (v - self.intrinsics[3]) * z / self.intrinsics[1]
                pts3d.append([x, y, z])
                pts2d.append(curr_kp[m.trainIdx].pt)
        
        if len(pts3d) < 20:  # Need at least 20 good matches
            self.prev_frame = (color, depth)
            return None
        
        # Convert to numpy arrays with proper types
        pts3d = np.array(pts3d, dtype=np.float32)
        pts2d = np.array(pts2d, dtype=np.float32)
        
        # Camera matrix
        camera_matrix = np.array([
            [self.intrinsics[0], 0, self.intrinsics[2]],
            [0, self.intrinsics[1], self.intrinsics[3]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Initial pose guess (using current orientation)
        rvec = R.from_matrix(self.current_pose[:3,:3]).as_rotvec().astype(np.float32)
        tvec = self.current_pose[:3,3].astype(np.float32)
        
        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pts3d,
            imagePoints=pts2d,
            cameraMatrix=camera_matrix,
            distCoeffs=None,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            iterationsCount=100,
            reprojectionError=2.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success or inliers is None or len(inliers) < 15:
            self.prev_frame = (color, depth)
            return None
        
        # Return translation difference
        self.prev_frame = (color, depth)
        return tvec.flatten() - self.current_pose[:3,3]
    
    def save_map(self, filename):
        """
        Save the current global point cloud to file
        Args:
            filename: Output file path (PLY, PCD, or other Open3D supported format)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        
        # Perform final optimization before saving
        self.global_pcd = self.global_pcd.voxel_down_sample(self.voxel_size)
        
        # Remove statistical outliers
        self.global_pcd, _ = self.global_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        
        if not self.global_pcd.has_normals():
            self.global_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.normal_estimation_radius, 
                    max_nn=30))
        
        o3d.io.write_point_cloud(filename, self.global_pcd)
        print(f"Map saved to {filename}")

    def preprocess_depth(self, depth):
        """Enhance depth data with careful preprocessing"""
        # Convert to float32 and scale
        depth = depth.astype(np.float32) / self.depth_scale
        
        # Apply bilateral filter for edge-preserving smoothing
        depth = cv2.bilateralFilter(depth, d=5, sigmaColor=0.5, sigmaSpace=1.5)
        
        # Clip to valid range
        return np.clip(depth, self.min_depth, self.max_depth)
    
    def preprocess_color(self, color):
        """Enhance color data"""
        # Convert to LAB color space for better contrast
        lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
        
        # CLAHE for adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def process_frame(self, color, depth, gyro=None, dt=0.033):
        # Preprocess inputs
        color = self.preprocess_color(color)
        depth = self.preprocess_depth(depth)
        
        # Update orientation
        if gyro is not None:
            orientation = self.orientation_tracker.update(gyro, dt)
            rot_matrix = R.from_quat([orientation.x, orientation.y, 
                                    orientation.z, orientation.w]).as_matrix()
            self.current_pose[:3,:3] = rot_matrix
        
        # Estimate translation
        translation = self.estimate_translation(color, depth)
        if translation is not None:
            self.current_pose[:3,3] += translation
        
        # Update map with careful integration
        self.update_map(color, depth)
        
        return self.current_pose
    
    def update_map(self, color, depth):
        """Enhanced map integration with proper normal estimation"""
        points, colors = self.depth_to_points(depth, color)
        if len(points) < 100:  # Skip frames with too few points
            return
            
        # Create frame point cloud
        frame_pcd = o3d.geometry.PointCloud()
        frame_pcd.points = o3d.utility.Vector3dVector(points)
        frame_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals for the new frame
        frame_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
        
        # Only attempt ICP if we have a prior map with normals
        if len(self.global_pcd.points) > 1000:
            # Ensure target cloud has normals
            if not self.global_pcd.has_normals():
                self.global_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30))
            
            # Perform Point-to-Plane ICP
            icp_result = o3d.pipelines.registration.registration_icp(
                source=frame_pcd,
                target=self.global_pcd,
                max_correspondence_distance=self.icp_threshold,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=50)
            )
            
            if icp_result.fitness > 0.5:  # Only accept good alignments
                self.current_pose = icp_result.transformation @ self.current_pose
                # Recompute points with corrected pose
                points, colors = self.depth_to_points(depth, color)
                frame_pcd.points = o3d.utility.Vector3dVector(points)
                frame_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Merge with global map
        self.global_pcd += frame_pcd
        
        # Conservative downsampling
        if len(self.global_pcd.points) > 50000:
            self.global_pcd = self.global_pcd.voxel_down_sample(self.voxel_size)
            # Re-estimate normals after downsampling
            self.global_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30))
        
        # Update visualization
        self.vis.update_geometry(self.global_pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def depth_to_points(self, depth, color):
        """Precise point cloud generation with proper scaling"""
        h, w = depth.shape
        fx, fy, cx, cy = self.intrinsics
        
        # Create coordinate grids
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        zs = depth
        
        # Filter valid points
        valid_mask = (zs > self.min_depth) & (zs < self.max_depth)
        xs = xs[valid_mask]
        ys = ys[valid_mask]
        zs = zs[valid_mask]
        
        # Convert to 3D coordinates with proper scaling
        points = np.column_stack([
            (xs - cx) * zs / fx,
            (ys - cy) * zs / fy,
            zs
        ])
        
        # Transform to global frame
        points = (self.current_pose[:3,:3] @ points.T + self.current_pose[:3,3].reshape(3,1)).T
        colors = color[ys, xs] / 255.0
        
        return points, colors

def main(input_dir, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading intrinsics from {input_dir}/intrinsics.txt...")
    intrinsics = np.loadtxt(f"{input_dir}/intrinsics.txt")
    slam = EnhancedSLAM(intrinsics)
    
    print("Scanning input directory for color and depth images...")
    color_files = sorted(Path(input_dir).glob("color_*.png"))
    depth_files = sorted(Path(input_dir).glob("depth_*.png"))
    
    print(f"Found {len(color_files)} color images and {len(depth_files)} depth images.")
    if len(color_files) == 0 or len(depth_files) == 0:
        print("Error: No input images found. Exiting.")
        return
    
    for i, (color_file, depth_file) in enumerate(zip(color_files, depth_files)):
        print(f"Processing frame {i + 1}/{len(color_files)}...")
        color = cv2.imread(str(color_file))
        depth = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
        
        if color is None or depth is None:
            print(f"Warning: Skipping frame {i + 1} due to missing data.")
            continue
            
        slam.process_frame(color, depth, gyro=None, dt=0.033)
        
        if i % 50 == 0:
            print(f"Saving intermediate map at frame {i + 1}...")
            slam.save_map(f"{output_dir}/map_{i:06d}.ply")
    
    print("Saving final map...")
    slam.save_map(f"{output_dir}/final_map.ply")
    slam.close()
    print("3D mapping complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing color and depth images")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        exit(1)
        
    print("Starting 3D mapping...")
    main(args.input_dir, args.output_dir)