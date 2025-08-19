import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import cv2
import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d import structures
from tqdm import tqdm
import imageio
from scipy.spatial.transform import Rotation as R

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    PointLights,
    TexturesVertex,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    RasterizationSettings,
)

def get_cube_point_cloud(side_length, center, color, points_per_face=100):
    """
    Generate points on the surface of a cube.
    
    Args:
        side_length (float): Length of cube sides
        center (np.ndarray): 3D coordinates of cube center
        color (np.ndarray): RGB color values in range [0,1]
        points_per_face (int): Number of points to generate per face
        
    Returns:
        points (np.ndarray): Nx3 array of point coordinates
        colors (np.ndarray): Nx3 array of RGB colors
    """
    # Convert inputs to numpy arrays
    center = np.asarray(center)
    color = np.asarray(color)
    
    # Generate grid of points for one face
    side_points = int(np.sqrt(points_per_face))
    x = np.linspace(-side_length/2, side_length/2, side_points)
    y = np.linspace(-side_length/2, side_length/2, side_points)
    xx, yy = np.meshgrid(x, y)
    
    points = []
    # Front and back faces (fixed z)
    for z in [-side_length/2, side_length/2]:
        points.append(np.column_stack((xx.flatten(), yy.flatten(), np.full_like(xx.flatten(), z))))
    
    # Left and right faces (fixed x)
    for x in [-side_length/2, side_length/2]:
        points.append(np.column_stack((np.full_like(xx.flatten(), x), yy.flatten(), xx.flatten())))
    
    # Top and bottom faces (fixed y)
    for y in [-side_length/2, side_length/2]:
        points.append(np.column_stack((xx.flatten(), np.full_like(xx.flatten(), y), yy.flatten())))
        
    # Combine all points and shift to center
    points = np.vstack(points) + center
    
    # Create color array
    colors = np.tile(color, (points.shape[0], 1))
    
    return points, colors

def voxel_points_and_features_from_voxel_grid(voxel_grid, action_voxels, action_colors):
    """
    Convert voxel grid to point cloud and features
    Args:
        voxel_grid: Voxel grid tensor/array of shape (features, X, Y, Z)
        action_voxels: Action voxel indices tensor/array of shape (num_actions, 3)
        action_colors: Colors for action voxels tensor/array of shape (num_actions, 3)
    Returns:
        points: Point cloud tensor/array of shape (N, 3)
    """

    # Convert inputs to numpy arrays if they are torch tensors
    if torch.is_tensor(voxel_grid):
        voxel_grid = voxel_grid.detach().cpu().numpy()
    if torch.is_tensor(action_voxels):
        action_voxels = action_voxels.detach().cpu().numpy()
    if torch.is_tensor(action_colors):
        action_colors = action_colors.detach().cpu().numpy()
        
    # Convert voxel grid to (X, Y, Z, features) format
    if voxel_grid.shape[0] < voxel_grid.shape[-1]:  # If in (features, X, Y, Z) format
        vis_grid = np.transpose(voxel_grid, (1, 2, 3, 0))
    else:
        vis_grid = voxel_grid
    
    # Remove the action voxels from the voxel grid for point cloud visualization
    for i in range(action_voxels.shape[0]):
        vis_grid[action_voxels[i, 0], 
                 action_voxels[i, 1], 
                 action_voxels[i, 2], 
                 -1] = 0.
    
    # Mask out the points that are not in the voxel grid
    mask = (vis_grid[..., -1] == 1)       #np.linalg.norm(vis_grid[..., :3], axis=-1) > 0
    points = vis_grid[np.where(mask)][..., 6:9]  # Voxel center coordinates
    features = vis_grid[np.where(mask)][..., 3:6]  # RGB colors
    
    # Go from -1 to 1 range to 0 to 1 range (for RGB)
    features = (features + 1) / 2

    for i in range(action_voxels.shape[0]):
        # Get action voxel center coordinates
        action_voxel_center = vis_grid[action_voxels[i, 0], 
                                        action_voxels[i, 1], 
                                        action_voxels[i, 2], 
                                        6:9]  # Voxel center coordinates
        
        # Compute how the voxel center z-coordinate (-2 element) is shifted when moving along the z-indices of the grid
        voxel_size = (voxel_grid[-2, 0, 0, 1] - voxel_grid[-2, 0, 0, 0]) * 2
        action_points, action_features = get_cube_point_cloud(voxel_size, action_voxel_center, action_colors[i])

        points = np.vstack((points, action_points))
        features = np.vstack((features, action_features))

    return points, features

def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def get_points_renderer(image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def capture_image(structures, renderer, device, R = None, T = None, fov = 60, lights = None):

    if (R is None) or (T is None):
        R, T = look_at_view_transform(dist=3., elev=5, azim=0)
    
    # Prepare the camera:
    cameras = FoVPerspectiveCameras(
        R=R, T=T, fov=fov, device=device
    )

    # Place a point light in front of the cow.
    if lights is None:
        lights = PointLights(location=[[0, 0, -3]], device=device)

    if lights is not False:
        image = renderer(structures, cameras=cameras, lights=lights)
    else:
        image = renderer(structures, cameras=cameras)
    image = image.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

    image_uint8 = (image * 255).astype(np.uint8)

    return image_uint8

def render_360_gif(structures: structures, 
                   output_file: str, 
                   image_size: int = 512,
                   light_location: list[float] = [0, 0, -3],
                   fps: int = 30, 
                   dist: float = 3, 
                   elev: float = 5,
                   point_radius: float = 0.01,
                   center_of_rotation = None,   
                   azimuth_range: tuple = (0, 360, 5),
                   up_vector: list[float] = [0, 0, 1],
                   fov: float = 60):
    """
    Render a 360-degree GIF with customizable camera parameters.
    
    Args:
        structures: Pytorch3D structures to render
        output_file: Output GIF file path
        image_size: Rendered image size
        light_location: Light position [x, y, z]
        fps: Frames per second
        dist: Camera distance from center
        elev: Fixed elevation angle
        point_radius: Point cloud point radius
        center_of_rotation: Center point for rotation [x, y, z] (if None, uses scene centroid)
        azimuth_range: (start, end, step) in degrees
        up_vector: The rotation axis direction [x, y, z]
    """
    
    device = structures.device

    # Calculate center of rotation
    if center_of_rotation is None:
        pcd = structures.get_cloud(0)[0]
        center_of_rotation = pcd.mean(dim=0).cpu().numpy().tolist()
    
    # Normalize up_vector to get rotation axis
    up_vector = np.array(up_vector, dtype=np.float32)
    up_vector = up_vector / np.linalg.norm(up_vector)
    
    # Create rotation matrix to align up_vector with Y-axis (standard rotation axis)
    # We want to rotate the scene so that up_vector becomes [0, 1, 0]
    y_axis = np.array([0, 1, 0], dtype=np.float32)
    
    # If up_vector is already aligned with Y-axis, no rotation needed
    if np.allclose(up_vector, y_axis) or np.allclose(up_vector, -y_axis):
        rotation_matrix = np.eye(3)
    else:
        # Calculate rotation axis and angle
        rotation_axis = np.cross(up_vector, y_axis)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Calculate rotation angle
        cos_angle = np.dot(up_vector, y_axis)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Create rotation matrix using Rodrigues' rotation formula
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                     [rotation_axis[2], 0, -rotation_axis[0]],
                     [-rotation_axis[1], rotation_axis[0], 0]])
        rotation_matrix = (np.eye(3) + 
                          np.sin(angle) * K + 
                          (1 - np.cos(angle)) * np.dot(K, K))
    
    # Transform the point cloud
    points = structures.get_cloud(0)[0]
    center_tensor = torch.tensor(center_of_rotation, device=device, dtype=torch.float32)
    
    # Center the points, rotate, then translate back
    points_centered = points - center_tensor
    rotation_tensor = torch.tensor(rotation_matrix, device=device, dtype=torch.float32)
    points_transformed = torch.matmul(points_centered, rotation_tensor.T) + center_tensor
    
    # Create transformed point cloud
    features = structures.features_list()[0]
    transformed_structures = Pointclouds(
        points=[points_transformed],
        features=[features]
    ).to(device)

    # Calculate azimuth angles
    start_azim, end_azim, step_azim = azimuth_range
    azims = np.arange(start_azim, end_azim + step_azim, step_azim)
    frames = len(azims)

    lights = PointLights(location=[light_location], device=device)
    renderer = get_points_renderer(image_size=image_size, radius=point_radius, device=device)
    
    images = []

    print(f"Rendering {frames} frames with azimuth range: {start_azim}° to {end_azim}° (step: {step_azim}°)")
    print(f"Camera distance: {dist}, Elevation: {elev}°")
    print(f"Center of rotation: {center_of_rotation}")
    print(f"Rotation axis (up_vector): {up_vector}")
    
    for azim in azims:
        R, T = look_at_view_transform(
            dist=dist, 
            elev=elev, 
            azim=azim,
            at=(center_of_rotation, ),
            up=((0, 1, 0), )  # Always use Y-axis as up since we transformed the scene
        )
        R = R.to(device)
        T = T.to(device)

        image = capture_image(transformed_structures, renderer, device, R, T, fov=fov, lights=lights)
        images.append(image)

    imageio.mimsave(output_file, images, duration=frames // fps)
    print(f"Saved GIF to: {output_file}")

def make_line(start_point, end_point, density, color):
    """
    Generate a simple line between two 3D points.
    
    Args:
        start_point: Starting 3D coordinates of the line
        end_point: Ending 3D coordinates of the line
        density: Number of points along the line
        color: RGB color values for the line
        
    Raises:
        ValueError: If start and end points are the same
    """
    # Convert points to numpy arrays
    start = np.array(start_point)
    end = np.array(end_point)
    
    # Calculate direction vector
    direction = end - start
    if np.allclose(direction, 0):
        raise ValueError("Start and end points cannot be the same")
    
    # Generate evenly spaced points along the line
    t = np.linspace(0, 1, density)
    points = np.outer(1-t, start) + np.outer(t, end)

    # Colors
    colors = np.tile(np.array(color), (points.shape[0], 1))

    return points, colors

def make_coordinate_frame(rotation, translation, length=1.0, density=50):
    """
    Generate coordinate frame points and features at a specific SE(3) transformation.
    
    Args:
        rotation: 3x3 rotation matrix (torch.Tensor or numpy array)
        translation: 3D translation vector (torch.Tensor or numpy array)
        length: Length of each axis (float)
        density: Number of points per axis (int)
    
    Returns:
        points: Nx3 array of point coordinates
        features: Nx3 array of RGB colors
    """
    # Convert to numpy if needed
    if torch.is_tensor(rotation):
        rotation = rotation.detach().cpu().numpy()
    if torch.is_tensor(translation):
        translation = translation.detach().cpu().numpy()
    
    # Ensure rotation is 3x3 and translation is 3D
    rotation = np.array(rotation).reshape(3, 3)
    translation = np.array(translation).reshape(3)
    
    # Generate base axes at origin using start/end points
    origin = np.zeros(3)
    x_pts, x_col = make_line(origin, np.array([length, 0.0, 0.0]), density=density, color=(1, 0, 0))
    y_pts, y_col = make_line(origin, np.array([0.0, length, 0.0]), density=density, color=(0, 1, 0))
    z_pts, z_col = make_line(origin, np.array([0.0, 0.0, length]), density=density, color=(0, 0, 1))
    
    # Combine all points and colors
    all_points = np.vstack([x_pts, y_pts, z_pts])
    all_colors = np.vstack([x_col, y_col, z_col])
    
    # Apply transformation: R * points + t
    transformed_points = np.dot(all_points, rotation.T) + translation
    
    return transformed_points, all_colors

def make_cylinder_line(start_point, end_point, radius=0.01, axial_density=50, radial_density=20, color=(1, 0, 0)):
    """
    Generate a cylindrical surface of points connecting two 3D points.

    Args:
        start_point: Starting 3D coordinates of the cylinder axis (array-like)
        end_point: Ending 3D coordinates of the cylinder axis (array-like)
        radius: Cylinder radius (float)
        axial_density: Number of samples along the cylinder axis (int)
        radial_density: Number of samples around the circumference (int)
        color: RGB color values for all points (tuple of floats in [0, 1])

    Returns:
        points: (axial_density * radial_density, 3) array of point coordinates
        colors: (axial_density * radial_density, 3) array of RGB colors
    """
    # Convert inputs to numpy arrays
    if torch.is_tensor(start_point):
        start_point = start_point.detach().cpu().numpy()
    if torch.is_tensor(end_point):
        end_point = end_point.detach().cpu().numpy()

    start = np.array(start_point, dtype=np.float32).reshape(3)
    end = np.array(end_point, dtype=np.float32).reshape(3)

    axis_vector = end - start
    axis_length = np.linalg.norm(axis_vector)
    if axis_length <= 0:
        raise ValueError("Start and end points cannot be the same")
    if radius <= 0:
        raise ValueError("Radius must be positive")
    if axial_density < 2 or radial_density < 3:
        raise ValueError("axial_density must be >= 2 and radial_density must be >= 3")

    axis_unit = axis_vector / axis_length

    # Build an orthonormal basis {v1, v2} orthogonal to axis_unit
    arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(arbitrary, axis_unit)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v1 = np.cross(axis_unit, arbitrary)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(axis_unit, v1)

    # Sample along axis and around circumference
    t = np.linspace(0.0, 1.0, axial_density, dtype=np.float32)
    thetas = np.linspace(0.0, 2.0 * np.pi, radial_density, endpoint=False, dtype=np.float32)

    base_points = start + np.outer(t, axis_unit * axis_length)  # (A, 3)
    circle_dirs = np.outer(np.cos(thetas), v1) + np.outer(np.sin(thetas), v2)  # (R, 3)
    circle_dirs *= radius

    # Broadcast to create cylinder surface points
    points = base_points[:, None, :] + circle_dirs[None, :, :]
    points = points.reshape(-1, 3)

    colors = np.tile(np.array(color, dtype=np.float32), (points.shape[0], 1))

    return points, colors

def make_gripper_visualization(rotation, translation, length=1.0, density=50, color=(1, 0, 0)):
    """
    Generate coordinate frame points and features at a specific SE(3) transformation.
    
    Args:
        rotation: 3x3 rotation matrix (torch.Tensor or numpy array)
        translation: 3D translation vector (torch.Tensor or numpy array)
        length: Length of each axis (float)
        density: Number of points per axis (int)
    
    Returns:
        points: Nx3 array of point coordinates
        features: Nx3 array of RGB colors
    """
    # Convert to numpy if needed
    if torch.is_tensor(rotation):
        rotation = rotation.detach().cpu().numpy()
    if torch.is_tensor(translation):
        translation = translation.detach().cpu().numpy()

    base_pts, base_col = make_line(np.array([0., -length, -length]),
                                   np.array([0., length, -length]),
                                   density=density, color=color)
    
    # Left and right are as if you are looking at the gripper from the manipulator base, at home position

    left_finger_pts, left_finger_col = make_line(np.array([0., -length, length]),
                                                 np.array([0., -length, -length]),
                                                 density=density, color=color)
    
    right_finger_pts, right_finger_col = make_line(np.array([0., length, length]),
                                                   np.array([0., length, -length]),
                                                   density=density, color=color)
    
    top_pts, top_col = make_line(np.array([0., 0., -length]),
                                 np.array([0., 0., -length - (2 * length)]),
                                 density=density, color=color)
    
    # Combine all points and colors
    all_points = np.vstack([base_pts, left_finger_pts, right_finger_pts, top_pts])
    all_colors = np.vstack([base_col, left_finger_col, right_finger_col, top_col])
    
    # Apply transformation: R * points + t
    transformed_points = np.dot(all_points, rotation.T) + translation
    
    return transformed_points, all_colors

def gaussian_3d_pcd(mean, std, num_points, color = None):
    """
    Generate a point cloud sampled uniformly from inside a 3D ellipsoid
    (centered at mean, axes lengths given by std),
    and color the points using the plasma colormap based on Mahalanobis distance.
    """
    # Uniformly sample inside a unit sphere
    points = []
    while len(points) < num_points:
        p = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(p) <= 1:
            points.append(p)
    points = np.array(points)
    # Scale by std and shift by mean
    points = points * std + mean

    if color is None:
        # Color by Mahalanobis distance (for visualization)
        mahal = np.sqrt(np.sum(((points - mean) / std) ** 2, axis=1))
        mahal = mahal / mahal.max()  # Normalize to [0, 1]
        colors = plt.cm.plasma(1 - mahal)[:, :3]
    else:
        colors = color

    return points, (colors * 255).astype(np.uint8)

def transform_pcd(pcd: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transforms the given point cloud by the given transformation matrix.

    Args:
    -----
        pcd (np.ndarray): Nx3 point cloud
        transform (np.ndarray): 4x4 transformation matrix

    Returns:
    --------
            pcd_new (np.ndarray): Nx3 transformed point cloud
    """

    if pcd.shape[1] != 4:
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd_new = np.matmul(transform, pcd.T)[:-1, :].T
    return pcd_new

def reshape_to_points(data):
    """
    Reshape data to have shape (-1, 3) by finding the dimension with length 3
    and moving it to the end, then flattening all other dimensions.
    Also creates a writable copy of the data.
    
    Args:
        data: numpy array with one dimension of length 3
        
    Returns:
        writable reshaped data with shape (-1, 3)
    """
    # Ensure data has 2-3 dimensions
    if len(data.shape) > 3 or len(data.shape) < 2:
        raise ValueError("Data must have 2 or 3 dimensions")

    # Find dimension with length 3 and move it to end
    three_dim = None
    for i, dim in enumerate(data.shape):
        if dim == 3:
            three_dim = i
            break
    
    if three_dim is None:
        raise ValueError("Data must have one dimension of length 3")
        
    # Move dimension with length 3 to end and reshape
    if three_dim != len(data.shape)-1:
        dims = list(range(len(data.shape)))
        dims.remove(three_dim)
        dims.append(three_dim)
        data = np.transpose(data, dims)
    
    data = data.reshape(-1, 3)
    
    # Create writable copy
    data_new = np.zeros(data.shape)
    data_new[:] = data[:]
    
    return data_new

def plot_pcd(pcd, colors=None, frame=False):

    if type(pcd) == torch.Tensor:
        pcd = pcd.cpu().detach().numpy()
    if colors is not None and type(colors) == torch.Tensor:
        colors = colors.cpu().detach().numpy()

    # Reshape point cloud to (-1, 3) and create writable copy
    pcd_new = reshape_to_points(pcd)

    pts_vis = o3d.geometry.PointCloud()
    pts_vis.points = o3d.utility.Vector3dVector(pcd_new)

    if colors is not None:
        # Apply the same reshaping to colors as we did to pcd
        colors_new = reshape_to_points(colors)
        
        # Ensure colors are in the right range [0, 1]
        if colors_new.max() > 1.0:
            colors_new = colors_new / 255.0
        
        pts_vis.colors = o3d.utility.Vector3dVector(colors_new)

    geometries = [pts_vis]

    if frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        geometries.append(frame)

    o3d.visualization.draw_geometries(geometries)

def plot_voxel_grid_with_action(voxel_grid: torch.Tensor, 
                    action_voxels: torch.Tensor,
                    action_colors: torch.Tensor):
    """
    Plot the voxel grid with the action translation in the voxel grid
    Args:
        voxel_grid: (10, D, H, W)
        action_voxels: (N, 3)
        action_colors: (N, 3)
    """
    
    vis_grid = voxel_grid.permute(1, 2, 3, 0)

    # Remove the action voxels from the voxel grid:
    vis_grid[action_voxels[:, 0], 
             action_voxels[:, 1], 
             action_voxels[:, 2], 
             :3] = 0.

    # Mask out the points that are not in the voxel grid:
    mask = torch.norm(vis_grid[..., :3], dim = -1) > 0      # Could just use occupancy instead...
    vis_pts = vis_grid[torch.where(mask)][..., 6:9]
    vis_rgb = vis_grid[torch.where(mask)][..., 3:6]

    # Add the action voxels to the voxel grid point cloud
    action_voxel_center = vis_grid[action_voxels[:, 0], 
                                   action_voxels[:, 1], 
                                   action_voxels[:, 2], 
                                   6:9]
    
    vis_pts = torch.cat([vis_pts, action_voxel_center], dim=0)
    vis_rgb = torch.cat([vis_rgb, action_colors], dim=0)

    plot_pcd(vis_pts, vis_rgb)

def create_cube_without_points(cube_size=64):
    """
    Create a single transparent cube with only its edges visible (no points).
    
    Args:
        cube_size: Size of the cube
    
    Returns:
        Open3D LineSet geometry for the cube edges
    """
    # Create the cube edges (12 edges of a cube)
    cube_points = [
        [0, 0, 0],           # 0: bottom front left
        [cube_size, 0, 0],   # 1: bottom front right
        [cube_size, cube_size, 0], # 2: bottom back right
        [0, cube_size, 0],   # 3: bottom back left
        [0, 0, cube_size],   # 4: top front left
        [cube_size, 0, cube_size], # 5: top front right
        [cube_size, cube_size, cube_size], # 6: top back right
        [0, cube_size, cube_size]  # 7: top back left
    ]
    
    # Define the 12 edges of the cube
    cube_edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    # Create line set for cube edges
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(cube_points)
    line_set.lines = o3d.utility.Vector2iVector(cube_edges)
    line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(cube_edges))  # Grey edges
    
    return line_set

def plot_voxel_grid_with_action_cubes(voxel_grid, action_voxels, action_colors, cube_size=1., action_cube_size=0.03):
    """
    Plot voxel grid with action voxels as larger cubes and enclose everything in a transparent cube.
    
    Args:
        voxel_grid: Voxel grid tensor/array of shape (features, X, Y, Z)
        action_voxels: Action voxel indices tensor/array of shape (num_actions, 3)
        action_colors: Colors for action voxels tensor/array of shape (num_actions, 3)
        cube_size: Size of the main voxel grid cube
        action_cube_size: Size multiplier for action cubes (relative to voxel size)
    """
    geometries = []
    
    # Create the main transparent cube that encloses everything
    main_cube = create_cube_without_points(cube_size)
    geometries.append(main_cube)
    
    # Convert inputs to numpy arrays if they are torch tensors
    if torch.is_tensor(voxel_grid):
        voxel_grid = voxel_grid.detach().cpu().numpy()
    if torch.is_tensor(action_voxels):
        action_voxels = action_voxels.detach().cpu().numpy()
    if torch.is_tensor(action_colors):
        action_colors = action_colors.detach().cpu().numpy()
        
    # Convert voxel grid to (X, Y, Z, features) format
    if voxel_grid.shape[0] < voxel_grid.shape[-1]:  # If in (features, X, Y, Z) format
        vis_grid = np.transpose(voxel_grid, (1, 2, 3, 0))
    else:
        vis_grid = voxel_grid
    
    # Remove the action voxels from the voxel grid for point cloud visualization
    for i in range(action_voxels.shape[0]):
        vis_grid[action_voxels[i, 0], 
                 action_voxels[i, 1], 
                 action_voxels[i, 2], 
                 :3] = 0.
    
    # Mask out the points that are not in the voxel grid
    mask = np.linalg.norm(vis_grid[..., :3], axis=-1) > 0
    vis_pts = vis_grid[np.where(mask)][..., 6:9]  # Voxel center coordinates
    vis_rgb = vis_grid[np.where(mask)][..., 3:6]  # RGB colors
    
    # Create point cloud from voxel grid
    if len(vis_pts) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis_pts)
        vis_rgb = (vis_rgb + 1) / 2
        pcd.colors = o3d.utility.Vector3dVector(vis_rgb)        # 0 TO 1 range
        geometries.append(pcd)
    
    # Create larger cubes for action voxels
    for i in range(action_voxels.shape[0]):
        # Get action voxel center coordinates
        action_voxel_center = vis_grid[action_voxels[i, 0], 
                                       action_voxels[i, 1], 
                                       action_voxels[i, 2], 
                                       6:9]  # Voxel center coordinates
        
        # Create cube for this action voxel
        cube = o3d.geometry.TriangleMesh.create_box(
            width=action_cube_size, 
            height=action_cube_size, 
            depth=action_cube_size
        )
        
        # Position the cube at the action voxel center
        cube.translate(action_voxel_center - action_cube_size/2)
        
        # Color the cube with action color
        action_color = action_colors[i]
        if action_color.max() > 1.0:  # Normalize color if needed
            action_color = action_color / 255.
        cube.paint_uniform_color(action_color)
        
        geometries.append(cube)
    
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=cube_size * 0.1, origin=[0, 0, 0]
    )
    geometries.append(coordinate_frame)
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Voxel Grid with Action Cubes",
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_back_face=True
    )

def visualize_image(pcd_vis, point_size=0.4, zoom_out=0.1):
    """Render point cloud image with standard camera setup"""
    # Set up Open3D visualization and render
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=512, height=512)
    vis.add_geometry(pcd_vis)

    # Set point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size

    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = np.eye(4)
    extrinsic[2, 3] = zoom_out  # Zoom out to capture everything
    camera_params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Render and capture the view
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)

    # Convert image to numpy array
    image_np = (np.asarray(image) * 255).astype(np.uint8)

    # Clean up
    vis.destroy_window()
    return image_np

def plot_prediction(data_path, scene_bounds=None, gripper_length=0.05, gripper_density=300, 
                   gt_color=(0, 255, 0), pred_color=(255, 0, 0)):
    """
    Plot prediction results with scene point cloud and gripper visualizations.
    
    Args:
        data_path: Path to the .npz file containing visualization data
        scene_bounds: Bounds for filtering scene points [x_min, y_min, z_min, x_max, y_max, z_max]
        gripper_length: Length of gripper visualization components
        gripper_density: Number of points per gripper line
        gt_color: RGB color for ground truth gripper (tuple)
        pred_color: RGB color for predicted gripper (tuple)
    """
    data = np.load(data_path)
    rgb = data["rgb"]
    pcd = data["pcd"]
    pred = data["pred"]
    gt = data["gt"]

    # ------------ Create the Scene Point Cloud ------------ #

    # Combine the different views along the first dimension
    rgb = np.transpose(rgb, (0, 2, 3, 1))  # (num_views, 3, H, W) -> (num_views, H, W, 3)
    pcd = np.transpose(pcd, (0, 2, 3, 1))  # (num_views, 3, H, W, 3) -> (num_views, H, W, 3)
    rgb = rgb.reshape(-1, 3)  # (num_views, H, W, 3) -> (num_views*H*W, 3)
    pcd = pcd.reshape(-1, 3)  # (num_views, H, W, 3) -> (num_views*H*W, 3)

    # Apply scene bounds filtering if provided
    if scene_bounds is not None:
        bound_mask = (pcd[:, 0] > scene_bounds[0]) & (pcd[:, 0] < scene_bounds[3]) & \
                     (pcd[:, 1] > scene_bounds[1]) & (pcd[:, 1] < scene_bounds[4]) & \
                     (pcd[:, 2] > scene_bounds[2]) & (pcd[:, 2] < scene_bounds[5])
        rgb = rgb[bound_mask]
        pcd = pcd[bound_mask]

    # -------------------------------------------------------- #

    # ------------ Create the Gripper Point Clouds ------------ #

    # GROUND TRUTH:
    gt_pos = gt[:3]
    gt_quat = gt[3:7]
    gt_rot = R.from_quat(gt_quat).as_matrix()

    # PREDICTION:
    pred_pos = pred[:3]
    pred_quat = pred[3:7]
    pred_rot = R.from_quat(pred_quat).as_matrix()

    gt_gripper_points, gt_gripper_colors = make_gripper_visualization(
        rotation=gt_rot,
        translation=gt_pos,
        length=gripper_length,
        density=gripper_density,
        color=gt_color
    )

    pred_gripper_points, pred_gripper_colors = make_gripper_visualization(
        rotation=pred_rot,
        translation=pred_pos,
        length=gripper_length,
        density=gripper_density,
        color=pred_color
    )

    # --------------------------------------------------------- #

    # Combine scene points with gripper points
    pcd = np.vstack([pcd, gt_gripper_points, pred_gripper_points])
    rgb = np.vstack([rgb, gt_gripper_colors, pred_gripper_colors])

    # Plot the point clouds
    plot_pcd(pcd, colors=rgb)

if __name__ == "__main__":
    
    import os
    
    # Get list of all step files and sort in descending order
    vis_dir = "train_logs/3DFA_Planning/run_2/vis"
    step_files = sorted([f for f in os.listdir(vis_dir) if f.startswith("step_")], 
                       key=lambda x: int(x.split("_")[1].split(".")[0]),
                       reverse=True)
    
    scene_bounds = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6])
    
    # Plot each step file from latest to earliest
    for step_file in step_files:
        step_num = int(step_file.split("_")[1].split(".")[0])
        print(f"Plotting step {step_num}")
        data_path = os.path.join(vis_dir, step_file)
        plot_prediction(data_path, scene_bounds=scene_bounds)