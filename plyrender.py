import open3d as o3d
import numpy as np
import os

# 加载 PLY 文件
ply_path = "audi.a2.final.ply"  # 你的 PLY 文件路径
mesh = o3d.io.read_triangle_mesh(ply_path)
mesh.compute_vertex_normals()

# 设定 12 个角度（环绕视角）
angles = np.linspace(0, 360, num=12, endpoint=False)  # 360度均匀分布的12个视角
save_dir = "exps/pics"
# os.makedirs(save_dir, exist_ok=True)

# 创建渲染器
width, height = 256, 256  # 设置渲染尺寸
r = o3d.visualization.rendering.OffscreenRenderer(width, height)

# 创建场景并添加模型
material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLit"  # 确保 PLY 使用正确的着色器
r.scene.add_geometry("model", mesh, material)
# r.scene.scene.set_background_color([1, 1, 1, 1])  # 设置白色背景

# **启用光照**
r.scene.scene.enable_sun_light(True)
r.scene.scene.set_sun_light(
    direction=[0, 0, -1],  # 光照方向
    color=[1, 1, 1],  # 纯白色光
    intensity=100000  # 增强光照强度
)


# **增加环境光**
r.scene.scene.enable_indirect_light(True)
r.scene.scene.set_indirect_light_intensity(30000)  # 适当增强环境光
# # 适配不同 Open3D 版本的背景设置
# try:
#     r.scene.set_background([1, 1, 1, 1])  # Open3D 0.16 可能支持
# except AttributeError:
#     try:
#         r.scene.set_skybox(None)  # Open3D 0.17+
#     except AttributeError:
#         print("Warning: Cannot set background color, using default.")

# # 设置光源
# try:
#     r.scene.scene.enable_indirect_light(True)  # 启用间接光照
#     r.scene.scene.set_indirect_light_intensity(50000)  # 调整光照强度
# except AttributeError:
#     print("Warning: Indirect light not supported, using default lighting.")

# 设置相机参数
cam_distance = 2.0  # 摄像机距离
cam_center = np.array([0, 0, 0])  # 目标点

for i, angle in enumerate(angles):
    theta = np.radians(angle)
    cam_pos = np.array([cam_distance * np.cos(theta), cam_distance * np.sin(theta), 0.5])  # 旋转视角
    up = np.array([0, 0, 1])  # 保持相机向上

    # 设置相机位置
    r.scene.camera.look_at(cam_center, cam_pos, up)

    # 渲染并保存图像
    img = r.render_to_image()
    image_path = os.path.join(save_dir, f"{i:02d}.png")
    o3d.io.write_image(image_path, img)
    print(f"Saved {image_path}")

print("All renders saved successfully!")