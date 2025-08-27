import numpy as np 
import cv2 
from plyfile import PlyData, PlyElement 
 
def read_ply_file(file_path):
    """读取PLY文件"""
    ply = PlyData.read(file_path) 
    vertices = np.array(ply['vertex'].data.tolist()) 
    # 读取面信息 
    print(ply.elements)
    # if 'face' in ply.elements: 
    faces = []
    for face in ply['face']:
        # 假设每个面是一个长度为3的列表 
        face_indices = face[0]
        if len(face_indices) == 3:
            faces.append(face_indices) 
    faces = np.array(faces,  dtype=np.int32) 
    # else:
    #     faces = None 
    

    print("顶点数量:", len(vertices))
    print("面数量:", len(faces) if faces is not None else "无面信息")
    return vertices, faces 
 
def process_texture_images(image_paths):
    """处理纹理图片"""
    textures = []
    for path in image_paths:
        try:
            img = cv2.imread(path) 
            img = cv2.resize(img, (1024, 1024))
        except:
            print("无法读取图片", path)
            continue
        # 进行去噪处理，例如高斯滤波 
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        textures.append(img) 
    return textures 
 
def stitch_textures(textures, output_size=(1024, 1024)):
    """拼接四张纹理图片"""
    # textures = []
    # for path in texture_paths:
    #     img = cv2.imread(path) 
    #     img = cv2.resize(img,  output_size)
    #     textures.append(img) 
    
    # 拼接成一个大的纹理贴图 
    stitched_texture = np.zeros((output_size[1]  * 2, output_size[0] * 2, 3), dtype=np.uint8) 
    stitched_texture[:output_size[1], :output_size[0]] = textures[0]  # 前方 
    stitched_texture[:output_size[1], output_size[0]:] = textures[1]  # 右方 
    stitched_texture[output_size[1]:, :output_size[0]] = textures[2]  # 后方 
    stitched_texture[output_size[1]:, output_size[0]:] = textures[3]  # 左方 
    
    # test
    # stitched_texture[:output_size[1], :output_size[0]] = textures[2]  # 前方 
    # stitched_texture[:output_size[1], output_size[0]:] = textures[3]  # 右方 
    # stitched_texture[output_size[1]:, :output_size[0]] = textures[0]  # 后方 
    # stitched_texture[output_size[1]:, output_size[0]:] = textures[1]  # 左方 

    # 顺时针旋转纹理贴图
    # stitched_texture = cv2.rotate(stitched_texture, cv2.ROTATE_90_CLOCKWISE)

    return stitched_texture 
 
def compute_uv_coordinates(vertices, model_bounds, texture_size=(1024, 1024)):
    """计算UV坐标"""
    u_coords = ((vertices[:, 0] - model_bounds[0]) / (model_bounds[1] - model_bounds[0])) * texture_size[0]
    v_coords = ((vertices[:, 1] - model_bounds[2]) / (model_bounds[3] - model_bounds[2])) * texture_size[1]
    uv = np.stack([u_coords,  v_coords], axis=1)
    return uv.astype(int) 
 
def update_vertex_colors(vertices, stitched_texture):
    """根据UV坐标更新顶点颜色"""
    colors = []
    texture_height, texture_width = stitched_texture.shape[:2] 
    
    for vertex in vertices:
        # u = int(vertex['u'] * (texture_width - 1))
        # v = int(vertex['v'] * (texture_height - 1))
        u = int(vertex[6] * (texture_width - 1))
        v = int(vertex[7] * (texture_height - 1))
        if 0 <= u < texture_width and 0 <= v < texture_height:
            color = stitched_texture[v, u]
        else:
            color = [0, 0, 0]  # 超出范围设置为黑色 
        
        colors.append(color) 
    
    return np.array(colors,  dtype=np.uint8) 
 
def save_ply_file(vertices, faces, output_path):
    """保存更新后的PLY文件"""
    vertex_dtype = [
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('nx', 'f4'),
        ('ny', 'f4'),
        ('nz', 'f4'),
        ('u', 'f4'),
        ('v', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1')
    ]
    
    vertex_array = np.array(vertices, dtype=np.float32)
    vertex_array = np.empty(len(vertices), dtype=vertex_dtype)
    # print(vertices.shape)
    # print(vertices[:10])
    vertex_array['x'] = vertices[:, 0]
    vertex_array['y'] = vertices[:, 1]
    vertex_array['z'] = vertices[:, 2]
    vertex_array['nx'] = vertices[:, 3]
    vertex_array['ny'] = vertices[:, 4]
    vertex_array['nz'] = vertices[:, 5]
    vertex_array['u'] = vertices[:, 6]
    vertex_array['v'] = vertices[:, 7]
    vertex_array['red'] = vertices[:, 8]
    vertex_array['green'] = vertices[:, 9]
    vertex_array['blue'] = vertices[:, 10]

    # vertex_array = np.array(vertices, dtype=vertex_dtype)
    # vertex_array = np.unique(vertex_array, axis=0)  # 去重

    if faces is not None:
        # 正确处理面数据格式
        face_dtype = [('vertex_indices', 'i4', 3)]
        face_data = np.empty(len(faces), dtype=face_dtype)
        for i, face in enumerate(faces):
            face_data[i] = (face,)  # 只需要顶点索引
        face_element = PlyElement.describe(face_data, 'face')
    else:
        face_element = None

    ply_elements = [PlyElement.describe(vertex_array, 'vertex')]
    if face_element is not None:
        ply_elements.append(face_element)
    
    ply_data = PlyData(ply_elements, text=True)
    
    ply_data.write(output_path)

 
# 主程序 
if __name__ == "__main__":
    # 输入路径 
    ply_input_path = 'exps/models/audi.a2.ply' 
    # 把一张1x4的纹理图片切分成四张图片
    texture_path = 'exps/textures/audi.a2.png'
    import cv2
    img = cv2.imread(texture_path)
    h, w = img.shape[:2]
    texture_paths = []
    for i, name in enumerate(['front', 'right', 'back', 'left']):
        texture_paths.append(f'exps/textures/audi.a2_{name}.png')
        cv2.imwrite(texture_paths[-1], img[:, i*w//4:(i+1)*w//4])

    # print(texture_paths)
    # texture_paths = ['exps/textures/front.jpg',  'exps/textures/right.jpg',  'exps/textures/back.jpg',  'exps/textures/left.jpg']
    
    # 输出路径 
    ply_output_path = 'exps/models/audi.a2_textured.ply' 
    
    # 读取PLY文件 
    vertices, faces = read_ply_file(ply_input_path)
    
    # 提取模型边界 
    # min_x, max_x = np.min(vertices[:,  0]), np.max(vertices[:,  0])
    # min_y, max_y = np.min(vertices[:,  1]), np.max(vertices[:,  1])
    # model_bounds = (min_x, max_x, min_y, max_y)
    
    # 处理纹理图片 
    textures = process_texture_images(texture_paths)
    
    # 拼接纹理 
    stitched_texture = stitch_textures(textures)
    
    # 计算UV坐标 
    # uv_coords = compute_uv_coordinates(vertices[:, :3], model_bounds, stitched_texture.shape[:2]) 
    
    # 更新顶点颜色 
    new_colors = update_vertex_colors(vertices, stitched_texture)
    
    # 合并位置和颜色 
    updated_vertices = np.concatenate([vertices[:,  :8], new_colors], axis=1)
    
    # 保存新的PLY文件 
    save_ply_file(updated_vertices, faces, ply_output_path)
    
    print("纹理映射完成，结果已保存为", ply_output_path)