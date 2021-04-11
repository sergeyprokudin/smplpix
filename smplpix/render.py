import os
import numpy as np
import pyrender
import trimesh

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def render_mesh(mesh_trimesh, camera_center, camera_transl, focal_length, img_width, img_height):

    mesh_trimesh = trimesh.load()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    vertex_colors = np.loadtxt(os.path.join(script_dir, 'smplx_verts_colors.txt'))
    mesh_new = trimesh.Trimesh(vertices=mesh_trimesh.vertices, faces=mesh_trimesh.faces, vertex_colors=vertex_colors)
    mesh_new.vertex_colors = vertex_colors
    print("mesh visual kind: %s" % mesh_new.visual.kind)

    mesh = pyrender.Mesh.from_trimesh(mesh_new, smooth=False, wireframe=False)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_transl

    camera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=camera_center[0], cy=camera_center[1])
    scene.add(camera, pose=camera_pose)

    light = pyrender.light.DirectionalLight()

    scene.add(light)
    r = pyrender.OffscreenRenderer(viewport_width=img_width,
                                   viewport_height=img_height,
                                   point_size=1.0)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    output_img = color[:, :, 0:3]
    output_img = (output_img * 255).astype(np.uint8)

    return output_img