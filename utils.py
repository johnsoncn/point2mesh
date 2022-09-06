import torch
import numpy as np
import os
import uuid
from options import MANIFOLD_DIR
import glob

def manifold_upsample(mesh, save_path, Mesh, num_faces=2000, res=3000, simplify=True):
    # export before upsample
    fname = os.path.join(save_path, 'recon_{}.obj'.format(len(mesh.faces)))
    mesh.export(fname)

    temp_file = os.path.join(save_path, random_file_name('obj'))
    opts = ' ' + str(res) if res is not None else ''

    manifold_script_path = os.path.join(MANIFOLD_DIR, 'manifold')
    if not os.path.exists(manifold_script_path):
        raise FileNotFoundError(f'{manifold_script_path} not found')
    cmd = "{} {} {}".format(manifold_script_path, fname, temp_file + opts)
    os.system(cmd)

    if simplify:
        cmd = "{} -i {} -o {} -f {}".format(os.path.join(MANIFOLD_DIR, 'simplify'), temp_file,
                                                             temp_file, num_faces)
        os.system(cmd)

    m_out = Mesh(temp_file, hold_history=True, device=mesh.device)
    fname = os.path.join(save_path, 'recon_{}_after.obj'.format(len(m_out.faces)))
    m_out.export(fname)
    [os.remove(_) for _ in list(glob.glob(os.path.splitext(temp_file)[0] + '*'))]
    return m_out


def read_pts(pts_file):
    '''
    :param pts_file: file path of a plain text list of points
    such that a particular line has 6 float values: x, y, z, nx, ny, nz
    which is typical for (plaintext) .ply or .xyz
    :return: xyz, normals
    '''
    xyz, normals = [], []
    with open(pts_file, 'r') as f:
        # line = f.readline()
        spt = f.read().split('\n')
        # while line:
        for line in spt:
            parts = line.strip().split(' ')
            try:
                x = np.array(parts, dtype=np.float32)
                xyz.append(x[:3])
                normals.append(x[3:])
            except:
                pass
    return np.array(xyz, dtype=np.float32), np.array(normals, dtype=np.float32)


def load_obj(file):
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


def export(file, vs, faces, vn=None, color=None):
    with open(file, 'w+') as f:
        for vi, v in enumerate(vs):
            if color is None:
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            else:
                f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
            if vn is not None:
                f.write("vn %f %f %f\n" % (vn[vi, 0], vn[vi, 1], vn[vi, 2]))
        for face in faces:
            f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))


def random_file_name(ext, prefix='temp'):
    return f'{prefix}{uuid.uuid4()}.{ext}'

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= (lens + 1e-8)
    arr[:,1] /= (lens + 1e-8)
    arr[:,2] /= (lens + 1e-8)
    return arr

def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    normals = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    normals[faces[:, 0]] += n
    normals[faces[:, 1]] += n
    normals[faces[:, 2]] += n
    normalize_v3(normals)

    return normals

def read_ply_xyzrgbnormal(filename = "/home/dingchaofan/point2mesh/data/modified.ply"):
    from plyfile import PlyData, PlyElement
    """ read XYZ RGB normals point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        print(plydata)
        print(plydata['vertex'].data)
        print(f"plydata['face''].data = ", plydata["face"].data)

        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 9], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,6] = plydata['vertex'].data['red']
        vertices[:,7] = plydata['vertex'].data['green']
        vertices[:,8] = plydata['vertex'].data['blue']
        # vertices[:,9] = plydata['vertex'].data['alpha']

        print(vertices[:,8])

        # compute normals
        xyz = np.array([[x, y, z] for x, y, z, _, _, _, _, _, _, _ in plydata["vertex"].data])
        print(xyz)
        face = np.array([f[0] for f in plydata["face"].data])
        print(face)
        nxnynz = compute_normal(xyz, face)
        vertices[3:6] = nxnynz

        print(plydata['vertex'].data)
    return vertices

def read_ply_open3d(pts_file):
    import open3d as o3d
    import numpy as np

    ply = o3d.io.read_point_cloud(pts_file, 'auto', True, True)
    print(ply)
    print(np.asarray(ply.points))
    print(np.asarray(ply.points).shape)

    # ply.paint_uniform_corlor([0,0,1])
    # # 法线估计 : ALT + - 号加长减小法线
    radius = 0.01  # 搜索半径
    max_nn = 30  # 邻域内用于估算法线的最大点数
    ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

    print(np.asarray(ply.normals))
    print(np.asarray(ply.normals).shape)

    return np.asarray(ply.points), np.asarray(ply.normals)

if __name__ == "__main__":

    read_ply_xyzrgbnormal('/home/dingchaofan/point2mesh/data/modified.ply')
