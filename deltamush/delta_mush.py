
import numpy as np
import trimesh
from pdb import set_trace as bp

def write_off(file_path, verts, faces=None):
    """Export point cloud into .off file.

    Positional arguments:
    file_path: output path
    verts: Nx3 array (float)

    Kwargs:
    faces: Mx3 array (int)
    """
    off = open(file_path, 'w')
    assert isinstance(verts, np.ndarray), "Invalid data type for vertices: %s" % type(verts)
    assert len(verts.shape) == 2 and verts.shape[1] == 3, "Invalid array shape for vertices: %s" % str(verts.shape)
    verts_count = verts.shape[0]
    if faces is not None:
        assert isinstance(faces, np.ndarray), "Invalid data type for faces: %s" % type(faces)
        assert len(faces.shape) == 2 and faces.shape[1] == 3, "Invalid array shape for faces: %s" % str(faces.shape)
        faces_count = faces.shape[0]
    # write header
    off.write('OFF\n')
    if faces is not None:
        off.write('%d %d 0\n' % (verts_count, faces_count))
    else:
        off.write('%d 0 0\n' % (verts_count))
    # write vertices
    np.savetxt(off, verts, fmt='%.6f')
    # write faces
    if faces is not None:
        augmented_faces = np.hstack((np.ones((faces.shape[0], 1), dtype=np.int)*3, faces))
        np.savetxt(off, augmented_faces, fmt='%d')
    off.close()


## base function
NORM = np.linalg.norm
def lap_smooth(v,f,adj):
    smoothed = v.copy()
    for i in range(v.shape[0]):
        neibour = adj[i]
        base_point = v[i]
        if 1:
            laplacian = np.vstack((v[neibour]))
            smoothed[i] = np.average(laplacian,0) 
            
        else:
            laplacian = np.zeros_like((base_point))
            edge_cost = 1/ NORM(v[neibour] - v[i],axis=1)
            laplacian += np.sum(v[neibour] * edge_cost.reshape(-1,1),axis=0)
            # laplacian += base_point
            total_weight = np.sum(edge_cost)
            if total_weight > 0:
                smoothed[i] = laplacian/total_weight
        # else:
        
    return smoothed

def smooth2(v,f,adj,iteration):
    for i in range(iteration):
        v = lap_smooth(v,f,adj)
    return v

def get_smoothed_mesh(v,f,iteration=5):
    adj = get_adj(v,f)
    smooth_verts = smooth2(v,f,adj,iteration)
    tri_mesh = trimesh.Trimesh(vertices=smooth_verts,faces=f,process=False)
    return tri_mesh

def get_adj(v,f):
    adj = []
    for i,vt in enumerate(v):
        neibour = set(f[np.where(f==i)[0]].flatten())
        # pdb.set_trace()
        # print(neibour)
        # print(i)
        neibour.remove(i)
        neibour = list(neibour)
        adj.append(neibour)
    return adj

def get_tagent_space_naive(mesh):
    normals = mesh.vertex_normals
    tangents = np.cross(normals,normals+[0,1,0])
    tangents = tangents/np.linalg.norm(tangents,axis=1).reshape(-1,1)
    bitangents = np.cross(normals,tangents)
    bitangents = bitangents/np.linalg.norm(bitangents,axis=1).reshape(-1,1)
    return tangents,normals,bitangents


def rotation_matrix_x(angle):
    rad = angle * np.pi / 180
    return np.array([[1,0,0],[0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])


def rotation_matrix_y(angle):
    rad = angle * np.pi / 180
    return np.array([[np.cos(rad), 0, np.sin(rad)],[0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])

def rotation_matrix_z(angle):
    rad = angle * np.pi / 180
    return np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])


def rotate_plane(vec1, vec2 ):
    """
    giving two vector, return the rotation matrix
    """
    
    #vec1 = vec1 / np.linalg.norm(vec1) #unit vector
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    cos_theta = np.dot(vec1,vec2)/norm
    sin_theta = np.linalg.norm(np.cross(vec1,vec2))/norm
    if sin_theta == 0:
        return np.eye(3)
    k = np.cross(vec1,vec2) /(norm*sin_theta)
    K = np.array([[0,-k[2],k[1]],
                  [k[2],0,-k[0]],
                  [-k[1],k[0],0]])
    R = np.eye(3) + sin_theta*K +(1-cos_theta)*np.dot(K,K)

    return R

def get_index_list(full,part):
    idlist = []
    for pt in part:
        arr = NORM(full-pt,axis=1) < 0.001 
        id = np.where(arr)
        idlist.append(id[0][0])
    return idlist

def get_Rs(tangents,normals,bitangents):
    return np.dstack(( tangents,normals,bitangents))


def get_delta_mushed_target(source_v,target_v,f):
    smooth_time = 25
    smoothed_source_mesh = get_smoothed_mesh(source_v,f,smooth_time)
    st,sn,sb =  get_tagent_space_naive(smoothed_source_mesh)
    Rs = get_Rs(st,sn,sb)
    vd = np.einsum('ijk,ik->ij' ,np.linalg.pinv(Rs),source_v-smoothed_source_mesh.vertices)
    smoothed_target_mesh = get_smoothed_mesh(target_v,f,smooth_time)
    tn = smoothed_target_mesh.vertex_normals
    tt = np.zeros_like(tn)
    tb = np.zeros_like(tn)
    # key part: get rotated tangent space
    for i,vec1 in enumerate(tn):
        Rn = rotate_plane(sn[i],tn[i])
        tt[i],tb[i] = Rn @ st[i], Rn @ sb[i]
    Cs = get_Rs(tt,tn,tb)
    deformed = np.einsum('ijk,ik->ij' ,Cs,vd) + smoothed_target_mesh.vertices
    return deformed


def demo():
    # load source mesh
    source_mesh = trimesh.load_mesh('tube_r.off',process=False)
    v,f = source_mesh.vertices,source_mesh.faces
    # rotate part of tube
    rotation_angle_y = 45
    center = np.average(v,0)
    select = np.where(v[:,0]>center[0]+1)
    R = rotation_matrix_z(rotation_angle_y)
    target = v.copy()
    target[:,0] -= 1  
    target[select] = (R @ target[select].T).T
    target[:,0] += 1
    # get delta mushed target mesh
    deformed = get_delta_mushed_target(v,target,f)
    write_off('deformed.off',deformed,f)

if __name__ == '__main__':
    demo()