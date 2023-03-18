### Requirements
### pip install torch delaunay-watershed-3d largesteps

import torch
import numpy as np
from tqdm import tqdm
from largesteps.solvers import CholeskySolver
from largesteps.optimize import AdamUniform

def laplacian_uniform(verts, faces):
    """
    Compute the uniform laplacian
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()

def compute_matrix(verts, faces, lambda_):
    """
    Build the parameterization matrix.
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    lambda_ : float
        Hyperparameter lambda of our method, used to compute the
        parameterization matrix as (I + lambda_ * L)
    """
    L = laplacian_uniform(verts, faces)

    idx = torch.arange(verts.shape[0], dtype=torch.long, device=verts.device)
    eye = torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0), torch.ones(verts.shape[0], dtype=torch.float, device=verts.device), (verts.shape[0], verts.shape[0]))
    M = torch.add(eye, lambda_*L) # M = I + lambda_ * L
    return M.coalesce()

# We provide routines to compute surface energy and volume penalization

def compute_surface_energy(Verts,Faces,Coeffs_faces):
    Coords = Verts[Faces]
    cross_prods = torch.cross(Coords[:,1]-Coords[:,0],Coords[:,2]-Coords[:,0],dim=1)
    Areas = 0.5*torch.norm(cross_prods,dim=1)
    Energy_triangles = Areas*Coeffs_faces
    return(torch.sum(Energy_triangles))

def compute_volume_cells_tensor(Verts,Faces, Idx_volumes):
    Coords = Verts[Faces]
    cross_prods = torch.cross(Coords[:,1],Coords[:,2],dim=1)
    determinants = torch.sum(cross_prods*Coords[:,0],dim=1)
    Volumes = torch.stack([torch.sum(determinants*Idx_volumes[key])/6 for key in range(1,num_cells+1)])
    return(Volumes)

"""
Vertices are imported as a [nv,3] float array | Faces are imported as a [nt, 3] int array
They are converted as torch tensor, and we indicate that we will need the gradient with respect to the vertices by putting Verts.requires_grad to True
"""
Verts_init, Faces_mm = np.load("Meshes/Doublet_coarse.npy",allow_pickle=True)
Faces = torch.tensor(Faces_mm[:,:3])
Verts = torch.tensor(Verts_init,dtype = torch.float)
Verts.requires_grad = True

num_cells = 2
dict_surface_tensions = {(0,1): 0.7,
                         (0,2): 1.4,
                         (1,2): 1.0}

#Precompute energy and volume coeffs (For faster computations)
Coeffs_faces = torch.zeros(len(Faces_mm))
Idx_volumes = {key:torch.zeros(len(Faces_mm)) for key in range(num_cells+1)}
for i in range(len(Faces_mm)):
    Idx_volumes[Faces_mm[i,3]][i]=1
    Idx_volumes[Faces_mm[i,4]][i]=-1
    Coeffs_faces[i] = dict_surface_tensions[(min(Faces_mm[i,3],Faces_mm[i,4]),max(Faces_mm[i,3],Faces_mm[i,4]))]
    
with torch.no_grad():
    Volumes_target = compute_volume_cells_tensor(Verts, Faces, Idx_volumes) #Compute the initial volumes (for volume conservation)

optimizer =AdamUniform([{'params': Verts,'lr':0.001}]) #Choice of gradient descent scheme

#Precomputation for regularization
lambda_=10.0
M = compute_matrix(Verts, Faces, lambda_)
solver = CholeskySolver(M@M)

for i in (pbar:=tqdm(range(3000))):
    
    #Compute energy and volume gradients
    E_grad = torch.autograd.functional.jacobian(lambda x: compute_surface_energy(x,Faces,Coeffs_faces),Verts)
    V_grad = - torch.autograd.functional.jacobian(lambda x: compute_volume_cells_tensor(x,Faces,Idx_volumes),Verts)
    
    #Volume conservation by projection
    P = torch.sum(torch.sum((V_grad*V_grad.unsqueeze(1)),dim=3),dim=2)
    Q = torch.sum(torch.sum(E_grad*V_grad,dim=2),dim=1)
    
    R = (Volumes_target- compute_volume_cells_tensor(Verts,Faces,Idx_volumes))
    F = torch.linalg.solve(P,Q)
    M = torch.linalg.solve(P,R)
    with torch.no_grad():
        Verts-= torch.sum(((V_grad.transpose(0,2))*M).transpose(0,2),dim=0)
    
    #Update of vertices gradients and regularisation
    Verts.grad=E_grad-torch.sum(((V_grad.transpose(0,2))*F).transpose(0,2),dim=0)
    Verts.grad = solver.solve(Verts.grad)
    
    
    #Gradient descent step
    pbar.set_description("Current surface energy:"+str([compute_surface_energy(Verts,Faces,Coeffs_faces).item()])+"  Volumes:"+str(compute_volume_cells_tensor(Verts,Faces,Idx_volumes).detach().numpy()))
    optimizer.step()  #Update Verts according to the gradient descent schemes

import polyscope as ps
ps.init()
ps.set_ground_plane_mode('none')
ps.register_surface_mesh("Mesh_init",Verts_init,Faces_mm[:,:3])
ps.register_surface_mesh("Mesh_result",Verts.detach().numpy(),Faces.numpy())
ps.show()