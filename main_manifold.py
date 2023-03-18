import numpy as np 
import torch
from tqdm import tqdm
from largesteps.solvers import CholeskySolver
from largesteps.optimize import AdamUniform
import polyscope as ps

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

"""
Vertices are imported as a [nv,3] float array | Faces are imported as a [nt, 3] int array
They are converted as torch tensor, and we indicate that we will need the gradient with respect to the vertices by putting Verts.requires_grad to True
"""
Verts_init,Faces_init = np.load("Meshes/Mesh_cube.npy",allow_pickle=True)
Verts = torch.tensor(Verts_init)
Faces = torch.tensor(Faces_init)
Verts.requires_grad = True 

# We provide two routines to compute areas and volumes of arrays
def compute_volume_manifold(Verts,Faces):
    Coords = Verts[Faces]
    cross_prods = torch.cross(Coords[:,1],Coords[:,2],dim=1)
    determinants = torch.sum(cross_prods*Coords[:,0],dim=1)
    Vol = torch.sum(determinants)/6
    return(Vol)

def compute_area_manifold(Verts,Faces):
    Coords = Verts[Faces]
    cross_prods = torch.cross(Coords[:,1]-Coords[:,0],Coords[:,2]-Coords[:,0],dim=1)
    Areas =  0.5*torch.norm(cross_prods,dim=1)
    return(torch.sum(Areas))

with torch.no_grad():
    Volume_target = compute_volume_manifold(Verts,Faces)#Equal to initial volume
    
optimizer =AdamUniform([{'params': Verts,'lr':0.01}]) #Choice of gradient descent scheme

#Precomputation for regularization
lambda_=10.0
M = compute_matrix(Verts, Faces, lambda_)
solver = CholeskySolver(M@M)

for i in (pbar:=tqdm(range(1000))):
    
    #Compute energy and volume gradients
    E_grad = torch.autograd.functional.jacobian(lambda x: compute_area_manifold(x,Faces),Verts)
    V_grad = - torch.autograd.functional.jacobian(lambda x: compute_volume_manifold(x,Faces),Verts)
    
    #Volume conservation by projection
    P = torch.sum(torch.sum(V_grad*V_grad,dim=1))
    Q = torch.sum(torch.sum(E_grad*V_grad,dim=1))
    R = (Volume_target-compute_volume_manifold(Verts,Faces))
    F = Q/P
    M = R/P
    with torch.no_grad():
        Verts-= V_grad*M
        
    #Update of vertices gradients and regularisation
    Verts.grad=E_grad-F*V_grad
    Verts.grad = solver.solve(Verts.grad)
    
    #Gradient descent step
    pbar.set_description("Area:"+str([compute_area_manifold(Verts,Faces).item()])+"  Volume:"+str(compute_volume_manifold(Verts,Faces).item()))
    optimizer.step()  #Update Verts according to the gradient descent schemes
    
ps.init()
ps.set_ground_plane_mode('none')
ps.register_surface_mesh("Mesh_init",Verts_init,Faces_init)
ps.register_surface_mesh("Mesh_result",Verts.detach().numpy(),Faces.numpy())
ps.show()