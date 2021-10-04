import nibabel as nib
import numpy as np

def tract2tck(data, filename, affine_to_rasmm=None):
    if not affine_to_rasmm:
        affine_to_rasmm = np.eye(4,4)

    t_gram = nib.streamlines.Tractogram(data, affine_to_rasmm=affine_to_rasmm)
    tck_out = nib.streamlines.TckFile(t_gram)
    tck_out.save(filename)



