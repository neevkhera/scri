from __future__ import print_function, division, absolute_import
import scri
import math
import numpy as np
import quaternion
import spherical_functions as sf
from quaternion import rotate_vectors
from matplotlib import pyplot as plt
from quaternion import as_rotation_matrix

def modes_constructor(constructor_statement, data_functor, **kwargs):
    """WaveformModes object filled with data from the input functor

    Additional keyword arguments are mostly passed to the WaveformModes initializer, though some more reasonable
    defaults are provided.

    Parameters
    ----------
    constructor_statement : str
        This is a string form of the function call used to create the object.  This is passed to the WaveformBase
        initializer as the parameter of the same name.  See the docstring for more information.
    data_functor : function
        Takes a 1-d array of time values and an array of (ell, m) values and returns the complex array of data.
    t : float array, optional
        Time values of the data.  Default is `np.linspace(-10., 100., num=1101))`.
    ell_min, ell_max : int, optional
        Smallest and largest ell value present in the data.  Defaults are 2 and 8.

    """
    t = np.array(kwargs.pop('t', np.linspace(-10., 100., num=1101)), dtype=float)
    frame = np.array(kwargs.pop('frame', []), dtype=np.quaternion)
    frameType = int(kwargs.pop('frameType', scri.Inertial))
    dataType = int(kwargs.pop('dataType', scri.h))
    r_is_scaled_out = bool(kwargs.pop('r_is_scaled_out', True))
    m_is_scaled_out = bool(kwargs.pop('m_is_scaled_out', True))
    ell_min = int(kwargs.pop('ell_min', abs(scri.SpinWeights[dataType])))
    ell_max = int(kwargs.pop('ell_max', abs(scri.SpinWeights[dataType])))
    if kwargs:
        import pprint
        warnings.warn("\nUnused kwargs passed to this function:\n{0}".format(pprint.pformat(kwargs, width=1)))
    data = data_functor(t, sf.LM_range(ell_min, ell_max))
    w = scri.WaveformModes(t=t, frame=frame, data=data,
                           history=['# Called from constant_waveform'],
                           frameType=frameType, dataType=dataType,
                           r_is_scaled_out=r_is_scaled_out, m_is_scaled_out=m_is_scaled_out,
                           constructor_statement=constructor_statement,
                           ell_min=ell_min, ell_max=ell_max)
    return w

def test_waveform(**kwargs):
    """Waveform object having the form h = h(u,x^A) = a(u) \sum_{m = -2}^{2} Y^{2, m}_{-2}`.
       a(u) = cos(u)
       \theta and \phi are taken to be random for each iteration.
       Wigner_D_element is used to compute the SWSH at a given point.

    """
    if kwargs:
        import pprint
        warnings.warn("\nUnused kwargs passed to this function:\n{0}".format(pprint.pformat(kwargs, width=1)))
    def data_functor(t, LM):
        data = np.empty((t.shape[0], LM.shape[0]), dtype=complex)
        for i in range(t.shape[0]):
            for m in range(LM.shape[0]):
                θ, ϕ = np.random.rand(2) * np.array([np.pi, 2*np.pi])
                R = quaternion.from_spherical_coords(θ,ϕ)
                data[i,m]=(np.cos(t[i])*sf.Wigner_D_element(R,2,LM[m,1],2))
        return data
    return modes_constructor('constant_waveform(**{0})'.format(kwargs), data_functor)
        
h = test_waveform()

A = scri.momentum_flux(h.rotate_decomposition_basis(np.quaternion(1,1,0,0).normalized()))
B = scri.momentum_flux(h)
R = np.quaternion(1,1,0,0).normalized()

for i in range(len(B)):
    B[i,:] = rotate_vectors(R,B[i,:],axis = -1)

C = np.subtract(B[:,1],A[:,1])
D = np.add(A[:,1], B[:,1])


x = h.t
y = np.log(np.absolute(np.divide(C,D)))
plt.title("log fractional error of momentum flux in y") 
plt.xlabel("Time") 
plt.ylabel("ln((p_f(R.h)-R.p_f(h))/(p_f(R.h)+R.p_f(h)))") 
plt.plot(x,y)
plt.show()


