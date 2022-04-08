import jax
from jax import random, grad, jit, vmap, lax
from jax.config import config
import jax.numpy as np
from jax.scipy import ndimage
from jax.example_libraries import optimizers,stax
#from jax.opsimport index, index_update
import random as py_random
from livelossplot import PlotLosses
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm
import os
import requests
from io import BytesIO

import cv2
import scipy.ndimage
from scipy.special import binom

from tqdm.notebook import tqdm as tqdm
import numpy as onp

from phantominator import shepp_logan, ct_shepp_logan, ct_modified_shepp_logan_params_3d
from pdb import set_trace
## Random seed
rand_key = random.PRNGKey(10)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#@markdown #Global Defaults

#@markdown resolution
RES = 96 #@param

#@title NP Area Resize Code

# from https://gist.github.com/shoyer/c0f1ddf409667650a076c058f9a17276

def _reflect_breaks(size: int) -> np.ndarray:
  """Calculate cell boundaries with reflecting boundary conditions."""
  result = np.concatenate([[0], 0.5 + np.arange(size - 1), [size - 1]])
  assert len(result) == size + 1
  return result
  
def _interval_overlap(first_breaks: np.ndarray,
                      second_breaks: np.ndarray) -> np.ndarray:
  """Return the overlap distance between all pairs of intervals.

  Args:
    first_breaks: breaks between entries in the first set of intervals, with
      shape (N+1,). Must be a non-decreasing sequence.
    second_breaks: breaks between entries in the second set of intervals, with
      shape (M+1,). Must be a non-decreasing sequence.

  Returns:
    Array with shape (N, M) giving the size of the overlapping region between
    each pair of intervals.
  """
  first_upper = first_breaks[1:]
  second_upper = second_breaks[1:]
  upper = np.minimum(first_upper[:, np.newaxis], second_upper[np.newaxis, :])

  first_lower = first_breaks[:-1]
  second_lower = second_breaks[:-1]
  lower = np.maximum(first_lower[:, np.newaxis], second_lower[np.newaxis, :])

  return np.maximum(upper - lower, 0)

def _resize_weights(
    old_size: int, new_size: int, reflect: bool = False) -> np.ndarray:
  """Create a weight matrix for resizing with the local mean along an axis.

  Args:
    old_size: old size.
    new_size: new size.
    reflect: whether or not there are reflecting boundary conditions.

  Returns:
    NumPy array with shape (new_size, old_size). Rows sum to 1.
  """
  if not reflect:
    old_breaks = np.linspace(0, old_size, num=old_size + 1)
    new_breaks = np.linspace(0, old_size, num=new_size + 1)
  else:
    old_breaks = _reflect_breaks(old_size)
    new_breaks = (old_size - 1) / (new_size - 1) * _reflect_breaks(new_size)

  weights = _interval_overlap(new_breaks, old_breaks)
  weights /= np.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (new_size, old_size)
  return weights

def resize(array: np.ndarray,
           shape: [int, ...],
           reflect_axes: [int] = ()) -> np.ndarray:
  """Resize an array with the local mean / bilinear scaling.

  Works for both upsampling and downsampling in a fashion equivalent to
  block_mean and zoom, but allows for resizing by non-integer multiples. Prefer
  block_mean and zoom when possible, as this implementation is probably slower.

  Args:
    array: array to resize.
    shape: shape of the resized array.
    reflect_axes: iterable of axis numbers with reflecting boundary conditions,
      mirrored over the center of the first and last cell.

  Returns:
    Array resized to shape.

  Raises:
    ValueError: if any values in reflect_axes fall outside the interval
      [-array.ndim, array.ndim).
  """
  reflect_axes_set = set()
  for axis in reflect_axes:
    if not -array.ndim <= axis < array.ndim:
      raise ValueError('invalid axis: {}'.format(axis))
    reflect_axes_set.add(axis % array.ndim)

  output = array
  for axis, (old_size, new_size) in enumerate(zip(array.shape, shape)):
    reflect = axis in reflect_axes_set
    weights = _resize_weights(old_size, new_size, reflect=reflect)
    product = np.tensordot(output, weights, [[axis], [-1]])
    output = np.moveaxis(product, -1, axis)
  return output

#@title Shepp Data Gen

def get_shepp_dataset_3D(rand_key, num_grid_search_samples, test_samples):
    total_samples = num_grid_search_samples + test_samples

    ct_params = np.array(ct_modified_shepp_logan_params_3d())

    shepps = []
    for i in range(total_samples):
        rand_key, subkey = random.split(rand_key)
        i_ct_params = ct_params + random.normal(subkey, shape=ct_params.shape)/20.0
        shepps.append(np.clip(ct_shepp_logan((RES,RES,RES), E=i_ct_params, zlims=(-0.25,0.25)), 0.0, 1.0))

    samples = np.stack(shepps, axis=0)

    out = {
        "data_grid_search":np.array(samples[:num_grid_search_samples,:,:]),
        "data_test":np.array(samples[num_grid_search_samples:,:,:]),
    }
    return out

#@title ATLAS Data Gen

def get_atlas_dataset_3D(rand_key, num_grid_search_samples, test_samples):
  total_samples = num_grid_search_samples + test_samples
  filename="atlas_3d.npz"
  data = np.load(filename)['data']/255.0

  scan_ids = [0, 1, 4, 7, 9, 11, 14, 16, 18, 20, 23, 24, 28] 
  samples = resize(data[scan_ids,...], (len(scan_ids), RES, RES, RES))
  new_samples = random.permutation(rand_key, samples)

  out = {
      "data_grid_search":np.array(new_samples[:num_grid_search_samples,:,:]),
      "data_test":np.array(new_samples[num_grid_search_samples:,:,:]),
  }
  return out


#@title Load Datasets

visualize = True #@param {type:"boolean"}
num_grid_search_samples = 6 #@param
num_test_samples =  6#@param

#@markdown Shepp Dataset
load_shepp = False #@param {type:"boolean"}

#@markdown ATLAS Dataset
load_atlas = True #@param {type:"boolean"}

datasets = {}
if load_shepp:
    print('Loading Shepp Dataset')
    datasets['shepp'] = get_shepp_dataset_3D(rand_key, num_grid_search_samples, num_test_samples)
    print('Shepp Dataset Loaded')
if load_atlas:
    print('Loading ATLAS Dataset')
    datasets['atlas'] = get_atlas_dataset_3D(rand_key, num_grid_search_samples, num_test_samples)
    print('ATLAS Dataset Loaded')

x1 = np.linspace(0, 1, RES+1)[:-1] # use full image resolution 
x_train = np.stack(np.meshgrid(x1,x1,x1), axis=-1)
x_test = x_train

def plot_dataset(dataset):
    plt.imshow(dataset['data_test'][0,:,:,0])
    plt.colorbar()
    plt.show()

if visualize:
    for dataset in datasets:
        print(f'Dataset {dataset}')
        plot_dataset(datasets[dataset])
        
#@title Define ReLU Network

network_depth = 4 #@param
network_width = 256 #@param

def make_network(num_layers, num_channels):
  layers = []
  for i in range(num_layers-1):
      layers.append(stax.Dense(num_channels))
      layers.append(stax.Relu)
  layers.append(stax.Dense(1))
  layers.append(stax.Sigmoid)
  return stax.serial(*layers)

init_fn, apply_fn = make_network(network_depth, network_width)

#@title Generate Fixed Embeddings

embedding_size = 256# 256 #@param

include_basic = False #@param {type:"boolean"}
include_posenc = False #@param {type:"boolean"}
#@markdown same as posenc, but with more samples
include_new_posenc = True #@param {type:"boolean"}
visualize = []

enc_dict = {}

input_encoder = lambda x, a, b: np.concatenate([a * np.sin((2.*np.pi*x) @ b.T), 
                                                a * np.cos((2.*np.pi*x) @ b.T)], axis=-1) #/ np.linalg.norm(a) * np.sqrt(a.shape[0])

def compute_new_posenc(mres):
  bvals = 2.**np.linspace(0,mres,embedding_size//3) - 1.
  bvals = np.stack([bvals, np.zeros_like(bvals), np.zeros_like(bvals)], -1)
  bvals = np.concatenate([bvals, np.roll(bvals, 1, axis=-1), np.roll(bvals, 2, axis=-1)], 0) 
  avals = np.ones((bvals.shape[0])) 
  return avals, bvals

def compute_basic():
  bvals = np.eye(3)
  avals = np.ones((bvals.shape[0])) 
  return avals, bvals

def visualize_encoders(enc_dict, keys=None):
    if keys is None:
        keys = enc_dict.keys()

    P = len(keys)
    plt.figure(figsize=(15,5))
    slices = {}
    for i, enc in enumerate(keys):
        plt.subplot(1,P,i+1)
        avals, bvals = enc_dict[enc]
        plt.scatter(bvals[:,0], bvals[:,1], marker='o', s=10, label=enc)
        plt.title(f'{enc} b values')
        plt.axis('equal')
    plt.show()
    
    
def mri_mask(shape, nsamp):
  mean = np.array(shape)//2
  cov = np.eye(len(shape)) * (2*shape[0])
  samps = random.multivariate_normal(rand_key, mean, cov, shape=(1,nsamp*10))[0,...].astype(np.int32)
  samps = np.unique(samps,axis=0)
  samps = samps[:nsamp]
  # Very dirty way to account for redundant samples.
  mask = np.zeros(shape)
  inds = []
  for i in range(samps.shape[-1]):
    inds.append(samps[...,i])
  #mask = index_update(mask, index[inds], 1.)
  mask = mask.at[tuple(inds)].set(1.) 
  return mask

def mri_mask_physical(shape, nsamp):
  z = shape[2]
  shape = shape[:2]
  mean = np.array(shape)//2
  cov = np.eye(len(shape)) * (2*shape[0])
  nsamp = nsamp//z 
  samps = random.multivariate_normal(rand_key, mean, cov, shape=(1,nsamp*10))[0,...].astype(np.int32)
  samps = np.unique(samps,axis=0)
  samps = samps[:nsamp]
  # Very dirty way to account for redundant samples.
  mask = np.zeros(shape)
  inds = []
  for i in range(samps.shape[-1]):
    inds.append(samps[...,i])
  #mask = index_update(mask, index[inds], 1.)
  mask = mask.at[tuple(inds)].set(1.) 
  mask = np.repeat(mask[:,:,np.newaxis],z,axis=2)
  return mask

@jit
def run_model(params, x, avals, bvals):
    if avals is not None:
        x = input_encoder(x, avals, bvals)
    return np.reshape(apply_fn(params, np.reshape(x, (-1, x.shape[-1]))), (x.shape[0], x.shape[1], x.shape[2]))
ifft= lambda x :  np.fft.ifft(np.fft.ifft(np.fft.ifft(x, axis=0), axis=1), axis=2)
fft= lambda x :  np.fft.fft(np.fft.fft(np.fft.fft(x, axis=0), axis=1), axis=2)
compute_mri = jit(lambda params, x, a, b, y, mask: np.fft.fft(np.fft.fft(np.fft.fft(run_model(params, x, a, b), axis=0), axis=1), axis=2)*mask.astype(np.complex64))
model_loss_mri = jit(lambda params, x, a, b, y, mask: .5 * np.mean(np.abs((compute_mri(params, x, a, b, y, mask) - y*mask.astype(np.complex64)) ** 2)))#
model_loss_mri = jit(lambda params, x, a, b, y, mask: .5 * np.mean(np.abs(run_model(params, x, a, b) - ifft(y*mask.astype(np.complex64)))**2))
model_loss = jit(lambda params, x, a, b, y, mask, image: .5 * np.abs(np.mean((np.clip(run_model(params, x, a, b), 0.0, 1.0) - image) ** 2)))
model_loss_us = jit(lambda params, x, a, b, y, mask, image: .5 * np.abs(np.mean((np.clip(np.abs(ifft(y*mask.astype(np.complex64))), 0.0, 1.0) - image) ** 2)))
model_psnr = jit(lambda params, x, a, b, y, mask, image: -10 * np.log10(2.*model_loss(params, x, a, b, y, mask, image)))
model_psnr_us = jit(lambda params, x, a, b, y, mask, image: -10 * np.log10(2.*model_loss_us(params, x, a, b, y, mask, image)))
model_grad_loss = jit(lambda params, x, a, b, y, mask, image: jax.grad(model_loss_mri)(params, x, a, b, y, mask))


GROUPS_MODEL = {'Test PSNR':[], 'Train PSNR':[],'Test PSNR undersampled':[], 'Train PSNR undersampled':[]}
def train_model(lr, iters, train_data, name='', plot_groups=None):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_update = jit(opt_update)

    if train_data[2] is not None:
      init_shape = train_data[2].shape[0]*2
    else:
      init_shape = 3
    _, params = init_fn(rand_key, (-1, init_shape))
    opt_state = opt_init(params)

    train_psnrs = []
    test_psnrs = []
    train_psnrs_us = []
    test_psnrs_us = []
    xs = []
    if plot_groups is not None:
        plot_groups['Test PSNR'].append(f'{name}_test')
        plot_groups['Train PSNR'].append(f'{name}_train')
        plot_groups['Test PSNR undersampled'].append(f'{name}_test_us')
        plot_groups['Train PSNR undersampled'].append(f'{name}_train_us')
    #train_data = [train_data[0],train_data[1], train_data[2], train_data[3]*train_data[4].astype(np.complex64),train_data[4],train_data[5]]
    for i in tqdm(range(iters), desc='train iter', leave=False):
        opt_state = opt_update(i, model_grad_loss(get_params(opt_state), *train_data), opt_state)
        if i % 25 == 0:
            train_psnr = model_psnr(get_params(opt_state), *train_data)
            test_psnr = train_psnr #model_psnr(get_params(opt_state), *test_data)
            train_psnr_us = model_psnr_us(get_params(opt_state), *train_data)
            test_psnr_us = train_psnr_us #model_psnr(get_params(opt_state), *test_data)
            train_psnrs.append(train_psnr)
            # test_psnrs.append(test_psnr)
            test_psnrs.append(test_psnr)
            train_psnrs_us.append(train_psnr_us)
            # test_psnrs.append(test_psnr)
            test_psnrs_us.append(test_psnr_us)
            xs.append(i)
            if plot_groups is not None:
                plotlosses_model.update({f'{name}_train':train_psnr, f'{name}_test':test_psnr,f'{name}_train_us':train_psnr_us, f'{name}_test_us':test_psnr_us}, current_step=i)
    if plot_groups is not None:
        plotlosses_model.send()
    set_trace()

    if plot_groups is not None:
        plotlosses_model.send()
    results = {
        'state': get_params(opt_state),
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'xs': xs,
        'final_test' : run_model(get_params(opt_state), train_data[0], train_data[1], train_data[2])
    }
    return results


def train_gridopt(lr, iters, train_data, name='', plot_groups=None):

    compute_mri = jit(lambda params, y, mask: np.fft.fft(np.fft.fft(np.fft.fft(params, axis=0), axis=1), axis=2)*mask.astype(np.complex64))
    model_loss_mri = jit(lambda params, x, a, b, y, mask: .5 * np.mean(np.abs((compute_mri(params, x, a, b, y, mask) - y*mask.astype(np.complex64)) ** 2)))
    model_loss = jit(lambda params, y, mask, image: .5 * np.abs(np.mean((np.clip(jax.nn.sigmoid(params), 0.0, 1.0) - image) ** 2)))
    model_psnr = jit(lambda params, y, mask, image: -10 * np.log10(2.*model_loss(params, y, mask, image)))
    model_grad_loss = jit(lambda params, y, mask, image: jax.grad(model_loss_mri)(params, y, mask))

    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_update = jit(opt_update)

    grid = np.zeros((RES, RES, RES))
    opt_state = opt_init(grid)

    train_psnrs = []
    test_psnrs = []
    xs = []
    if plot_groups is not None:
        plot_groups['Test PSNR'].append(f'{name}_test')
        plot_groups['Train PSNR'].append(f'{name}_train')
    for i in tqdm(range(iters), desc='train iter', leave=False):
        opt_state = opt_update(i, model_grad_loss(get_params(opt_state), *train_data), opt_state)
        if i % 25 == 0:
            train_psnr = model_psnr(get_params(opt_state), *train_data)
            test_psnr = train_psnr
            train_psnrs.append(train_psnr)
            test_psnrs.append(test_psnr)
            xs.append(i)
            if plot_groups is not None:
                plotlosses_model.update({f'{name}_train':train_psnr, f'{name}_test':test_psnr}, current_step=i)
        if i % 100 == 0 and i != 0 and plot_groups is not None:
            plotlosses_model.send()
    if plot_groups is not None:
        plotlosses_model.send()
    results = {
        'state': get_params(opt_state),
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'xs': xs,
        'final_test' : jax.nn.sigmoid(get_params(opt_state))
    }
    return results


#@title MRI parameters

nsamp = 200000#@param
mask = np.fft.fftshift(mri_mask((RES,RES,RES), nsamp)).astype(np.complex64)
#mask = np.fft.fftshift(mri_mask_physical((RES,RES,RES), nsamp)).astype(np.complex64)

print('sparsity:', np.sum(np.abs(mask))/np.prod(np.array(mask.shape)))


#@title Grid Search bval distributions
#@markdown The models are trained on a different set of data. The top distribution is added to the list.

lr =  2e-3#@param
training_steps =  1000#@param
target_distribution = "atlas" #@param ["shepp", "atlas"]
num_images =  1#@param
min_scale =  3#0.5#@param
max_scale =  3#3.5#@param
num_scales =  1#7#@param

bvals = random.normal(rand_key, (embedding_size, 3))
avals = np.ones((bvals.shape[0])) 
scales = np.linspace(min_scale, max_scale, num_scales)
print(f'searching over, {scales}')

if num_images == 1:
    plt_groups = {'Test PSNR':[], 'Train PSNR':[], "Test PSNR undersampled":[], "Train PSNR undersampled":[]}
    plotlosses_model = PlotLosses(groups=plt_groups)
else:
    plt_groups = None
result_psnrs = []
for scale in tqdm(scales, desc='Scale', leave=False):
    scale_results = []
    for i in tqdm(range(num_images), desc='Image', leave=False):
        image = datasets[target_distribution]['data_grid_search'][i,:,:,:]
        # y for MRI is just FFT (masking is in compute_mri)
        y_train = np.fft.fft(np.fft.fft(np.fft.fft(image, axis=0), axis=1), axis=2)
        train_data = (x_train, avals, bvals*scale, y_train, mask, image)
        scale_results.append(train_model(lr, training_steps, train_data, 
                                         name=scale, plot_groups=plt_groups)['test_psnrs'][-1])
    result_psnrs.append(scale_results)

result_psnrs = np.array(result_psnrs)
plt.errorbar(scales, np.mean(result_psnrs, axis=-1), yerr=np.std(result_psnrs, axis=-1))
plt.title('Grid search')
plt.xlabel('gaussian scale')
plt.ylabel('PSNR')
plt.show()

best_scale = scales[np.argmax(np.mean(result_psnrs, axis=-1))]
print(f'Adding gaussian scale {best_scale} to encoding methods')
enc_dict[f'gaussian_{"%.2f" % best_scale}'] = (avals, bvals*best_scale)

del result_psnrs

#@title Grid Search posenc distributions
#@markdown The models are trained on a different set of data. The top distribution is added to the list.

lr =  2e-3#@param
training_steps =  1000#@param
target_distribution = "atlas" #@param ["shepp", "atlas"]
num_images =  1#@param
min_scale =  2#@param
max_scale =  2#@param
scales = np.arange(min_scale, max_scale+1)

if num_images == 1:
    plt_groups = {'Test PSNR':[], 'Train PSNR':[]}
    plotlosses_model = PlotLosses(groups=plt_groups)
else:
    plt_groups = None
result_psnrs = []
for scale in tqdm(scales, desc='Scale', leave=False):
    avals, bvals = compute_new_posenc(scale)
    scale_results = []
    for i in tqdm(range(num_images), desc='Image', leave=False):
        image = datasets[target_distribution]['data_grid_search'][i,:,:,:]
        # y for MRI is just FFT (masking is in compute_mri)
        y_train = np.fft.fft(np.fft.fft(np.fft.fft(image, axis=0), axis=1), axis=2)
        train_data = (x_train, avals, bvals, y_train, mask, image)
        scale_results.append(train_model(lr, training_steps, train_data, 
                                         name=scale, plot_groups=plt_groups)['test_psnrs'][-1])
    result_psnrs.append(scale_results)

result_psnrs = np.array(result_psnrs)
plt.errorbar(scales, np.mean(result_psnrs, axis=-1), yerr=np.std(result_psnrs, axis=-1))
plt.title('Grid search')
plt.xlabel('posenc scale')
plt.ylabel('PSNR')
plt.show()

best_scale = scales[np.argmax(np.mean(result_psnrs, axis=-1))]
print(f'Adding posenc scale {best_scale} to encoding methods')
enc_dict[f'posenc_{"%.2f" % best_scale}'] = compute_new_posenc(best_scale)

del result_psnrs

#@title Define Final Experiment Parameters

params = {
    'shepp': {
        'lr': 2e-3,
        'no_enc_lr': 2e-3,
        'basic_lr': 2e-3,
        'gridopt_lr': 2e-3,
        'posenc_scale': 4,
        'gaussian_scale': 2,
    },
    'atlas': {
        'lr': 2e-3,
        'no_enc_lr': 2e-3,
        'basic_lr': 2e-3,
        'gridopt_lr': 2e-3,
        'posenc_scale': 4,
        'gaussian_scale': 5,
    }
}

#@title Train Models

training_steps =  1000#@param

target_distribution = "atlas" #@param ["shepp", "atlas"]

num_images =  1#@param

if num_images == 1:
    plt_groups = {'Test PSNR':[], 'Train PSNR':[], "Test PSNR undersampled":[], "Train PSNR undersampled":[]}
    plotlosses_model = PlotLosses(groups=plt_groups)
else:
    plt_groups = None

outputs = {}

# no encoding
lr = params[target_distribution]['no_enc_lr']
outputs['no_encoding'] = []
for i in tqdm(range(num_images), desc='no encoding', leave=False):
  image = datasets[target_distribution]['data_test'][i,:,:,:]
  y_train = np.fft.fft(np.fft.fft(np.fft.fft(image, axis=0), axis=1), axis=2)
  outputs['no_encoding'].append(train_model(lr, training_steps, 
                                            (x_train, None, None, y_train, mask, image), 
                                            name='no_encoding', plot_groups=plt_groups))


# basic encoding 
avals, bvals = compute_basic()
lr = params[target_distribution]['basic_lr']
outputs['basic'] = []
for i in tqdm(range(num_images), desc='basic', leave=False):
  image = datasets[target_distribution]['data_test'][i,:,:,:]
  y_train = np.fft.fft(np.fft.fft(np.fft.fft(image, axis=0), axis=1), axis=2)
  train_data = (x_train, avals, bvals, y_train, mask, image)
  outputs['basic'].append(train_model(lr, training_steps, train_data, name='basic', plot_groups=plt_groups))


# new posenc 
avals, bvals = compute_new_posenc(params[target_distribution]['posenc_scale'])
lr = params[target_distribution]['lr']
outputs['new_posenc'] = []
for i in tqdm(range(num_images), desc='new posenc', leave=False):
  image = datasets[target_distribution]['data_test'][i,:,:,:]
  y_train = np.fft.fft(np.fft.fft(np.fft.fft(image, axis=0), axis=1), axis=2)
  train_data = (x_train, avals, bvals, y_train, mask, image)
  outputs['new_posenc'].append(train_model(lr, training_steps, train_data, name='new_posenc', plot_groups=plt_groups))


# gaussian
bvals = random.normal(rand_key, (embedding_size, 3)) * params[target_distribution]['gaussian_scale']
avals = np.ones((bvals.shape[0])) 
lr = params[target_distribution]['lr']
outputs['gaussian'] = []
for i in tqdm(range(num_images), desc='gaussian', leave=False):
  image = datasets[target_distribution]['data_test'][i,:,:,:]
  y_train = np.fft.fft(np.fft.fft(np.fft.fft(image, axis=0), axis=1), axis=2)
  train_data = (x_train, avals, bvals, y_train, mask, image)
  outputs['gaussian'].append(train_model(lr, training_steps, train_data, name='gaussian', plot_groups=plt_groups))


# grid optimization baseline
outputs['gridopt'] = []
lr = params[target_distribution]['gridopt_lr']
for i in tqdm(range(num_images), desc='gridopt', leave=False):
  image = datasets[target_distribution]['data_test'][i,:,:,:]
  y_train = np.fft.fft(np.fft.fft(np.fft.fft(image, axis=0), axis=1), axis=2)
  train_data = (y_train, mask, image)
  test_data = train_data
  outputs['gridopt'].append(train_gridopt(lr, training_steps, train_data, name='gridopt', plot_groups=plt_groups))

# grid optimization baseline
outputs['gridopt'] = []
lr = 1e-1#params[target_distribution]['gridopt_lr']
for i in tqdm(range(num_images), desc='gridopt', leave=False):
  image = datasets[target_distribution]['data_test'][i,:,:,:]
  y_train = np.fft.fft(np.fft.fft(np.fft.fft(image, axis=0), axis=1), axis=2)
  train_data = (y_train, mask, image)
  test_data = train_data
  outputs['gridopt'].append(train_gridopt(lr, training_steps, train_data, name='gridopt', plot_groups=plt_groups))

#@title Plot Results

bar_graph = True #@param {type:"boolean"}
image_reconstructions = True #@param {type:"boolean"}
test_img_id =  1#@param

names = list(outputs.keys())

image_test = datasets[target_distribution]['data_test'][test_img_id,:,:,:]

xvals = np.arange(len(names))
test_values = np.array([[outputs[n][i]['test_psnrs'][-1] for i in range(len(outputs[n]))] for n in names])
test_values_mean = np.mean(test_values, axis=-1)
test_values_std = np.std(test_values, axis=-1)
train_values = np.array([[outputs[n][i]['train_psnrs'][-1] for i in range(len(outputs[n]))] for n in names])
train_values_mean = np.mean(train_values, axis=-1)
train_values_std = np.std(train_values, axis=-1)
inds = np.argsort(test_values_mean)
names_sort = [names[i] for i in inds]

if bar_graph:
    plt.figure(figsize=(20,5))
    plt.subplot(1,2,1)
    plt.bar(xvals+2, test_values_mean[inds], color=colors[0], alpha=.5, yerr=test_values_std)
    plt.xticks([])
    plt.ylim(test_values_mean.min()-test_values_std.max()-1, test_values_mean.max()+test_values_std.max()+1)
    plt.title(f'Fitting {target_distribution} Test')
    prnt_vals = ['%.2f' % x for x in test_values_mean[inds].tolist()]
    plt.table(cellText=[prnt_vals], 
              rowLabels=['PSNR'], 
              colLabels=names_sort, 
              loc='bottom', 
              bbox=[0, -.2, 1, 0.2])

    plt.subplot(1,2,2)
    plt.bar(xvals, train_values_mean[inds], color=colors[0], alpha=.5, yerr=train_values_std)
    # plt.xticks(xvals, names_sort, rotation=60)
    plt.xticks([])
    plt.ylim(train_values_mean.min()-train_values_std.max()-1, train_values_mean.max()+train_values_std.max()+1)
    plt.title(f'Fitting {target_distribution} Train')
    plt.table(cellText=[['%.2f' % x for x in train_values_mean[inds].tolist()]],
        rowLabels=['PSNR'],
        colLabels=names_sort,
        loc='bottom',
        bbox=[0, -.2, 1, 0.2])
    
    plt.show()

if image_reconstructions:
    print('----------------------------------------')
    print('                  Test')
    print('----------------------------------------')
    plt.figure(figsize=(28,6))
    for i, p in enumerate(names_sort):
        pred = outputs[p][test_img_id]['final_test']
        plt.subplot(1,len(names)+1,i+1)
        plt.imshow(pred[:,:,20])
        plt.title(p)

    plt.subplot(1,len(names)+1,len(names)+1)
    plt.imshow(image_test[:,:,20])
    plt.title('truth test')
    plt.show()
    
np.savez('3D_MRI_atlas.npz', outputs)
test_img_id = 1
image_test = datasets[target_distribution]['data_test'][test_img_id,:,:,:]
plt.figure(figsize=(28,6))
for i, p in enumerate(names_sort):
    pred = outputs[p][test_img_id]['final_test']
    plt.subplot(1,len(names)+1,i+1)
    plt.imshow(pred[:,:,35])
    plt.title(p)

plt.subplot(1,len(names)+1,len(names)+1)
plt.imshow(image_test[:,:,35])
plt.title('truth test')
plt.show()
ind = 1
slice = 35
pred_ne = outputs['no_encoding'][ind]['final_test'][:,:,slice]
pred_g = outputs['gaussian'][ind]['final_test'][:,:,slice]
plt.imsave('3D_MRI_no_encoding.png', pred_ne, cmap='gray')
plt.imsave('3D_MRI_gaussian_encoding.png', pred_g, cmap='gray')
