import jax.numpy as jnp
import jax
from jax import jacobian, hessian, grad
from fim.dictionaryconversions import flatten_dictionary, unflatten_dictionary
from fim import solver
from fim.gwlikelihood import get_gw_likelihood
from fim.imagelikelihood import get_image_likelihood

# NOTE: NOT JAX-COMPATIBLE!
def compute_phi_im(phi, fixed_parameters, lens_model_list=['SIS']):
    # Solve the lens equation and get the image parameters
    phi_unflattened = unflatten_dictionary( phi ) # Same as kwargs_params
    x_img, y_img, arrival_times, magnifications = solver.solve_lens_equation( phi_unflattened, lens_model_list, fixed_parameters=fixed_parameters ) # Solve the lens equation
    n_img = len(x_img) # Number of images

    # Replace the source position parameters in the dictionary with the image parameters
    phi_im = phi.copy() # Copy the dictionary
    del phi_im['gw-0-src_center_x']
    del phi_im['gw-0-src_center_y']
    for i in range(n_img):
        phi_im['gw-0-img_center_x_%d' % i] = x_img[i]
        phi_im['gw-0-img_center_y_%d' % i] = y_img[i]
    return phi_im

from herculens.MassModel.mass_model import MassModel
from fim.gwlikelihood import get_images

from copy import deepcopy
def compute_phi( phi_im, fixed_parameters, lens_model_list=['SIS']):
    # Get the phi_im_parameters
    phi_im_unflattened = unflatten_dictionary(phi_im)
    lens_mass_model = MassModel(lens_model_list)
    # Get the image position parameters:
    x_img, y_img = get_images(phi_im)
    n_img = len(x_img)
    x_src, y_src = lens_mass_model.ray_shooting(x_img, y_img, phi_im_unflattened['kwargs_lens'])
    # print("x_src, y_src", x_src, y_src)
    phi = deepcopy(phi_im)
    phi['gw-0-src_center_x'] = jnp.mean(x_src)
    phi['gw-0-src_center_y'] = jnp.mean(y_src)
    for i in range(n_img):
        del phi['gw-0-img_center_x_%d' % i]
        del phi['gw-0-img_center_y_%d' % i]
    return phi

def get_log_likelihood( kwargs_likelihood, lens_model_list):
    # Get individual likelihoods
    log_likelihood_gw = get_gw_likelihood( kwargs_likelihood['kwargs_gw_likelihood'], lens_model_list )
    log_likelihood_image = get_image_likelihood( kwargs_likelihood['kwargs_image_likelihood'] )
    # Define the log-likelihood function
    def log_likelihood(phi_im):
        # Compute the gravitational-wave likelihood
        # print(log_likelihood_gw(phi_im), log_likelihood_image(phi_im))
        return log_likelihood_gw(phi_im) + log_likelihood_image(phi_im)
    return log_likelihood

# Assume 100% errors on the priors for now as a default
def default_log_prior( phi_im, phi_im_maxP ):
    phi_im_keys = list(phi_im.keys())
    phi_diff = jnp.array([phi_im[phi_im_keys[i]] - phi_im_maxP[phi_im_keys[i]] for i in range(len(phi_im_keys))])
    variances = jnp.array([phi_im_maxP[phi_im_keys[i]] for i in range(len(phi_im_keys))])**2+0.1
    return jnp.sum(phi_diff**2 /(2* variances)) # 100% errors on the priors


def compute_inverse_covariance_matrix( kwargs_params_maxP, kwargs_likelihood, fixed_parameters, log_prior=None, lens_model_list=['SIS'] ):
    if log_prior is not None:
        raise NotImplementedError("Only default log prior implemented for now")
    log_prior = default_log_prior
    # This monstrosity is needed because of how herculens/lenstronomy is coded up using dictionaries (instead of arrays), and how JAX hates nested dictionaries
    phi_maxP = flatten_dictionary( kwargs_params_maxP ) # Flattened dictionary (guaranteed to not be a nested dictionary)
    phi_unflattened_maxP = unflatten_dictionary( phi_maxP ) # Same as kwargs_params
    # Transform the image parameters to a dictionary: 
    phi_im_maxP = compute_phi_im(phi_maxP, lens_model_list=lens_model_list, fixed_parameters=fixed_parameters)

    # Create the likelihood function 
    log_likelihood = get_log_likelihood( kwargs_likelihood, lens_model_list=lens_model_list )
    log_posterior = lambda phi_im: log_likelihood(phi_im) + default_log_prior(phi_im, phi_im_maxP) # Posterior function
    
    # Take the hessian with respect to phi_im_maxP:
    hess_log_likelihood = hessian(log_posterior)(phi_im_maxP) # Hessian of the log-likelihood function
    # Print it
    # print("Hessian of the log-likelihood function: ", hess_log_likelihood)
    # Transform into matrix
    keys = list(hess_log_likelihood.keys())
    hessian_matrix_form = jnp.array([[hess_log_likelihood[keys[i]][keys[j]] for j in range(len(keys))] for i in range(len(keys))])

    # Compute dphi/dphi_im Jacobian:
    phi_func = lambda phi_im: compute_phi(phi_im, fixed_parameters=fixed_parameters, lens_model_list=lens_model_list)
    jac = jacobian(phi_func)(phi_im_maxP) # Jacobian of the phi function
    # Convert to a matrix
    keys1 = list(jac.keys())
    keys2 = list(jac[keys1[0]].keys())
    jac_matrix_form = jnp.array([[jac[keys1[i]][keys2[j]] for j in range(len(keys2))] for i in range(len(keys1))])
    jac_pinv = jnp.linalg.pinv(jac_matrix_form) # Pseudo-inverse of the Jacobian matrix

    # Compute the hessian matrix in the new coordinates:
    hessian_matrix_form_new = -1.*jac_pinv.T @ hessian_matrix_form @ jac_pinv
    # print(jnp.shape(hessian_matrix_form), jnp.shape(jac_matrix_form), jnp.shape(hessian_matrix_form_new), len(list(phi_maxP.keys())), len(list(phi_im_maxP.keys())))
    # exit(1)

    # Transform back into a dictionary:
    hess_log_likelihood = {}
    keys = list(phi_maxP.keys())
    for i in range(len(keys)):
        # print(keys[i])
        hess_log_likelihood[keys[i]] = {}
        for j in range(len(keys)):
            hess_log_likelihood[keys[i]][keys[j]] = hessian_matrix_form_new[i][j]

    return keys, hess_log_likelihood # Return the log-likelihood function
