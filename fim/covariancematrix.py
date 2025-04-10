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

def get_log_likelihood( kwargs_likelihood, lens_model_list):
    # Get individual likelihoods
    log_likelihood_gw = get_gw_likelihood( kwargs_likelihood['kwargs_gw_likelihood'], lens_model_list )
    log_likelihood_image = get_image_likelihood( kwargs_likelihood['kwargs_image_likelihood'] )
    # Define the log-likelihood function
    def log_likelihood(phi_im):
        # Compute the gravitational-wave likelihood
        print(log_likelihood_gw(phi_im), log_likelihood_image(phi_im))
        return log_likelihood_gw(phi_im) + log_likelihood_image(phi_im)
    return log_likelihood

def compute_covariance_matrix( kwargs_params_maxP, kwargs_likelihood, fixed_parameters, log_prior=None, lens_model_list=['SIS'] ):
    # This monstrosity is needed because of how herculens/lenstronomy is coded up using dictionaries (instead of arrays), and how JAX hates nested dictionaries
    phi_maxP = flatten_dictionary( kwargs_params_maxP ) # Flattened dictionary (guaranteed to not be a nested dictionary)
    phi_unflattened_maxP = unflatten_dictionary( phi_maxP ) # Same as kwargs_params
    # Transform the image parameters to a dictionary: 
    phi_im_maxP = compute_phi_im(phi_maxP, lens_model_list=lens_model_list, fixed_parameters=fixed_parameters)

    # Create the likelihood function 
    log_likelihood = get_log_likelihood( kwargs_likelihood, lens_model_list=lens_model_list )

    return log_likelihood( phi_im_maxP ) # Return the log-likelihood function