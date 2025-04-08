import jax.numpy as jnp
from fim import solver
from fim.defaults import cosmo
from fim.dictionaryconversions import flatten_dictionary, unflatten_dictionary
from herculens.MassModel.mass_model import MassModel

def get_images( phi_im ):
    # Get the number of images from the dictionary
    n_img = len([key for key in phi_im.keys() if 'gw-0-img_center_x_' in key])
    # Get the image position parameters:
    x_img = jnp.array([phi_im['gw-0-img_center_x_%d' % i] for i in range(n_img)])
    y_img = jnp.array([phi_im['gw-0-img_center_y_%d' % i] for i in range(n_img)])
    return x_img, y_img

def get_fermat_potentials_magnifications(phi_im, lens_model_list):
    # Get the phi_im_parameters
    phi_im_unflattened = unflatten_dictionary(phi_im)
    lens_mass_model = MassModel(lens_model_list)
    # Get the image position parameters:
    x_img, y_img = get_images(phi_im)
    # Get the image arrival time delays and magnifications
    # print("phi_im_unflattened['kwargs_lens']", phi_im_unflattened['kwargs_lens'], x_img, y_img)
    fermat_potentials = lens_mass_model.fermat_potential(x_img, y_img, kwargs_lens=phi_im_unflattened['kwargs_lens'])
    magnifications = lens_mass_model.magnification(x_img, y_img, kwargs=phi_im_unflattened['kwargs_lens'])
    return x_img, y_img, fermat_potentials, magnifications

def get_gw_likelihood_params(kwargs_params, log_sigma_t, log_sigma_d, lens_model_list, fixed_parameters):
    # Get the true time delays and effective luminosity distances (assumed to be noiseless)
    _, _, fermat_potentials, magnifications = solver.solve_lens_equation( kwargs_params, lens_model_list, fixed_parameters=fixed_parameters)
    deltat = jnp.diff( fermat_potentials ) # Unnormalised time delays (fermat potentials)
    luminosity_distance = fixed_parameters['luminosity_distance'] # True luminosity distance
    luminosity_distance_eff = luminosity_distance / jnp.sqrt(jnp.abs(magnifications))
    # Gravitational-wave likelihood:
    kwargs_gw_likelihood = {}
    kwargs_gw_likelihood['log_delta_t_maxp'] = jnp.log(deltat)
    kwargs_gw_likelihood['log_luminosity_distance_eff_maxp'] = jnp.log(luminosity_distance_eff)
    kwargs_gw_likelihood['log_sigma_t'] = jnp.ones(len(deltat)) * log_sigma_t
    kwargs_gw_likelihood['log_sigma_d'] = jnp.ones(len(luminosity_distance_eff)) * log_sigma_d
    kwargs_gw_likelihood['luminosity_distance_maxp'] = luminosity_distance
    # Return the gravitational-wave likelihood parameters
    return kwargs_gw_likelihood 

def get_gw_likelihood( kwargs_gw_likelihood, lens_model_list ):
    # Get the gravitational-wave likelihood parameters
    log_delta_t_maxp = kwargs_gw_likelihood['log_delta_t_maxp']
    log_luminosity_distance_eff_maxp = kwargs_gw_likelihood['log_luminosity_distance_eff_maxp']
    log_sigma_t = kwargs_gw_likelihood['log_sigma_t']
    log_sigma_d = kwargs_gw_likelihood['log_sigma_d']
    luminosity_distance_maxp = kwargs_gw_likelihood['luminosity_distance_maxp']

    # Define the log-likelihood function
    def log_likelihood(phi_im):
        phi_im_unflattened = unflatten_dictionary(phi_im)
        # print("phi_im", phi_im)
        x_img, y_img, fermat_potentials, magnifications = get_fermat_potentials_magnifications(phi_im, lens_model_list)
        # print("x_img, y_img, fermat_potentials, magnifications", x_img, y_img, fermat_potentials, magnifications)
        # Compute the gravitational-wave likelihood
        deltat = jnp.diff(fermat_potentials)
        luminosity_distance_eff = luminosity_distance_maxp / jnp.sqrt(jnp.abs(magnifications))
        # print(len(luminosity_distance_eff), len(log_luminosity_distance_eff_maxp), len(log_sigma_d))
        return -0.5 * jnp.sum((deltat - log_delta_t_maxp)**2 / log_sigma_t**2) \
            - 0.5 * jnp.sum((jnp.log(luminosity_distance_eff) - log_luminosity_distance_eff_maxp)**2 / log_sigma_d**2)

    return log_likelihood
