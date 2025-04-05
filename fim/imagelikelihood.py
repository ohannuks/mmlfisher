from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.psf import PSF
from herculens.MassModel.mass_model import MassModel
from herculens.LightModel.light_model import LightModel
from fim.dictionaryconversions import flatten_dictionary, unflatten_dictionary
import jax.numpy as jnp
from copy import deepcopy

def get_image_likelihood_params(lens_model_list=['SIS'], source_light_model_list=['SERSIC_ELLIPSE'], npix=80, pix_scl=0.08, fwhm=0.3):
    """
    Generates parameters for image likelihood computation in a lensing model.

    Args:
        lens_model_list (list, optional): List of lens mass models. Defaults to ['SIS'].
        source_light_model_list (list, optional): List of source light models. Defaults to ['SERSIC_ELLIPSE'].
        npix (int, optional): Number of pixels along one axis of the grid. Defaults to 80.
        pix_scl (float, optional): Pixel scale in angular units. Defaults to 0.08.
        fwhm (float, optional): Full width at half maximum of the PSF. Defaults to 0.3.

    Returns:
        dict: Dictionary containing parameters for image likelihood computation.
    """
    # Pixel grid parameters
    half_size = npix * pix_scl / 2
    transform_pix2angle = pix_scl * jnp.eye(2)  # transformation matrix pixel <-> angle
    kwargs_pixel = {'nx': npix, 'ny': npix, 'ra_at_xy_0': -half_size + pix_scl / 2, 'dec_at_xy_0': -half_size + pix_scl / 2, 'transform_pix2angle': pix_scl * jnp.eye(2)}
    pixel_grid = PixelGrid(**kwargs_pixel)

    # PSF:
    psf = PSF(psf_type='GAUSSIAN', fwhm=fwhm, pixel_size=pix_scl)

    # Lens mass
    lens_mass_model_class = MassModel(lens_model_list)

    # Source light
    source_model_class = LightModel(source_light_model_list)

    # Defaults for testing FIXME: To be changed later
    lens_light_model_class = None
    noise_class = None

    # Define image likelihood parameters
    kwargs_numerics = {'supersampling_factor': 5}
    kwargs_image_likelihood = dict( pixel_grid=pixel_grid, 
                                    psf=psf, 
                                    noise_class=noise_class,
                                    lens_mass_model_class=lens_mass_model_class,
                                    source_model_class=source_model_class,
                                    lens_light_model_class=lens_light_model_class,
                                    kwargs_numerics=kwargs_numerics )
    return kwargs_image_likelihood

from herculens.LensImage.lens_image import LensImage
def get_image_likelihood( kwargs_params, lens_model_list ):
    # lens_image = LensImage(deepcopy(pixel_grid), deepcopy(psf), noise_class=deepcopy(noise),
    #                      lens_mass_model_class=deepcopy(lens_mass_model_input),
    #                      source_model_class=deepcopy(source_model_input),
    #                      lens_light_model_class=deepcopy(lens_light_model_input),
    #                      kwargs_numerics=kwargs_numerics_fit)
    lens_image = LensImage(**kwargs_params)
    def log_likelihood(phi_im):
        phi_im_unflattened = unflatten_dictionary(phi_im)
        model_image = lens_image.image(phi_im_unflattened)
        residual = 