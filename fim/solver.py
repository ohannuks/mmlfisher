# import helens # Helens seems to be somewhat unstable so we use lenstronomy instead
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model import LensModel
from fim.defaults import cosmo


def solve_lens_equation(kwargs_parameters, lens_model_list, fixed_parameters):
    """
    Solves the gravitational lens equation to determine image positions, 
    arrival time delays, and magnifications for a given source and lens model.

    Args:
        kwargs_parameters (dict): Dictionary containing lens and source parameters.
            - 'kwargs_gw': List of dictionaries with gravitational wave parameters, 
              including 'src_center_x', 'src_center_y', 'z_lens', and 'z_source'.
            - 'kwargs_lens': List of dictionaries with lens model parameters.
        lens_model_list (list): List of lens model names defining the mass distribution.

    Returns:
        tuple: A tuple containing:
            - x_img (list): X-coordinates of the image positions.
            - y_img (list): Y-coordinates of the image positions.
            - arrival_times (list): Time delays for the images.
            - magnifications (list): Magnifications of the images.
    """
    # Get GW parameters
    src_x, src_y = kwargs_parameters['kwargs_gw'][0]['src_center_x'], kwargs_parameters['kwargs_gw'][0]['src_center_y']
    z_lens, z_source = fixed_parameters['zl'], fixed_parameters['zs']
    # Specify the lens mass model
    # lens_model = LensModel(lens_model_list, z_lens=z_lens, z_source=z_source, cosmo=cosmo)
    lens_model = LensModel(lens_model_list, cosmo=cosmo)
    # Get the image position from the source position
    lens_eq_solver = LensEquationSolver(lens_model)
    x_img, y_img = lens_eq_solver.image_position_from_source(sourcePos_x=src_x, sourcePos_y=src_y, kwargs_lens=kwargs_parameters['kwargs_lens'])
    # Get the image arrival time delays and magnifications
    fermat_potentials = lens_model.fermat_potential(x_img, y_img, kwargs_lens=kwargs_parameters['kwargs_lens'])
    magnifications = lens_model.magnification(x_img, y_img, kwargs=kwargs_parameters['kwargs_lens'])
    return x_img, y_img, fermat_potentials, magnifications