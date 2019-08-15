import bindings
from bindings import LogLevel
from bindings import Device


def render():
    renderer = bindings.RenderActions()
    # Log levels can be set differently for loader and renderer
    renderer.set_renderer_log_level(LogLevel.PEDANTIC)
    renderer.set_loader_log_level(LogLevel.WARNING)
    
    # Name is case insensitive
    renderer.enable_renderer("PT", Device.CPU)# | Device.CUDA)
    # Name is NOT case insensitive! There may be a 'Radiance' and a 'RaDiAnCe' target at the same time
    # Second parameter determines whether variance should be captured as well
    renderer.enable_render_target("Radiance", False);
    renderer.enable_render_target("Normal", True);
    
    # Example file path to one of our testscenes (the default render target defaults to 'Radiance',
	# which we don't care about because we already enabled them prior)
    renderer.load_json(sceneJson="../../testscenes/material/blender_mat_preview.json")
    renderer.load_scenario("Glossy Orange")
    # Animation frames are implicitly clipped in the available range and start at 0
    renderer.set_current_animation_frame(3)
    # If desired you may loop over frame ranges by querying the available range
    print(renderer.get_start_animation_frame(), " - ", renderer.get_current_animation_frame(),
          " - ", renderer.get_end_animation_frame())
    # Setting renderer parameters is categorized by type (names are case sensitive!)
    renderer.renderer_set_parameter_int("Max. path length", 8);
    # You'll receive a warning (if the log level is high enough) for parameters that don't exist
    renderer.renderer_set_parameter_float("FakeFloatParam", 1.7);
    renderer.renderer_set_parameter_enum("FakeEnumParam", "EnumValueName");

    # The screenshot pattern has multiple options to bake configuration details into them;
    # see the function 'take_screenshot' for more
    renderer.screenshotPattern = "screenshot_folder/" + renderer.screenshotPattern

    # Screenshots are implicitly taken and are (partially) opt-out
    renderer.render_for_seconds(5)
    # Printing defaults to False
    renderer.render_for_iterations(16, printProgress=True, progressSteps=8, takeScreenshot=False)
    # For manual screenshot taking there are two options: take_screenshot saves all currently
    # enabled render targets, while take_denoised_screenshot looks for a render target 'Radiance'
    # and uses that to save a denoised version of said render target; it optionally incorporates
    # the targets 'Normal' and 'Albedo', if they exist and are enabled
    renderer.take_screenshot(16)
    # You can also disable render targets (but not enable - not recorded is not recorded)
    # prior to screenshotting; here the second parameter indicates whether ONLY the variance
    # should be disabled
    renderer.disable_render_target("Normal", False);
    renderer.take_denoised_screenshot(16)

render()