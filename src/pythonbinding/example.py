import bindings
from bindings import LogLevel
from bindings import Device


def render():
    renderer = bindings.RenderActions()
    renderer.set_renderer_log_level(LogLevel.PEDANTIC)
    renderer.set_loader_log_level(LogLevel.WARNING)
    
    # Name is case insensitive
    renderer.enable_renderer("PT", Device.CPU | Device.CUDA)
    #renderer.enable_renderer("pt", Device.CUDA)
    #renderer.enable_renderer("patHtraCer", Device.CPU)
    
    
    renderer.load_json(sceneJson="pathToScene.json")
    renderer.load_scenario("ScenarioName")
    renderer.set_current_animation_frame(3)
    print(renderer.get_start_animation_frame(), " - ", renderer.get_current_animation_frame(),
          " - ", renderer.get_end_animation_frame())
    renderer.renderer_set_parameter_float("FloatParam", 1.0);
    renderer.renderer_set_parameter_enum("EnumParam", "EnumValueName");

    renderer.render_for_seconds(10)
    # Printing defaults to False
    renderer.render_for_iterations(16, printProgress=True, progressSteps=8)

render()