import bindings


def render():
    renderer = bindings.RenderActions()
    if not renderer.enable_renderer("Pathtracer"):
        print("Could not find specified renderer")
        return
    if renderer.load_json(sceneJson="Path/To/Scene.json") != 0:  # 0 == Loader Succeded
        return
    if not renderer.load_scenario("TestScenario"):
        print("Could not find specified scenario")
        return

    renderer.render_for_seconds(10)
    renderer.render_for_iterations(10)

render()