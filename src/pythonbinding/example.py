import bindings


def render():
    renderer = bindings.RenderActions()
    renderer.enable_renderer(0)
    if renderer.load_json(szeneJson="Path/To/Scene.json") != 0:  # 0 == Loader Succeded
        return

    renderer.render_for_seconds(10)
    renderer.render_for_iterations(10)

render()