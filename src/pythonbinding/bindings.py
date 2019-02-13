from ctypes import *
from time import *


class DllHolder:
    def __init__(self):
        self.core = cdll.LoadLibrary("core.dll")
        self.mffLoader = cdll.LoadLibrary("mffloader.dll")


class DllInterface:

    def __init__(self):
        self.dllHolder = DllHolder()
        self.dllHolder.core.mufflon_initialize()
        self.disable_profiling()

    def disable_profiling(self):
        self.dllHolder.core.profiling_disable()
        self.dllHolder.mffLoader.loader_profiling_disable()

    def loader_load_json(self, szeneJson):
        return self.dllHolder.mffLoader.loader_load_json(c_char_p(szeneJson.encode('utf-8')))

    def renderer_set_parameter_bool(self, parameterName, value):
        return self.dllHolder.core.renderer_set_parameter_bool(c_char_p(parameterName.encode('utf-8')), c_bool(value))

    def renderer_set_parameter_float(self, parameterName, value):
        return self.dllHolder.core.renderer_set_parameter_float(c_char_p(parameterName.encode('utf-8')), c_float(value))

    def renderer_set_parameter_int(self, parameterName, value):
        return self.dllHolder.core.renderer_set_parameter_int(c_char_p(parameterName.encode('utf-8')), c_int32(value))

    def render_iterate(self):
        return self.dllHolder.core.render_iterate()

    def render_get_current_iteration(self):
        return self.dllHolder.core.render_get_current_iteration()

    def render_save_screenshot(self, fileName, targetIndex, variance):
        return self.dllHolder.core.render_save_screenshot(c_char_p(fileName.encode('utf-8')), c_uint32(targetIndex), c_uint32(variance))

    def render_get_render_target_count(self):
        return self.dllHolder.core.render_get_render_target_count()

    def render_is_render_target_enabled(self, targetIndex, variance):
        return self.dllHolder.core.render_is_render_target_enabled(c_uint32(targetIndex), c_bool(variance))

    def render_enable_render_target(self, targetIndex, variance):
        return self.dllHolder.core.render_enable_render_target(c_uint32(targetIndex), c_bool(variance))

    def render_enable_renderer(self, rendererIndex):
        return self.dllHolder.core.render_enable_renderer(c_uint32(rendererIndex))

    def render_is_render_target_variance_enabled(self, targetIndex):
        return False

    def render_get_active_scene_name(self):
        return "scene_name"

    def render_get_active_scenario_name(self):
        return "scenario_name"

    def render_get_active_renderer_name(self):
        return "renderer_name"

    def render_get_target_name(self, index):
        return "target_name"


class RenderActions:
    screenshotPattern = "#scene-#scenario-#renderer-#iteration"

    def __init__(self):
        self.dllInterface = DllInterface()

    def load_json(self, szeneJson):
        returnValue = self.dllInterface.loader_load_json(szeneJson)
        self.dllInterface.render_enable_render_target(0, False)
        return returnValue

    def enable_renderer(self, rendererIndex):  # TODO: enable by string
        self.dllInterface.render_enable_renderer(rendererIndex)

    def enable_render_target(self, targetIndex, variance):
        self.dllInterface.render_enable_render_target(targetIndex, variance)

    def take_screenshot(self, iterationNr):
        filename = self.screenshotPattern
        filename = filename.replace("#scene", self.dllInterface.render_get_active_scene_name(), 1)
        filename = filename.replace("#scenario", self.dllInterface.render_get_active_scenario_name(), 1)
        filename = filename.replace("#renderer", self.dllInterface.render_get_active_renderer_name(), 1)
        filename = filename.replace("#iteration", str(iterationNr), 1)

        for targetIndex in range(self.dllInterface.render_get_render_target_count()):
            if self.dllInterface.render_is_render_target_enabled(targetIndex, False):
                fName = filename.replace("#target", str(self.dllInterface.render_get_target_name(targetIndex)), 1)
                self.dllInterface.render_save_screenshot(fName, targetIndex, False)
            if self.dllInterface.render_is_render_target_enabled(targetIndex, True):
                fName = filename.replace("#target", str(self.dllInterface.render_get_target_name(targetIndex)) + "(Variance)", 1)
                self.dllInterface.render_save_screenshot(fName, targetIndex, True)

    def render_for_iterations(self, iterationCount):
        for i in range(iterationCount):
            self.dllInterface.render_iterate()
        self.take_screenshot(self.dllInterface.render_get_current_iteration())

    def render_for_seconds(self, secondsToRender):
        startTime = process_time()
        curTime = startTime
        while curTime - startTime < secondsToRender:
            self.dllInterface.render_iterate()
            curTime = process_time()
        self.take_screenshot(self.dllInterface.render_get_current_iteration())
