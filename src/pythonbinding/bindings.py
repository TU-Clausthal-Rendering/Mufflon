from ctypes import *
from time import *
import ntpath

class ProcessTime(Structure):
    _fields_ = [
        ("cycles", c_ulonglong),
        ("microseconds", c_ulonglong)
    ]

class DllHolder:
    def __init__(self):
        self.core = cdll.LoadLibrary("core.dll")
        self.mffLoader = cdll.LoadLibrary("mffloader.dll")


class DllInterface:

    def __init__(self):
        self.dllHolder = DllHolder()
        self.dllHolder.core.mufflon_initialize()
        self.disable_profiling()
        self.dllHolder.core.render_get_renderer_name.restype = c_char_p
        self.dllHolder.core.render_get_render_target_name.restype = c_char_p
        self.dllHolder.core.render_iterate.argtypes = [ POINTER(ProcessTime), POINTER(ProcessTime), POINTER(ProcessTime) ]
        self.dllHolder.core.scenario_get_name.restype = c_char_p
        self.dllHolder.core.world_get_current_scenario.restype = POINTER(c_int)
        self.dllHolder.core.world_find_scenario.restype = c_void_p
        self.dllHolder.core.world_load_scenario.restype = c_void_p
        
    def __del__(self):
        self.dllHolder.core.mufflon_destroy()

    def disable_profiling(self):
        self.dllHolder.core.profiling_disable()
        self.dllHolder.mffLoader.loader_profiling_disable()

    def loader_load_json(self, sceneJson):
        return self.dllHolder.mffLoader.loader_load_json(c_char_p(sceneJson.encode('utf-8')))

    def renderer_set_parameter_bool(self, parameterName, value):
        return self.dllHolder.core.renderer_set_parameter_bool(c_char_p(parameterName.encode('utf-8')), c_bool(value))

    def renderer_set_parameter_float(self, parameterName, value):
        return self.dllHolder.core.renderer_set_parameter_float(c_char_p(parameterName.encode('utf-8')), c_float(value))

    def renderer_set_parameter_int(self, parameterName, value):
        return self.dllHolder.core.renderer_set_parameter_int(c_char_p(parameterName.encode('utf-8')), c_int32(value))

    def render_iterate(self):
        iterateTime = ProcessTime(0,0)
        preTime = ProcessTime(0,0)
        postTime = ProcessTime(0,0)
        self.dllHolder.core.render_iterate(byref(iterateTime), byref(preTime), byref(postTime))
        return iterateTime, preTime, postTime

    def render_reset(self):
        return self.dllHolder.core.render_reset()

    def render_get_current_iteration(self):
        return self.dllHolder.core.render_get_current_iteration()

    def render_save_screenshot(self, fileName, targetIndex, variance):
        return self.dllHolder.core.render_save_screenshot(c_char_p(fileName.encode('utf-8')), c_uint32(targetIndex), c_uint32(variance))

    def render_get_render_target_count(self):
        return self.dllHolder.core.render_get_render_target_count()

    def render_get_render_target_name(self, index):
        return self.dllHolder.core.render_get_render_target_name(c_uint32(index)).decode()

    def render_is_render_target_enabled(self, targetIndex, variance):
        return self.dllHolder.core.render_is_render_target_enabled(c_uint32(targetIndex), c_bool(variance))

    def render_enable_render_target(self, targetIndex, variance):
        return self.dllHolder.core.render_enable_render_target(c_uint32(targetIndex), c_bool(variance))
		
    def render_disable_render_target(self, targetIndex, variance):
        return self.dllHolder.core.render_disable_render_target(c_uint32(targetIndex), c_bool(variance))

    def render_enable_renderer(self, rendererIndex):
        return self.dllHolder.core.render_enable_renderer(c_uint32(rendererIndex))

    def render_get_renderer_count(self):
        return self.dllHolder.core.render_get_renderer_count()

    def render_get_renderer_name(self, rendererIndex):
        return self.dllHolder.core.render_get_renderer_name(c_uint32(rendererIndex)).decode()

    def render_get_active_scenario_name(self):
        return self.dllHolder.core.scenario_get_name( self.dllHolder.core.world_get_current_scenario()).decode()

    def world_find_scenario(self, name):
        return self.dllHolder.core.world_find_scenario(c_char_p(name.encode('utf-8')))

    def world_load_scenario(self, hdl):
        return self.dllHolder.core.world_load_scenario(c_void_p(hdl))


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


class RenderActions:
    screenshotPattern = "#scene-#scenario-#renderer-#iteration"
    sceneName = ""
    activeRendererName = ""

    def __init__(self):
        self.dllInterface = DllInterface()

    def load_json(self, sceneJson):
        fileName = path_leaf(sceneJson)
        self.sceneName = fileName.split(".")[0]
        returnValue = self.dllInterface.loader_load_json(sceneJson)
        self.dllInterface.render_enable_render_target(0, False)
        return returnValue

    def enable_renderer(self, rendererString):
        for i in range(self.dllInterface.render_get_renderer_count()):
            name = self.dllInterface.render_get_renderer_name(i)
            if name == rendererString:
                self.activeRendererName = rendererString
                self.dllInterface.render_enable_renderer(i)
                return True
        return False
        
    def load_scenario(self, scenarioName):
        hdl = self.dllInterface.world_find_scenario(scenarioName)
        if not hdl:
            return False
        if self.dllInterface.world_load_scenario(hdl):
            return True
        return False

    def enable_render_target(self, targetIndex, variance):
        self.dllInterface.render_enable_render_target(targetIndex, variance)
		
    def disable_render_target(self, targetIndex, variance):
        self.dllInterface.render_disable_render_target(targetIndex, variance)

    def take_screenshot(self, iterationNr, iterateTime=ProcessTime(0,0), preTime=ProcessTime(0,0), postTime=ProcessTime(0,0)):
        filename = self.screenshotPattern
        filename = filename.replace("#scene", self.sceneName, 1)
        filename = filename.replace("#scenario", self.dllInterface.render_get_active_scenario_name(), 1)
        filename = filename.replace("#renderer", self.activeRendererName, 1)
        filename = filename.replace("#iteration", str(iterationNr), 1)
        filename = filename.replace("#iterateTime", str(iterateTime.microseconds / 1000) + "ms", 1)
        filename = filename.replace("#preTime", str(iterateTime.microseconds / 1000) + "ms", 1)
        filename = filename.replace("#postTime", str(iterateTime.microseconds / 1000) + "ms", 1)
        filename = filename.replace("#iterateCycles", str(iterateTime.cycles / 1000000) + "MCycles", 1)
        filename = filename.replace("#preCycles", str(preTime.cycles / 1000000) + "MCycles", 1)
        filename = filename.replace("#postCycles", str(postTime.cycles / 1000000) + "MCycles", 1)

        for targetIndex in range(self.dllInterface.render_get_render_target_count()):
            if self.dllInterface.render_is_render_target_enabled(targetIndex, False):
                fName = filename.replace("#target", self.dllInterface.render_get_render_target_name(targetIndex), 1)
                self.dllInterface.render_save_screenshot(fName, targetIndex, False)
            if self.dllInterface.render_is_render_target_enabled(targetIndex, True):
                fName = filename.replace("#target", self.dllInterface.render_get_render_target_name(targetIndex) + "(Variance)", 1)
                self.dllInterface.render_save_screenshot(fName, targetIndex, True)

    def render_for_iterations(self, iterationCount):
        accumIterateTime = ProcessTime(0,0)
        accumPreTime = ProcessTime(0,0)
        accumPostTime = ProcessTime(0,0)
        for i in range(iterationCount):
            [iterateTime, preTime, postTime] = self.dllInterface.render_iterate()
            accumIterateTime.microseconds += iterateTime.microseconds
            accumIterateTime.cycles += iterateTime.cycles
            accumPreTime.microseconds += preTime.microseconds
            accumPreTime.cycles += preTime.cycles
            accumPostTime.microseconds += postTime.microseconds
            accumPostTime.cycles += postTime.cycles
        self.take_screenshot(self.dllInterface.render_get_current_iteration(), accumIterateTime, accumPreTime, accumPostTime)

    def render_for_seconds(self, secondsToRender):
        startTime = process_time()
        curTime = startTime
        while curTime - startTime < secondsToRender:
            self.dllInterface.render_iterate()
            curTime = process_time()
        self.take_screenshot(self.dllInterface.render_get_current_iteration())

    def render_reset(self):
        self.dllInterface.render_reset()

    def renderer_set_parameter_bool(self, parameterName, value):
        return self.dllInterface.renderer_set_parameter_bool(parameterName, value)

    def renderer_set_parameter_float(self, parameterName, value):
        return self.dllInterface.renderer_set_parameter_float(parameterName, value)

    def renderer_set_parameter_int(self, parameterName, value):
        return self.dllInterface.renderer_set_parameter_int(parameterName, value)
