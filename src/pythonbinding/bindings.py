from ctypes import *
from time import *
from enum import IntEnum
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

class Device(IntEnum):
    CPU = 1,
    CUDA = 2,
    OPENGL = 4

class LogLevel(IntEnum):
    PEDANTIC = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL_ERROR = 4
    
class LoaderStatus(IntEnum):
    SUCCESS = 0,
    ERROR = 1,
    ABORT = 2

class DllInterface:

    def __init__(self):
        self.dllHolder = DllHolder()
        self.dllHolder.core.mufflon_initialize()
        self.disable_profiling()
        self.dllHolder.core.render_get_renderer_name.restype = c_char_p
        self.dllHolder.core.render_get_renderer_short_name.restype = c_char_p
        self.dllHolder.core.render_get_render_target_name.restype = c_char_p
        self.dllHolder.core.renderer_set_parameter_enum.argtypes = [c_char_p, c_int32]
        self.dllHolder.core.renderer_get_parameter_enum_value_from_name.argtypes = [c_char_p, c_char_p, c_void_p]
        self.dllHolder.core.render_enable_render_target.argtypes = [c_char_p, c_bool]
        self.dllHolder.core.render_enable_render_target.argtypes = [c_char_p, c_bool]
        self.dllHolder.core.render_iterate.argtypes = [ POINTER(ProcessTime) ]
        self.dllHolder.core.scenario_get_name.restype = c_char_p
        self.dllHolder.core.scenario_get_name.argtypes = [c_void_p]
        self.dllHolder.core.world_set_frame_current.argtypes = [c_uint]
        self.dllHolder.core.world_get_current_scenario.restype = c_void_p
        self.dllHolder.core.world_find_scenario.restype = c_void_p
        self.dllHolder.core.world_load_scenario.restype = c_void_p
        self.dllHolder.core.render_save_denoised_radiance.argtypes = [c_char_p]
        
    def __del__(self):
        self.dllHolder.core.mufflon_destroy()
        
    def core_set_log_level(self, logLevel):
        return self.dllHolder.core.core_set_log_level(c_int32(logLevel)) != 0
        
    def loader_set_log_level(self, logLevel):
        return self.dllHolder.mffLoader.loader_set_log_level(c_int32(logLevel)) != 0

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
    
    def renderer_set_parameter_enum(self, parameterName, value):
        return self.dllHolder.core.renderer_set_parameter_enum(c_char_p(parameterName.encode('utf-8')), c_int32(value))

    def renderer_get_parameter_enum_value(self, parameterName, valueName):
        value = 0
        if not self.dllHolder.core.renderer_get_parameter_enum_value_from_name(c_char_p(parameterName.encode('utf-8')), c_char_p(valueName.encode('utf-8')), byref(value)):
            raise Exception("Failed to retrieve enum parameter '" + parameterName + "' value '" + valueName + "'")
        return value

    def renderer_get_parameter_enum_count(self, parameterName):
        count = 0
        if not self.dllHolder.core.renderer_get_parameter_enum_count(c_char_p(parameterName.encode('utf-8')), byref(count)):
            raise Exception("Failed to retrieve enum parameter '" + parameterName + "' count")
        return count

    def render_iterate(self):
        iterateTime = ProcessTime(0,0)
        preTime = ProcessTime(0,0)
        postTime = ProcessTime(0,0)
        self.dllHolder.core.render_iterate(byref(iterateTime))
        return iterateTime, preTime, postTime

    def render_reset(self):
        return self.dllHolder.core.render_reset()

    def render_get_current_iteration(self):
        return self.dllHolder.core.render_get_current_iteration()

    def render_save_screenshot(self, fileName, targetIndex, variance):
        return self.dllHolder.core.render_save_screenshot(c_char_p(fileName.encode('utf-8')), c_uint32(targetIndex), c_uint32(variance))
    
    def render_save_denoised_radiance(self, fileName):
        return self.dllHolder.core.render_save_denoised_radiance(c_char_p(fileName.encode('utf-8')))
        
    def render_get_render_target_count(self):
        return self.dllHolder.core.render_get_render_target_count()

    def render_get_render_target_name(self, index):
        return self.dllHolder.core.render_get_render_target_name(c_uint32(index)).decode()

    def render_is_render_target_enabled(self, targetIndex, variance):
        return self.dllHolder.core.render_is_render_target_enabled(c_uint32(targetIndex), c_bool(variance))

    def render_enable_render_target(self, targetName, variance):
        return self.dllHolder.core.render_enable_render_target(c_char_p(targetName.encode('utf-8')), c_bool(variance))
    
    def render_disable_render_target(self, targetName, variance):
        return self.dllHolder.core.render_disable_render_target(c_char_p(targetName.encode('utf-8')), c_bool(variance))

    def render_enable_renderer(self, rendererIndex, variation):
        return self.dllHolder.core.render_enable_renderer(c_uint32(rendererIndex), c_uint32(variation))

    def render_get_renderer_count(self):
        return self.dllHolder.core.render_get_renderer_count()

    def render_get_renderer_variations(self, index):
        return self.dllHolder.core.render_get_renderer_variations(c_uint32(index))

    def render_get_renderer_name(self, rendererIndex):
        return self.dllHolder.core.render_get_renderer_name(c_uint32(rendererIndex)).decode()

    def render_get_renderer_short_name(self, rendererIndex):
        return self.dllHolder.core.render_get_renderer_short_name(c_uint32(rendererIndex)).decode()

    def render_get_renderer_devices(self, rendererIndex, variation):
        return self.dllHolder.core.render_get_renderer_devices(c_uint32(rendererIndex), c_uint32(variation))

    def render_get_active_scenario_name(self):
        return self.dllHolder.core.scenario_get_name( self.dllHolder.core.world_get_current_scenario()).decode()

    def world_set_frame_current(self, frame):
        return self.dllHolder.core.world_set_frame_current(c_uint(frame))
    
    def world_get_frame_current(self, frame):
        return self.dllHolder.core.world_get_frame_current(byref(frame))
    
    def world_get_frame_start(self, frame):
        return self.dllHolder.core.world_get_frame_start(byref(frame))
    
    def world_get_frame_end(self, frame):
        return self.dllHolder.core.world_get_frame_end(byref(frame))

    def world_find_scenario(self, name):
        return self.dllHolder.core.world_find_scenario(c_char_p(name.encode('utf-8')))

    def world_load_scenario(self, hdl):
        return self.dllHolder.core.world_load_scenario(c_void_p(hdl))

    def world_get_current_scenario(self):
        return self.dllHolder.core.world_get_current_scenario()
    
    def world_set_tessellation_level(self, level):
        self.dllHolder.core.world_set_tessellation_level(level)

    def world_get_tessellation_level(self):
        return self.dllHolder.core.world_get_tessellation_level()

    def scene_request_retessellation(self):
        return self.dllHolder.scene_request_retessellation()


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
        if returnValue != LoaderStatus.SUCCESS:
            raise Exception("Failed to load scene '" + sceneJson + "' (error code: " + returnValue.name + ")")
        self.enable_render_target("Radiance", False)

    def enable_renderer(self, rendererName, devices):
        for i in range(self.dllInterface.render_get_renderer_count()):
            name = self.dllInterface.render_get_renderer_name(i)
            shortName = self.dllInterface.render_get_renderer_short_name(i)
            if (name.lower() == rendererName.lower() or shortName.lower() == rendererName.lower()):
                for v in range(self.dllInterface.render_get_renderer_variations(i)):
                    if devices == self.dllInterface.render_get_renderer_devices(i, v):
                        self.activeRendererName = rendererName
                        self.dllInterface.render_enable_renderer(i, v)
                        return
        deviceStr = "[ "
        for dev in Device:
            if (devices & dev) != 0:
                deviceStr += dev.name + " "
        deviceStr += "]"
        raise Exception("Could not find renderer '" + rendererName + "' for the devices " + deviceStr)
        
    def load_scenario(self, scenarioName):
        hdl = self.dllInterface.world_find_scenario(scenarioName)
        if not hdl:
            raise Exception("Failed to find scenario '" + scenarioName + "'")
        if not self.dllInterface.world_load_scenario(hdl):
            raise Exception("Failed to load scenario '" + scenarioName + "'")

    def set_current_animation_frame(self, frame):
        if not self.dllInterface.world_set_frame_current(frame):
            raise Exception("Failed to set animation frame to '" + str(frame) + "'")

    def get_current_animation_frame(self):
        frame = c_uint(0)
        if not self.dllInterface.world_get_frame_current(frame):
            raise Exception("Failed to get current animation frame")
        return frame.value

    def get_start_animation_frame(self):
        frame = c_uint(0)
        if not self.dllInterface.world_get_frame_start(frame):
            raise Exception("Failed to get start animation frame")
        return frame.value

    def get_end_animation_frame(self):
        frame = c_uint(0)
        if not self.dllInterface.world_get_frame_end(frame):
            raise Exception("Failed to get end animation frame")
        return frame.value
        
    def set_renderer_log_level(self, logLevel):
        if not self.dllInterface.core_set_log_level(logLevel):
            raise Exception("Failed to set log level to '" + logLevel.name + "'")
            
    def set_loader_log_level(self, logLevel):
        if not self.dllInterface.loader_set_log_level(logLevel):
            raise Exception("Failed to set log level to '" + logLevel.name + "'")

    def enable_render_target(self, targetName, variance):
        if not self.dllInterface.render_enable_render_target(targetName, variance):
            raise Exception("Failed to enable render target " + targetName + " (variance: " + str(variance) + ")")
        
    def disable_render_target(self, targetName, variance):
      if not self.dllInterface.render_disable_render_target(targetName, variance):
            raise Exception("Failed to disable render target " + targetName + " (variance: " + str(variance) + ")")
            
    def take_denoised_screenshot(self, iterationNr, iterateTime=ProcessTime(0,0), preTime=ProcessTime(0,0), postTime=ProcessTime(0,0)):
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
        
        fName = filename.replace("#target", "denoised", 1)
        self.dllInterface.render_save_denoised_radiance(fName)

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

    def render_for_iterations(self, iterationCount, printProgress=False, progressSteps=1, takeScreenshot=True, denoise=False):
        accumIterateTime = ProcessTime(0,0)
        accumPreTime = ProcessTime(0,0)
        accumPostTime = ProcessTime(0,0)
        for i in range(iterationCount):
            if printProgress and (i % progressSteps == 0):
                print("--- ", (i + 1), " of ", iterationCount, " ---", flush=True)
            [iterateTime, preTime, postTime] = self.dllInterface.render_iterate()
            accumIterateTime.microseconds += iterateTime.microseconds
            accumIterateTime.cycles += iterateTime.cycles
            accumPreTime.microseconds += preTime.microseconds
            accumPreTime.cycles += preTime.cycles
            accumPostTime.microseconds += postTime.microseconds
            accumPostTime.cycles += postTime.cycles
        if takeScreenshot:
            if denoise:
                self.take_denoised_screenshot(self.dllInterface.render_get_current_iteration(), accumIterateTime, accumPreTime, accumPostTime)
            else:
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

    def renderer_set_parameter_enum(self, parameterName, valueName):
        return self.dllInterface(renderer_set_parameter_enum(parameterName, self.dllInterface.renderer_get_parameter_enum_value(parameterName, valueName)))
