using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using gui.Annotations;
using gui.Dll;

namespace gui.Model
{
    [Serializable]
    public struct SerializedRendererParameter
    {
        public Core.ParameterType Type { get; set; }
        public string Value { get; set; }
    }

    public class RendererParameter : INotifyPropertyChanged
    {
        private object m_value;

        public RendererParameter(uint index)
        {
            Core.ParameterType type;
            Name = Core.renderer_get_parameter_desc(index, out type);
            if (Name.Contains(";"))
                Logger.log("Renderer parameter '" + Name + "' contains prohibited symbol ';'; this may severely impact proper " +
                    "loading and storing of last renderer parameters", Core.Severity.Warning);
            if (Name.Contains("\n"))
                Logger.log("Renderer parameter '" + Name + "' contains prohibited symbol '\n'; this may severely impact proper " +
                    "loading and storing of last renderer parameters", Core.Severity.Warning);
            Type = type;
            Index = index;
            if (type == Core.ParameterType.Enum)
                m_value = "";
            else
                m_value = 0;
        }

        public Core.ParameterType Type { get; private set; }
        public string Name { get; private set; }
        public uint Index { get; private set; }
        public object Value
        {
            get
            {
                switch (Type)
                {
                    case Core.ParameterType.Bool:
                    {
                        uint val;
                        if (!Core.renderer_get_parameter_bool(Name, out val))
                            throw new Exception(Core.core_get_dll_error());
                        m_value = val;
                        return val != 0;
                    }
                    case Core.ParameterType.Int:
                    {
                        int val;
                        if (!Core.renderer_get_parameter_int(Name, out val))
                            throw new Exception(Core.core_get_dll_error());
                        m_value = val;
                        return val;
                    }
                    case Core.ParameterType.Float:
                    {
                        float val;
                        if (!Core.renderer_get_parameter_float(Name, out val))
                            throw new Exception(Core.core_get_dll_error());
                        m_value = val;
                        return val;
                        }
                    case Core.ParameterType.Enum:
                    {
                        int val;
                        if (!Core.renderer_get_parameter_enum(Name, out val))
                            throw new Exception(Core.core_get_dll_error());
                        m_value = val;
                        string name;
                        if (!Core.renderer_get_parameter_enum_name(Name, (int)m_value, out name))
                            throw new Exception(Core.core_get_dll_error());
                        return name;
                    }
                }
                return null;
            }

            set
            {
                if (m_value.Equals(value)) return;
                switch (Type)
                {
                    case Core.ParameterType.Bool:
                        if (!Core.renderer_set_parameter_bool(Name, ((bool) value) ? 1u : 0u))
                            throw new Exception(Core.core_get_dll_error());
                        m_value = value;
                        break;
                    case Core.ParameterType.Int:
                        if (!Core.renderer_set_parameter_int(Name, (int) value))
                            throw new Exception(Core.core_get_dll_error());
                        m_value = value;
                        break;
                    case Core.ParameterType.Float:
                        if (!Core.renderer_set_parameter_float(Name, (float) value))
                            throw new Exception(Core.core_get_dll_error());
                        m_value = value;
                        break;
                    case Core.ParameterType.Enum:
                        int val;
                        if(!Core.renderer_get_parameter_enum_value_from_name(Name, (string)value, out val))
                            throw new Exception(Core.core_get_dll_error());
                        if (!Core.renderer_set_parameter_enum(Name, val))
                            throw new Exception(Core.core_get_dll_error());
                        m_value = val;
                        break;
                }
                OnPropertyChanged(nameof(Value));
            }
        }

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    };

    /// <summary>
    /// information about the active renderer
    /// </summary>
    public class RendererModel : INotifyPropertyChanged
    {
        public delegate void ScreenshotHandler(bool denoised);
        public delegate void ParameterSaveHandler(RendererParameter param);
        public delegate void FrameCompletionHandler();
        public delegate void AnimationCompletionHandler();

        public event EventHandler RequestWorldClear;
        public event EventHandler RequestRedraw;
        public event ParameterSaveHandler RequestParameterSave;
        public event ScreenshotHandler RequestScreenshot;

        private volatile bool m_isRendering = false;

        public RendererModel()
        {
            // Initial state: renderer paused (this will always be in the main UI thread)
            PropertyChanged += OnRendererChanged;
        }

        // There are really only two places this should be touched: in IsRendering and the primary render loop
        // Its purpose is to send the render thread to sleep when we don't want to render, NOT do enforce
        // no-rendering when we change attributes (camera, lights, render targets etc.); this is
        // done in the core-dll itself via a mutex (actually two, but lets not split hairs).
        public ManualResetEvent RenderLock = new ManualResetEvent(false);
        private volatile int m_remainingIterations = -1;
        public int RemainingIterations { get => m_remainingIterations; }

        public bool IsRendering
        {
            get => m_isRendering;
            set
            {
                if(m_isRendering == value) return;
                m_isRendering = value;
                if(value)
                {
                    RenderLock.Set();
                }
                else
                {
                    RenderAnimation = false;
                    m_remainingIterations = -1;
                }
                OnPropertyChanged(nameof(IsRendering));
            }
        }

        // Controls whether the rendering sequence will switch to the next frame
        // upon completing the mandated number of iterations
        public bool RenderAnimation { get; set; } = false;
        // Additionally, these control min./max. of the rendering and a callback that gets invoked upon frame completion
        public int AnimationStart { get; set; } = -1;
        public int AnimationEnd { get; set; } = -1;
        public FrameCompletionHandler AnimationFrameComplete { get; set; } = null;
        public AnimationCompletionHandler AnimationComplete { get; set; } = null;

        public Core.RenderDevice RenderDevices { get => Core.render_get_renderer_devices(RendererIndex, RendererVariation); }

        public uint Iteration => Core.render_get_current_iteration();

        public Core.ProcessTime CurrentIterationTime { get; private set; }
        public Core.ProcessTime AverageIterationTime => new Core.ProcessTime()
        {
            cycles = TotalIterationTime.cycles / (Iteration == 0 ? 1 : Iteration),
            microseconds = TotalIterationTime.microseconds / (Iteration == 0 ? 1 : Iteration)
        };
        public Core.ProcessTime TotalIterationTime { get; private set; }

        public void Reset()
        {
            if (!Core.render_reset())
                throw new Exception(Core.core_get_dll_error());
            CurrentIterationTime = new Core.ProcessTime();
            OnPropertyChanged(nameof(CurrentIterationTime));
            UpdateIterationData();
        }

        // Takes screenshots of all active render targets
        public void TakeScreenshot(bool denoised)
        {
            RequestScreenshot(denoised);
        }

        // Updates the rendering bitmap without actually rendering anything; should be used
        // e.g. when selecting a different render target to display
        public void UpdateRenderBitmap()
        {
            RequestRedraw(this, null);
        }

        // Clears the world from the render thread
        public void ClearWorld()
        {
            RequestWorldClear(this, null);
        }

        // This iterates by leveraging the GUI and includes texture updates etc
        public void Iterate(uint times)
        {
            m_remainingIterations = (int)times;
            IsRendering = true;
        }

        // This iterates ONLY the renderer (and updates some data like times and iteration count), so this
        // really should only be called from exactly ONE place: the main render loop
        public void Iterate()
        {
            Core.ProcessTime time;
            Core.ProcessTime preTime;
            Core.ProcessTime postTime;
            if (!Core.render_iterate(out time, out preTime, out postTime))
                throw new Exception(Core.core_get_dll_error());

            CurrentIterationTime = time;
            OnPropertyChanged(nameof(CurrentIterationTime));

            // We also let the GUI know that an iteration has taken place
            Application.Current.Dispatcher.BeginInvoke(new Action(() => UpdateIterationData()));
        }

        private void UpdateIterationData()
        {
            if(Iteration <= 1)
                TotalIterationTime = new Core.ProcessTime();
            TotalIterationTime += CurrentIterationTime;
            OnPropertyChanged(nameof(AverageIterationTime));
            OnPropertyChanged(nameof(TotalIterationTime));
            OnPropertyChanged(nameof(Iteration));
        }

        public UInt32 RendererCount { get => Core.render_get_renderer_count(); }
        public UInt32 RendererVariationsCount { get => Core.render_get_renderer_variations(RendererIndex); }

        public UInt32 RendererIndex { get; private set; } = UInt32.MaxValue;
        public uint RendererVariation { get; private set; } = 0;

        public void SetRenderer(uint index, uint variation)
        {
            bool indexChanges = index != RendererIndex;
            bool variationChanges = variation != RendererVariation;

            if(indexChanges || variationChanges)
            {
                RendererIndex = index;
                RendererVariation = variation;
                if (!Core.render_enable_renderer(RendererIndex, RendererVariation))
                    throw new Exception(Core.core_get_dll_error());

                if(indexChanges)
                    OnPropertyChanged(nameof(RendererIndex));
                if(variationChanges)
                    OnPropertyChanged(nameof(RendererVariation));
            }
        }

        public bool UsesDevice(Core.RenderDevice dev)
        {
            return (Core.render_get_renderer_devices(RendererIndex, RendererVariation) & dev) != 0;
        }

        private string m_name;
        public string Name
        {
            get => m_name;
        }

        private string m_shortName;
        public string ShortName
        {
            get => m_shortName;
        }

        public IReadOnlyList<RendererParameter> Parameters { get; private set; }

        private void OnRendererChanged(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(RendererIndex):
                case nameof(RendererVariation):
                {
                    // Query renderer parameters for new renderer
                    List<RendererParameter> paramList = new List<RendererParameter>();
                    uint paramCount = Core.renderer_get_num_parameters();
                    for (uint i = 0u; i < paramCount; ++i)
                    {
                        var param = new RendererParameter(i);
                        param.PropertyChanged += OnParameterChanged;
                        paramList.Add(param);
                    }

                    Parameters = paramList;
                    m_name = Core.render_get_renderer_name(RendererIndex);
                    m_shortName = Core.render_get_renderer_short_name(RendererIndex);
                    OnPropertyChanged(nameof(Name));
                    OnPropertyChanged(nameof(ShortName));
                    OnPropertyChanged(nameof(RenderDevices));
                }   break;
            }
        }

        private void OnParameterChanged(object sender, PropertyChangedEventArgs args)
        {
            Reset();
            RequestParameterSave(sender as RendererParameter);
        }

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
}
