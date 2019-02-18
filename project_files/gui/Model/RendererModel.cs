using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using gui.Annotations;
using gui.Dll;

namespace gui.Model
{
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
            m_value = Value;
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
                        break;
                    case Core.ParameterType.Int:
                        if (!Core.renderer_set_parameter_int(Name, (int) value))
                            throw new Exception(Core.core_get_dll_error());
                        break;
                    case Core.ParameterType.Float:
                        if (!Core.renderer_set_parameter_float(Name, (float) value))
                            throw new Exception(Core.core_get_dll_error());
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
        private bool m_isRendering = false;
        private UInt32 m_rendererIndex = UInt32.MaxValue;
        private UInt32 m_rendererCount = 0u;

        public RendererModel()
        {
            // Initial state: renderer paused
            RenderLock.WaitOne();
            PropertyChanged += OnRendererChanged;

            RendererIndex = 0u;
        }

        public Semaphore RenderLock = new Semaphore(1, 1);

        public bool IsRendering
        {
            get => m_isRendering;
            set
            {
                if(m_isRendering == value) return;
                m_isRendering = value;
                if (value)
                    RenderLock.Release();
                else
                    RenderLock.WaitOne();
                OnPropertyChanged(nameof(IsRendering));
            }
        }

        public uint Iteration => Core.render_get_current_iteration();

        public void Reset()
        {
            if (!Core.render_reset())
                throw new Exception(Core.core_get_dll_error());
           UpdateIterationCount();
        }

        public void UpdateDisplayTexture()
        {
            // TODO: this belongs somewhere else for sure
            if(!IsRendering)
            {
                RenderLock.Release();
                RenderLock.WaitOne();
            }
        }

        public void Iterate(uint times)
        {
            for(uint i = 0u; i < times; ++i)
            {
                IsRendering = true;
                IsRendering = false;
                UpdateIterationCount();
            }
        }

        public void UpdateIterationCount()
        {
            OnPropertyChanged(nameof(Iteration));
        }

        public UInt32 RendererCount
        {
            get => m_rendererCount;
            set
            {
                if (m_rendererCount == value) return;
                m_rendererCount = value;
                OnPropertyChanged(nameof(RendererCount));
            }
        }

        public UInt32 RendererIndex
        {
            get => m_rendererIndex;
            set
            {
                if (m_rendererIndex == value) return;
                // Quickly save the parameters before changing the renderer
                m_rendererIndex = value;
                if (!Core.render_enable_renderer(RendererIndex))
                    throw new Exception(Core.core_get_dll_error());
                Reset();
                OnPropertyChanged(nameof(RendererIndex));
            }
        }

        public bool UsesDevice(Core.RenderDevice dev)
        {
            return Core.render_renderer_uses_device(RendererIndex, dev);
        }

        private string m_name;
        public string Name
        {
            get => m_name;
        }

        public IReadOnlyList<RendererParameter> Parameters { get; private set; }

        private void OnRendererChanged(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(RendererIndex):
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
                    LoadRendererParameters();
                    OnPropertyChanged(nameof(Name));
                }   break;
            }
        }

        private void OnParameterChanged(object sender, PropertyChangedEventArgs args)
        {
            Reset();
            SaveRendererParameters();
        }


        // This method tries to find the settings index in the string collection
        private static int FindRendererDictionaryIndex(string name)
        {
            if (gui.Properties.Settings.Default.RendererParameters == null)
                gui.Properties.Settings.Default.RendererParameters = new System.Collections.Specialized.StringCollection();
            if (gui.Properties.Settings.Default.RendererParameters.Count % 2 != 0)
            {
                Logger.log("Invalid saved renderer properties; resetting all renderer properties to defaults", Core.Severity.Error);
                gui.Properties.Settings.Default.RendererParameters.Clear();
            }
            for (int i = 0; i < gui.Properties.Settings.Default.RendererParameters.Count; i += 2)
            {
                if (gui.Properties.Settings.Default.RendererParameters[i].Equals(name))
                    return i + 1;
            }
            return -1;
        }

        private string getSettingsKey()
        {
            string key = Name;
            foreach (Core.RenderDevice dev in Enum.GetValues(typeof(Core.RenderDevice)))
            {
                if (UsesDevice(dev))
                    key += "-" + Enum.GetName(typeof(Core.RenderDevice), dev);
            }
            return key;
        }

        // Saves the current renderer's parameters in textform in the application settings
        private void SaveRendererParameters()
        {
            if (Parameters == null)
                return;

            string val = "";

            foreach (var param in Parameters)
                val += param.Name + ";" + Enum.GetName(typeof(Core.ParameterType), param.Type) + ";" + param.Value.ToString() + "\n";

            string key = getSettingsKey();
            int idx = FindRendererDictionaryIndex(key);
            if (idx < 0)
            {
                gui.Properties.Settings.Default.RendererParameters.Add(key);
                gui.Properties.Settings.Default.RendererParameters.Add(val);
            }
            else
            {
                gui.Properties.Settings.Default.RendererParameters[idx] = val;
            }
        }

        // Restores the current renderer's parameters from the application settings
        private void LoadRendererParameters()
        {
            string key = getSettingsKey();
            int idx = FindRendererDictionaryIndex(key);
            if (idx < 0)
            {
                gui.Properties.Settings.Default.RendererParameters.Add(key);
                gui.Properties.Settings.Default.RendererParameters.Add("");
            }
            else
            {
                string parameters = gui.Properties.Settings.Default.RendererParameters[idx];

                var splitParams = parameters.Split('\n');
                for (int i = 0; i < splitParams.Length; ++i)
                {
                    var paramVals = splitParams[i].Split(';');
                    if (paramVals.Length == 3 && Parameters[i].Name == paramVals[0])
                    {
                        if (paramVals[1] == Enum.GetName(typeof(Core.ParameterType), Core.ParameterType.Bool) && Parameters[i].Type == Core.ParameterType.Bool)
                        {
                            bool val;
                            if (Boolean.TryParse(paramVals[2], out val))
                                Parameters[i].Value = val;
                        }
                        else if (paramVals[1] == Enum.GetName(typeof(Core.ParameterType), Core.ParameterType.Int) && Parameters[i].Type == Core.ParameterType.Int)
                        {
                            int val;
                            if (Int32.TryParse(paramVals[2], out val))
                                Parameters[i].Value = val;
                        }
                        else if (paramVals[1] == Enum.GetName(typeof(Core.ParameterType), Core.ParameterType.Float) && Parameters[i].Type == Core.ParameterType.Float)
                        {
                            double val;
                            if (Double.TryParse(paramVals[2], out val))
                                Parameters[i].Value = (float)val;
                        }
                    }
                }
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
    }
}
