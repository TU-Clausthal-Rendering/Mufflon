using System;
using System.Windows;
using System.Windows.Media;

namespace gui.Dll
{
    class Logger
    {
        public delegate void LogEvent(string message, Brush color);
        public static event LogEvent Log;

        private static Core.Severity s_logLevel = Core.Severity.Info;
        public static Core.Severity LogLevel
        {
            get => s_logLevel;
            set
            {
                if (s_logLevel == value) return;
                s_logLevel = value;
                if (!Core.core_set_log_level(s_logLevel))
                    throw new Exception(Core.core_get_dll_error());
                if (!Loader.loader_set_log_level(s_logLevel))
                    throw new Exception(Loader.loader_get_dll_error());
            }
        }

        public static void log(string message, Core.Severity severity)
        {
            Brush color;
            if (severity >= LogLevel)
            {
                switch (severity)
                {
                    case Core.Severity.Pedantic: color = Brushes.Gray; break;
                    case Core.Severity.Warning: color = Brushes.Yellow; break;
                    case Core.Severity.Error: color = Brushes.Red; break;
                    case Core.Severity.FatalError: color = Brushes.Violet; break;
                    case Core.Severity.Info:
                    default:
                        color = Brushes.White; break;
                }
                if(Application.Current != null)
                    Application.Current.Dispatcher.BeginInvoke(Log, message, color);
            }
        }
    }
}
