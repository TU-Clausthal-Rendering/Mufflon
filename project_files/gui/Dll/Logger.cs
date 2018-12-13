using System.Windows;
using System.Windows.Media;

namespace gui.Dll
{
    class Logger
    {
        public delegate void LogEvent(string message, Brush color);
        public static event LogEvent Log;

        public static void log(string message, Core.Severity severity)
        {
            Brush color;
            switch (severity)
            {
                case Core.Severity.PEDANTIC: color = Brushes.Gray; break;
                case Core.Severity.WARNING: color = Brushes.Yellow; break;
                case Core.Severity.ERROR: color = Brushes.Red; break;
                case Core.Severity.FATAL_ERROR: color = Brushes.Violet; break;
                case Core.Severity.INFO:
                default:
                    color = Brushes.White; break;
            }
            Application.Current.Dispatcher.BeginInvoke(Log, message, color);
        }
    }
}
