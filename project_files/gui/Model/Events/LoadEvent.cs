using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace gui.Model.Events
{
    public class LoadEventArgs : EventArgs
    {
        public LoadEventArgs(LoadStatus status, string message = "")
        {
            Status = status;
            Message = message;
        }

        public enum LoadStatus
        {
            Started, // started loading => message can be ignored
            Loading, // still loading => message is the description of the current action
            Failed, // failed => message is the error message
            Finished // finished => message can be ignored
        }

        public string Message { get; }
        
        public LoadStatus Status { get; }
    }

    public delegate void LoadEventHandler(object sender, LoadEventArgs args);
}
