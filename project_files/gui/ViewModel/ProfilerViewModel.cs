using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Controls;
using System.Windows.Media;
using gui.Annotations;
using gui.Dll;
using gui.Model;

namespace gui.ViewModel
{
    public class ProfilerViewModel : INotifyPropertyChanged
    {
        private Models m_models;
        private TreeView m_profilerTree;

        public ProfilerViewModel(MainWindow window, Models models)
        {
            m_models = models;
            m_profilerTree = (TreeView)window.FindName("ProfilerTreeView");
            if (m_profilerTree == null)
                throw new System.Exception("Failed to aquire profiler tree");
            TreeViewItem top = new TreeViewItem();
            top.FontFamily = new FontFamily("Courier New");
            top.Header = "Profiler Data";
            m_profilerTree.Items.Add(top);
            m_models.Renderer.PropertyChanged += RendererOnPropertyChanged;
        }

        private void RendererOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(RendererModel.Iteration):
                    System.Windows.Application.Current.Dispatcher.Invoke(new Action( () => {
                        initProfilerData();
                    } ));
                    break;
                case nameof(RendererModel.IsRendering):
                    if(m_models.Renderer.IsRendering)
                        System.Windows.Application.Current.Dispatcher.Invoke(new Action(() => {
                            initProfilerData();
                        }));
                    break;
            }
        }

        // Initializes the 
        private void initProfilerData()
        {
            string csv = Core.profiling_get_current_and_snapshots();

            TreeViewItem top = new TreeViewItem();
            top.FontFamily = new FontFamily("Courier New");
            top.Header = "Profiler Data";

            // Parse the CSV string
            // Layout:
            // 'name',children:#,type:(cpu|gpu)(,(curr)snapshots:#)\n
            // csv-data\n
            // Children (see above) if applicable

            string[] lines = csv?.Split('\n');

            // Parse the top-level items
            uint currentLine = 0u;
            // Ignore last newline
            while ((currentLine + 1u) < lines.Length)
            {
                currentLine = appendChild(lines, currentLine, ref top);
            }

            // Expand the tree if necessary
            if(m_profilerTree.Items.Count >= 1)
            {
                TreeViewItem oldTop = ((TreeViewItem)m_profilerTree.Items[0]);
                if (oldTop.IsExpanded)
                {
                    top.IsExpanded = true;
                    expandPreviouslyExpanded(oldTop, ref top);
                }
            }
            m_profilerTree.Items.Clear();
            m_profilerTree.Items.Add(top);
            m_profilerTree.UpdateLayout();
        }

        // Appends a profiler child to the current profiling tree item
        private static uint appendChild(string[] lines, uint currentLine, ref TreeViewItem currItem)
        {
            string[] header = lines[currentLine].Split(',');
            currentLine += 1u;
            TreeViewItem child = new TreeViewItem();
            child.FontFamily = new FontFamily("Courier New");
            if (header.Length < 3u)
            {
                child.Header = "## INVALID PROFILER DATA ##";
            }
            else
            {
                child.Header = header[0u];
                // Parse the header data to identify the profiler
                if (header[2u].Equals("type:cpu"))
                    child.Header += " (CPU)";
                else if (header[2u].Equals("type:gpu"))
                    child.Header += " (GPU)";
                else
                    child.Header += " (unknown)";

                // Check if we have snapshots or not
                TreeViewItem profilerData = new TreeViewItem();
                profilerData.FontFamily = new FontFamily("Courier New");
                child.Items.Add(profilerData);
                if (header.Length >= 4u)
                {
                    string[] snapshotHeader = header[3u].Split(':');
                    if (snapshotHeader.Length != 2u)
                    {
                        profilerData.Header = "Invalid data";
                    }
                    else
                    {
                        // Check if we got current data as well
                        if(snapshotHeader[0u].Equals("currsnapshots"))
                        {
                            string[] data = lines[currentLine].Split(',');
                            if (header[2u].Equals("type:cpu"))
                                profilerData.Header = formatCpuData(data);
                            else if (header[2u].Equals("type:gpu"))
                                profilerData.Header = formatGpuData(data);
                            currentLine += 1u;
                        }

                        uint snapshots = UInt32.Parse(snapshotHeader[1u]);
                        TreeViewItem snaps = new TreeViewItem();
                        snaps.Header = "Snapshots";
                        child.Items.Add(snaps);

                        for (uint s = 0u; s < snapshots; ++s)
                        {
                            TreeViewItem snap = new TreeViewItem();
                            string[] snapData = lines[currentLine].Split(',');
                            if (header[2u].Equals("type:cpu"))
                                snap.Header = formatCpuData(snapData);
                            else if (header[2u].Equals("type:gpu"))
                                snap.Header = formatGpuData(snapData);
                            currentLine += 1u;
                            snaps.Items.Add(snap);
                        }
                    }
                }
                else
                {
                    string[] data = lines[currentLine].Split(',');
                    if (header[2u].Equals("type:cpu"))
                        profilerData.Header = formatCpuData(data);
                    else if (header[2u].Equals("type:gpu"))
                        profilerData.Header = formatGpuData(data);
                    currentLine += 1u;
                }

                string[] childrenHeader = header[1u].Split(':');
                if (childrenHeader.Length == 2u)
                {
                    uint children = 0u;
                    if (UInt32.TryParse(childrenHeader[1u], out children) && children > 0u)
                    {
                        // Recursively add other children as well
                        for (uint c = 0u; c < children; ++c)
                        {
                            currentLine = appendChild(lines, currentLine, ref child);
                        }
                    }
                }
            }
            currItem.Items.Add(child);
            return currentLine;
        }

        // Traverses the treeview and expands the views that were previously expanded as well
        private static void expandPreviouslyExpanded(TreeViewItem oldItem, ref TreeViewItem currItem)
        {
            // Iterate both trees and expand expanded items
            for (int c = 0; c < currItem.Items.Count; ++c)
            {
                TreeViewItem currChild = ((TreeViewItem)currItem.Items[c]);
                // See if the old item contains the same entry
                for (int i = 0; i < oldItem.Items.Count; ++i)
                {
                    TreeViewItem oldChild = ((TreeViewItem)oldItem.Items[i]);
                    if (oldChild.Header.Equals(currChild.Header))
                    {
                        // Expand the item if it was previously expanded and check sub-children as well
                        if(oldChild.IsExpanded)
                        {
                            currChild.IsExpanded = true;
                            expandPreviouslyExpanded(oldChild, ref currChild);
                        }
                    }
                }
            }
        }

        private static string formatCpuData(string[] data)
        {
            if (data.Length != 5u)
                return "Invalid data";

            ulong totalCycles = UInt64.Parse(data[0u]);
            ulong totalThreadMilliSecond = UInt64.Parse(data[1u]) / 1000;
            ulong totalProcessMilliSecond = UInt64.Parse(data[2u]) / 1000;
            ulong totalWallMilliSecond = UInt64.Parse(data[3u]) / 1000;
            ulong samples = UInt64.Parse(data[4u]);

            return String.Format("              |         Total  |       Average \n" +
                                 "Samples:      | {0,14} |\n" +
                                 "Cycles:       | {1,14} | {2,14}\n" +
                                 "Thread time:  | {3,12}ms | {4,12}ms\n" +
                                 "Process time: | {5,12}ms | {6,12}ms\n" +
                                 "Wall time:    | {7,12}ms | {8,12}ms\n",
                                 samples,
                                 totalCycles, (ulong)(totalCycles / (float)samples),
                                 totalThreadMilliSecond, (ulong)(totalThreadMilliSecond / (float)samples),
                                 totalProcessMilliSecond, (ulong)(totalProcessMilliSecond / (float)samples),
                                 totalWallMilliSecond, (ulong)(totalWallMilliSecond / (float)samples));
        }

        private static string formatGpuData(string[] data)
        {
            if (data.Length != 3u)
                return "Invalid data";

            ulong totalWallMilliSecond = UInt64.Parse(data[0u]) / 1000;
            ulong totalGpuMilliSecond = UInt64.Parse(data[1u]) / 1000;
            ulong samples = UInt64.Parse(data[2u]);

            return String.Format("              |         Total  |       Average \n" +
                                 "Samples:      | {0,14} |\n" +
                                 "Wall time:    | {1,13}ms | {2,13}ms\n" +
                                 "GPU time:     | {3,13}ms | {4,13}ms\n",
                                 samples,
                                 totalWallMilliSecond, (ulong)(totalWallMilliSecond / (float)samples),
                                 totalGpuMilliSecond, (ulong)(totalGpuMilliSecond / (float)samples));
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
