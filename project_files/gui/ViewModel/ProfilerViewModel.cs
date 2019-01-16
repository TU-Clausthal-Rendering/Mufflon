using System;
using System.ComponentModel;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Windows.Controls;
using System.Windows.Media;
using gui.Annotations;
using gui.Dll;
using gui.Model;
using gui.Model.Scene;

namespace gui.ViewModel
{
    public class ProfilerViewModel : INotifyPropertyChanged
    {
        private static int RENDERING_INDEX = 0;
        private static int LOADING_INDEX = 1;

        private Models m_models;
        private TreeView m_profilerTree;
        private TreeViewItem m_renderTree;
        private TreeViewItem m_loaderTree;

        public ProfilerViewModel(MainWindow window, Models models)
        {
            m_models = models;
            m_profilerTree = (TreeView)window.FindName("ProfilerTreeView");
            if (m_profilerTree == null)
                throw new System.Exception("Failed to aquire profiler tree");
            m_renderTree = (TreeViewItem)m_profilerTree.Items[RENDERING_INDEX];
            if (m_renderTree == null)
                throw new System.Exception("Failed to aquire render tree item");
            m_loaderTree = (TreeViewItem)m_profilerTree.Items[LOADING_INDEX];
            if (m_loaderTree == null)
                throw new System.Exception("Failed to aquire loader tree item");
            m_models.Renderer.PropertyChanged += RendererOnPropertyChanged;
            m_models.PropertyChanged += ModelsOnPropertyChanged;
        }

        private void ModelsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.World):
                    // add scene subscription
                    if (m_models.World != null)
                    {
                        m_models.World.PropertyChanged += SceneOnPropertyChanged;
                    }
                    break;
            }
        }

        private void RendererOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(RendererModel.Iteration):
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() => {
                        updateRenderingData();
                    }));
                    break;
                case nameof(RendererModel.IsRendering):
                    if (m_models.Renderer.IsRendering)
                        System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() => {
                            updateRenderingData();
                        }));
                    break;
            }
        }

        private void SceneOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(WorldModel.FullPath):
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() => {
                        updateLoadingData();
                    }));
                    break;
            }
        }

        // Initializes the rendering profile data
        private void updateRenderingData()
        {
            TreeViewItem renderItem = new TreeViewItem();
            renderItem.FontFamily = m_renderTree.FontFamily;
            renderItem.Header = m_renderTree.Header;

            // Parse the CSV string
            // Layout:
            // 'name',children:#,type:(cpu|gpu)(,(curr)snapshots:#)\n
            // csv-data\n
            // Children (see above) if applicable

            // Renderer
            {
                string csv = Core.profiling_get_total_and_snapshots();
                string[] lines = csv?.Split('\n');

                // Parse the top-level items
                uint currentLine = 0u;
                // Ignore last newline
                while ((currentLine + 1u) < lines.Length)
                {
                    currentLine = appendChild(lines, currentLine, ref renderItem);
                }
            }

            if (m_renderTree.IsExpanded)
            {
                renderItem.IsExpanded = true;
                expandPreviouslyExpanded(m_renderTree, ref renderItem);
            }
            m_profilerTree.Items[RENDERING_INDEX] = renderItem;
            m_renderTree = renderItem;
            m_profilerTree.UpdateLayout();
        }

        // Initializes the loading profiile data
        private void updateLoadingData()
        {
            TreeViewItem loadingItem = new TreeViewItem();
            loadingItem.FontFamily = m_loaderTree.FontFamily;
            loadingItem.Header = m_loaderTree.Header;

            // Parse the CSV string
            // Layout:
            // 'name',children:#,type:(cpu|gpu)(,(curr)snapshots:#)\n
            // csv-data\n
            // Children (see above) if applicable

            // Renderer
            {
                string csv = Loader.loader_profiling_get_total_and_snapshots();
                string[] lines = csv?.Split('\n');

                // Parse the top-level items
                uint currentLine = 0u;
                // Ignore last newline
                while ((currentLine + 1u) < lines.Length)
                {
                    currentLine = appendChild(lines, currentLine, ref loadingItem);
                }
            }

            if (m_loaderTree.IsExpanded)
            {
                loadingItem.IsExpanded = true;
                expandPreviouslyExpanded(m_loaderTree, ref loadingItem);
            }
            m_profilerTree.Items[LOADING_INDEX] = loadingItem;
            m_loaderTree = loadingItem;
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
            ulong totalThreadMicroSecond = UInt64.Parse(data[1u]);
            ulong totalProcessMicroSecond = UInt64.Parse(data[2u]);
            ulong totalWallMicroSecond = UInt64.Parse(data[3u]);
            ulong samples = UInt64.Parse(data[4u]);

            return String.Format("              |         Total  |       Average \n" +
                                 "Samples:      | {0,14} |\n" +
                                 "Cycles:       | {1,14} | {2,14}\n" +
                                 "Thread time:  | {3,14} | {4,14}\n" +
                                 "Process time: | {5,14} | {6,14}\n" +
                                 "Wall time:    | {7,14} | {8,14}\n",
                                 samples,
                                 formatCount(totalCycles), formatCount((ulong)(totalCycles / (float)samples)),
                                 formatTimePeriod(totalThreadMicroSecond), formatTimePeriod((ulong)(totalThreadMicroSecond / (float)samples)),
                                 formatTimePeriod(totalProcessMicroSecond), formatTimePeriod((ulong)(totalProcessMicroSecond / (float)samples)),
                                 formatTimePeriod(totalWallMicroSecond), formatTimePeriod((ulong)(totalWallMicroSecond / (float)samples)));
        }

        private static string formatGpuData(string[] data)
        {
            if (data.Length != 3u)
                return "Invalid data";

            ulong totalWallMicroSecond = UInt64.Parse(data[0u]) / 1000;
            ulong totalGpuMicroSecond = UInt64.Parse(data[1u]) / 1000;
            ulong samples = UInt64.Parse(data[2u]);

            return String.Format("              |         Total  |       Average \n" +
                                 "Samples:      | {0,14} |\n" +
                                 "Wall time:    | {1,14} | {2,14}\n" +
                                 "GPU time:     | {3,14} | {4,14}\n",
                                 samples,
                                 formatTimePeriod(totalWallMicroSecond), formatTimePeriod((ulong)(totalWallMicroSecond / (float)samples)),
                                 formatTimePeriod(totalGpuMicroSecond), formatTimePeriod((ulong)(totalGpuMicroSecond / (float)samples)));
        }

        private static string formatTimePeriod(ulong microseconds)
        {
            // Format the time so that it stays human-readable
            string formatted;
            if (microseconds < 1e4)
                formatted = String.Format(CultureInfo.InvariantCulture, "{0:#0.00}µs", microseconds);
            else if (microseconds < 1e7)
                formatted = String.Format(CultureInfo.InvariantCulture, "{0:#0.00}ms", microseconds / 1e3f);
            else if (microseconds < 6e7)
                formatted = String.Format(CultureInfo.InvariantCulture, "{0:#0.00}s", microseconds / 1e6f);
            else if (microseconds < 36e8)
                formatted = String.Format(CultureInfo.InvariantCulture, "{0:#0}m:{1:00.00}s",
                    microseconds / 6e7f, microseconds / 1e6f % 60f);
            else if (microseconds < 864e8)
                formatted = String.Format(CultureInfo.InvariantCulture, "{0:#0}h:{1:00}m:{2:00.00}s",
                    microseconds / 36e8f, microseconds / 6e7f % 60f, microseconds / 1e6f % 60f);
            else
                formatted = String.Format(CultureInfo.InvariantCulture, "{0:#0}d:{1:00}h:{2:00}m:{3:00.00}s",
                    microseconds / 864e8, microseconds / 36e8f % 24f,
                    microseconds / 6e7f % 60f, microseconds / 1e6f % 60f);
            return formatted;
        }

        private static string formatCount(ulong count)
        {
            // Format the count so that it stays human-readable
            string formatted;
            if (count < 1e4)
                formatted = String.Format(CultureInfo.InvariantCulture, "{0:#0.00}", count);
            else if (count < 1e7)
                formatted = String.Format(CultureInfo.InvariantCulture, "{0:#0.00}K", count / 1e3f);
            else if (count < 1e10)
                formatted = String.Format(CultureInfo.InvariantCulture, "{0:#0.00}M", count / 1e6f);
            else if (count < 1e13)
                formatted = String.Format(CultureInfo.InvariantCulture, "{0:#0.00}G", count / 1e9f);
            else
                formatted = String.Format(CultureInfo.InvariantCulture, "{0:#0.00}T", count / 1e12f);
            return formatted;
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
