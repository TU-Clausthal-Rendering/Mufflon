﻿//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//     Runtime Version:4.0.30319.42000
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace gui.Properties {
    
    
    [global::System.Runtime.CompilerServices.CompilerGeneratedAttribute()]
    [global::System.CodeDom.Compiler.GeneratedCodeAttribute("Microsoft.VisualStudio.Editors.SettingsDesigner.SettingsSingleFileGenerator", "15.9.0.0")]
    internal sealed partial class Settings : global::System.Configuration.ApplicationSettingsBase {
        
        private static Settings defaultInstance = ((Settings)(global::System.Configuration.ApplicationSettingsBase.Synchronized(new Settings())));
        
        public static Settings Default {
            get {
                return defaultInstance;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("")]
        public string LastWorldPath {
            get {
                return ((string)(this["LastWorldPath"]));
            }
            set {
                this["LastWorldPath"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("0")]
        public uint LastSelectedRenderer {
            get {
                return ((uint)(this["LastSelectedRenderer"]));
            }
            set {
                this["LastSelectedRenderer"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        public global::System.Collections.Specialized.StringCollection LastWorlds {
            get {
                return ((global::System.Collections.Specialized.StringCollection)(this["LastWorlds"]));
            }
            set {
                this["LastWorlds"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("True")]
        public bool AutoStartOnLoad {
            get {
                return ((bool)(this["AutoStartOnLoad"]));
            }
            set {
                this["AutoStartOnLoad"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("0")]
        public int LogLevel {
            get {
                return ((int)(this["LogLevel"]));
            }
            set {
                this["LogLevel"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("0")]
        public int CoreProfileLevel {
            get {
                return ((int)(this["CoreProfileLevel"]));
            }
            set {
                this["CoreProfileLevel"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("0")]
        public int LoaderProfileLevel {
            get {
                return ((int)(this["LoaderProfileLevel"]));
            }
            set {
                this["LoaderProfileLevel"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("#scene-#scenario-#renderer-#iteration-#target")]
        public string ScreenshotNamePattern {
            get {
                return ((string)(this["ScreenshotNamePattern"]));
            }
            set {
                this["ScreenshotNamePattern"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("")]
        public string ScreenshotFolder {
            get {
                return ((string)(this["ScreenshotFolder"]));
            }
            set {
                this["ScreenshotFolder"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        public global::System.Collections.Specialized.StringCollection ScreenshotNamePatternHistory {
            get {
                return ((global::System.Collections.Specialized.StringCollection)(this["ScreenshotNamePatternHistory"]));
            }
            set {
                this["ScreenshotNamePatternHistory"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("F2")]
        public string ScreenshotGesture {
            get {
                return ((string)(this["ScreenshotGesture"]));
            }
            set {
                this["ScreenshotGesture"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("ALT+P")]
        public string PlayPauseGesture {
            get {
                return ((string)(this["PlayPauseGesture"]));
            }
            set {
                this["PlayPauseGesture"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("ALT+R")]
        public string ResetGesture {
            get {
                return ((string)(this["ResetGesture"]));
            }
            set {
                this["ResetGesture"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("ALT+M")]
        public string ToggleCameraMovementGesture {
            get {
                return ((string)(this["ToggleCameraMovementGesture"]));
            }
            set {
                this["ToggleCameraMovementGesture"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("False")]
        public bool AllowCameraMovement {
            get {
                return ((bool)(this["AllowCameraMovement"]));
            }
            set {
                this["AllowCameraMovement"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("0")]
        public int LastSelectedRenderTarget {
            get {
                return ((int)(this["LastSelectedRenderTarget"]));
            }
            set {
                this["LastSelectedRenderTarget"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        public global::System.Collections.Specialized.StringCollection RendererParameters {
            get {
                return ((global::System.Collections.Specialized.StringCollection)(this["RendererParameters"]));
            }
            set {
                this["RendererParameters"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("50")]
        public int MaxConsoleMessages {
            get {
                return ((int)(this["MaxConsoleMessages"]));
            }
            set {
                this["MaxConsoleMessages"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("0")]
        public uint LastSelectedRendererVariation {
            get {
                return ((uint)(this["LastSelectedRendererVariation"]));
            }
            set {
                this["LastSelectedRendererVariation"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("False")]
        public bool InvertCameraControls {
            get {
                return ((bool)(this["InvertCameraControls"]));
            }
            set {
                this["InvertCameraControls"] = value;
            }
        }
        
        [global::System.Configuration.UserScopedSettingAttribute()]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [global::System.Configuration.DefaultSettingValueAttribute("1000")]
        public uint LastNIterationCommand {
            get {
                return ((uint)(this["LastNIterationCommand"]));
            }
            set {
                this["LastNIterationCommand"] = value;
            }
        }
    }
}
