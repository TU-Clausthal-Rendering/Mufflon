using gui.Dll;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Resources;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Media.Effects;

namespace gui.View.Display
{
    // This shader effect multiplies every pixel with a factor
    public class GammaEffect : ShaderEffect
    {
        /* The source for the shader: 
         * 
         * sampler2D implicitInput : register(s0
         * float factor : register(c0);
         * 
         * float rgbToSrgb(float val) {
         *     if(val <= 0.04045f)
         *     return val / 12.92f;
         *     return pow((val + 0.055f) / 1.055f, 2.4f);
         * }
         * 
         * float sRgbToRgb(float val) {
         *     if(val <= 0.0031308f)return val * 12.92f;
         *     return 1.055f * pow(val, 1.0f/2.4f) - 0.055f;
         * }
         *
         * float4 main(float2 uv : TEXCOORD) : COLOR
         * {
         *     float4 color = tex2D(implicitInput, uv);
         *     float3 srgb = factor * float3(rgbToSrgb(color.r), rgbToSrgb(color.g), rgbToSrgb(color.b));
         *     float4 rgb = float4(sRgbToRgb(srgb.r), sRgbToRgb(srgb.g), sRgbToRgb(srgb.b), color.a);
         *     return rgb;
         * }
         * 
         * // The reason for the rgb -> srgb -> rgb conversion is that we can't tell WPF that we want to
         * // do something prior to the conversion
         */

        private static PixelShader s_pixelShader = new PixelShader() { UriSource = MakePackUri("View/Display/GammaEffect.fx.ps") };

        public GammaEffect()
        {
            PixelShader = s_pixelShader;
            UpdateShaderValue(InputProperty);
            UpdateShaderValue(FactorProperty);
        }

        // Taken from https://blogs.msdn.microsoft.com/greg_schechter/2008/05/12/writing-custom-gpu-based-effects-for-wpf/
        private static Uri MakePackUri(string relativeFile)
        {
            return new Uri("pack://application:,,,/" + AssemblyShortName + ";component/" + relativeFile);
        }

        private static string m_assemblyShortName = null;
        private static string AssemblyShortName
        {
            get
            {
                if(m_assemblyShortName == null)
                {
                    Assembly a = typeof(GammaEffect).Assembly;
                    // Pull out the short name
                    m_assemblyShortName = a.ToString().Split(',')[0];
                }
                return m_assemblyShortName;
            }
        }

        #region Brush dependency property
        public Brush Input
        {
            get { return (Brush)GetValue(InputProperty); }
            set { SetValue(InputProperty, value); }
        }

        public static readonly DependencyProperty InputProperty =
            ShaderEffect.RegisterPixelShaderSamplerProperty("Input", typeof(GammaEffect), 0);
        #endregion

        #region Factor dependency property
        public double Factor
        {
            get { return (double)GetValue(FactorProperty); }
            set { SetValue(FactorProperty, value); }
        }

        public static readonly DependencyProperty FactorProperty =
            DependencyProperty.Register("Factor", typeof(double), typeof(GammaEffect),
                    new UIPropertyMetadata(1.0, PixelShaderConstantCallback(0)));
        #endregion
    }
}
