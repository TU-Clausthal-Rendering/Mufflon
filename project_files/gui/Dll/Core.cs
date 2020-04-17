using gui.Utility;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Light;

namespace gui.Dll
{
    /// <summary>
    /// DLL communication with core.dll
    /// </summary>
    static public class Core
    {
        internal static IntPtr muffInstHdl;

        [StructLayout(LayoutKind.Sequential)]
        public struct Vec2
        {
            public float x, y;
            public Vec2(float a, float b)
            {
                x = a;
                y = b;
            }
            public Vec2(Vec2<float> vec)
            {
                x = vec.X;
                y = vec.Y;
            }

            public Vec2<float> ToUtilityVec()
            {
                return new Utility.Vec2<float>(x, y);
            }
        };
        [StructLayout(LayoutKind.Sequential)]
        public struct Vec3
        {
            public float x, y, z;
            public Vec3(float a, float b, float c)
            {
                x = a;
                y = b;
                z = c;
            }
            public Vec3(Vec3<float> vec)
            {
                x = vec.X;
                y = vec.Y;
                z = vec.Z;
            }

            public Vec3<float> ToUtilityVec()
            {
                return new Utility.Vec3<float>(x, y, z);
            }
        };
        [StructLayout(LayoutKind.Sequential)]
        public struct Vec4
        {
            public float x, y, z, w;
            public Vec4(float a, float b, float c, float d)
            {
                x = a;
                y = b;
                z = c;
                w = d;
            }
            public Vec4(Vec4<float> vec)
            {
                x = vec.X;
                y = vec.Y;
                z = vec.Z;
                w = vec.W;
            }

            public Vec4<float> ToUtilityVec()
            {
                return new Utility.Vec4<float>(x, y, z, w);
            }
        };
        [StructLayout(LayoutKind.Sequential)]
        public struct UVec3
        {
            public uint x, y, z;
            public UVec3(uint a, uint b, uint c)
            {
                x = a;
                y = b;
                z = c;
            }
            public UVec3(Vec3<uint> vec)
            {
                x = vec.X;
                y = vec.Y;
                z = vec.Z;
            }

            public Vec3<uint> ToUtilityVec()
            {
                return new Utility.Vec3<uint>(x, y, z);
            }
        };
        [StructLayout(LayoutKind.Sequential)]
        public struct UVec4
        {
            public uint x, y, z, w;
            public UVec4(uint a, uint b, uint c, uint d)
            {
                x = a;
                y = b;
                z = c;
                w = d;
            }
            public UVec4(Vec4<uint> vec)
            {
                x = vec.X;
                y = vec.Y;
                z = vec.Z;
                w = vec.W;
            }

            public Vec4<uint> ToUtilityVec()
            {
                return new Utility.Vec4<uint>(x, y, z, w);
            }
        };

        [StructLayout(LayoutKind.Sequential)]
        public struct ProcessTime
        {
            public UInt64 cycles;
            public UInt64 microseconds;

            public static ProcessTime operator+(ProcessTime t1, ProcessTime t2)
            {
                t1.cycles += t2.cycles;
                t1.microseconds += t2.microseconds;
                return t1;
            }
        };

        public enum GeomAttributeType
        {
            CHAR,
            UCHAR,
            SHORT,
            USHORT,
            INT,
            UINT,
            LONG,
            ULONG,
            FLOAT,
            DOUBLE,
            UCHAR2,
            UCHAR3,
            UCHAR4,
            INT2,
            INT3,
            INT4,
            FLOAT2,
            FLOAT3,
            FLOAT4,
            UVEC4,
            SPHERE
        };

        [StructLayout(LayoutKind.Sequential)]
        public struct VertexAttributeHdl
        {
            GeomAttributeType type;
            IntPtr name;
        };

        [StructLayout(LayoutKind.Sequential)]
        public struct FaceAttributeHdl
        {
            GeomAttributeType type;
            IntPtr name;
        };

        [StructLayout(LayoutKind.Sequential)]
        public struct SphereAttributeHdl
        {
            GeomAttributeType type;
            IntPtr name;
        };

        public enum CameraType
        {
            Pinhole,
            Focus
        };

        public enum LightType
        {
            Point,
            Spot,
            Directional,
            Envmap,
            Monochrome,
            Sky
        };

        public enum BackgroundType
        {
            Monochrome,
            Envmap,
            SkyHosek
        };

        public static LightType FromModelLightType(LightModel.LightType type)
        {
            switch (type)
            {
                case LightModel.LightType.Point:
                    return LightType.Point;
                case LightModel.LightType.Directional:
                    return LightType.Directional;
                case LightModel.LightType.Spot:
                    return LightType.Spot;
                case LightModel.LightType.Envmap:
                    return LightType.Envmap;
                case LightModel.LightType.Goniometric:
                    default:
                    throw new NotImplementedException();
            }
        }

        public enum ProfilingLevel
        {
            All,
            High,
            Low,
            Off
        };

        public enum ParameterType
        {
            Int,
            Float,
            Bool,
            Enum
        };

        public enum Severity
        {
            Pedantic,
            Info,
            Warning,
            Error,
            FatalError
        };

        public enum ObjectFlags
        {
            NONE,
            EMISSIVE
        };

        public enum TextureSampling
        {
            Nearest,
            Linear
        };


        public enum MipmapType
        {
            NONE,
            AVG,
            MIN,
            MAX
        };

        [Flags]
        public enum RenderDevice
        {
            None = 0,
            Cpu = 1,
            Cuda = 2,
            OpenGL = 4
        };

        public enum TextureFormat
        {
            R8U,
            RG8U,
            RGBA8U,
            R16U,
            RG16U,
            RGBA16U,
            R16F,
            RG16F,
            RGBA16F,
            R32F,
            RG32F,
            RGBA32F,
            Invalid
        };

        public delegate void LogCallback(string message, Severity severity);
        public delegate void LodLoaderFunc(IntPtr userParams, IntPtr objHdl, UInt32 UInt32);
        public delegate void ObjMatIndicesFunc(IntPtr userParams, UInt32 objId, IntPtr indices, out UInt32 count);
        public delegate Vec4 TextureCallback(UInt32 x, UInt32 y, UInt32 layer, TextureFormat format, Vec4 value, IntPtr userParams);


        // DLL interface
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "core_get_dll_error")]
        private static extern IntPtr core_get_dll_error_();
        internal static string core_get_dll_error() { return StringUtil.FromNativeUTF8(core_get_dll_error_()); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "core_set_logger")]
        internal static extern Boolean core_set_logger(LogCallback callback);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "core_set_log_level")]
        internal static extern Boolean core_set_log_level(Severity level);
        // General mufflon interface

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "mufflon_initialize")]
        private static extern IntPtr mufflon_initialize_();
        internal static bool mufflon_initialize() { muffInstHdl = mufflon_initialize_(); return muffInstHdl != IntPtr.Zero; }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "mufflon_destroy")]
        private static extern void mufflon_destroy_(IntPtr instHdl);
        internal static void mufflon_destroy() { mufflon_destroy_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "mufflon_initialize_opengl")]
        private static extern Boolean mufflon_initialize_opengl_(IntPtr instHdls);
        internal static Boolean mufflon_initialize_opengl() { return mufflon_initialize_opengl_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "mufflon_get_cuda_device_index")]
        internal static extern Int32 mufflon_get_cuda_device_index();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "mufflon_is_cuda_available")]
        internal static extern Boolean mufflon_is_cuda_available();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "mufflon_set_lod_loader")]
        private static extern Boolean mufflon_set_lod_loader_(IntPtr instHdl, LodLoaderFunc loader, ObjMatIndicesFunc objFunc, IntPtr userParams);
        internal static Boolean mufflon_set_lod_loader(LodLoaderFunc loader, ObjMatIndicesFunc objFunc, IntPtr userParams) { return mufflon_set_lod_loader_(muffInstHdl, loader, objFunc, userParams); }

        // Render image functions
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "mufflon_get_target_image")]
        private static extern Boolean mufflon_get_target_image_(IntPtr instHdl, IntPtr name, UInt32 variance, out IntPtr ptr);
        internal static Boolean mufflon_get_target_image(string name, bool variance, out IntPtr ptr) { return mufflon_get_target_image_(muffInstHdl, StringUtil.ToNativeUtf8(name), variance ? 1u : 0u, out ptr); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "mufflon_get_target_image_num_channels")]
        private static extern Boolean mufflon_get_target_image_num_channels_(IntPtr instHdl, IntPtr numChannels);
        internal static Boolean mufflon_get_target_image_num_channels(IntPtr numChannels) { return mufflon_get_target_image_num_channels_(muffInstHdl, numChannels); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "mufflon_copy_screen_texture_rgba32")]
        private static extern Boolean mufflon_copy_screen_texture_rgba32_(IntPtr instHdl, IntPtr ptr, float factor);
        internal static Boolean mufflon_copy_screen_texture_rgba32(IntPtr ptr, float factor) { return mufflon_copy_screen_texture_rgba32_(muffInstHdl, ptr, factor); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "mufflon_get_pixel_info")]
        private static extern Boolean mufflon_get_pixel_info_(IntPtr instHdl, UInt32 x, UInt32 y, UInt32 borderClamp, out float r, out float g, out float b, out float a);
        internal static Boolean mufflon_get_pixel_info(UInt32 x, UInt32 y, bool borderClamp, out float r, out float g, out float b, out float a) {
            return mufflon_get_pixel_info_(muffInstHdl, x, y, borderClamp ? 1u : 0u, out r, out g, out b, out a);
        }

        // World loading/clearing
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_clear_all")]
        private static extern void world_clear_all_(IntPtr instHdl);
        internal static void world_clear_all() { world_clear_all_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_finalize")]
        private static extern Boolean world_finalize_(IntPtr instHdl, IntPtr msg);
        internal static Boolean world_finalize(IntPtr msg) { return world_finalize_(muffInstHdl, msg); }

        // Polygon interface
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_reserve")]
        internal static extern Boolean polygon_reserve(IntPtr lvlDtl, ulong vertices, ulong tris, ulong quads);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_request_vertex_attribute")]
        internal static extern VertexAttributeHdl polygon_request_vertex_attribute(IntPtr lvlDtl, IntPtr name, GeomAttributeType type);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_request_face_attribute")]
        internal static extern FaceAttributeHdl polygon_request_face_attribute(IntPtr lvlDtl, IntPtr name, GeomAttributeType type);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_add_vertex")]
        internal static extern Int32 polygon_add_vertex(IntPtr lvlDtl, Vec3 point, Vec3 normal, Vec2 uv);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_add_triangle")]
        internal static extern Int32 polygon_add_triangle(IntPtr lvlDtl, UVec3 vertices);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_add_triangle_material")]
        internal static extern Int32 polygon_add_triangle_material(IntPtr lvlDtl, UVec3 vertices, UInt16 idx);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_add_quad")]
        internal static extern Int32 polygon_add_quad(IntPtr lvlDtl, UVec4 vertices);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_add_quad_material")]
        internal static extern Int32 polygon_add_quad_material(IntPtr lvlDtl, UVec4 vertices, UInt16 idx);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_add_vertex_bulk")]
        internal static extern Int32 polygon_add_vertex_bulk(IntPtr lvlDtl, IntPtr uvsRead);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_set_vertex_attribute")]
        internal static extern Boolean polygon_set_vertex_attribute(IntPtr lvlDtl, IntPtr value);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_set_vertex_normal")]
        internal static extern Boolean polygon_set_vertex_normal(IntPtr lvlDtl, Int32 vertex, Vec3 normal);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_set_vertex_uv")]
        internal static extern Boolean polygon_set_vertex_uv(IntPtr lvlDtl, Int32 vertex, Vec2 uv);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_set_face_attribute")]
        internal static extern Boolean polygon_set_face_attribute(IntPtr lvlDtl, IntPtr value);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_set_material_idx")]
        internal static extern Boolean polygon_set_material_idx(IntPtr lvlDtl, Int32 face, UInt16 idx);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_set_vertex_attribute_bulk")]
        internal static extern ulong polygon_set_vertex_attribute_bulk(IntPtr lvlDtl, IntPtr stream);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_set_face_attribute_bulk")]
        internal static extern ulong polygon_set_face_attribute_bulk(IntPtr lvlDtl, IntPtr stream);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_set_material_idx_bulk")]
        internal static extern ulong polygon_set_material_idx_bulk(IntPtr lvlDtl, IntPtr stream);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_get_vertex_count")]
        internal static extern ulong polygon_get_edge_count(IntPtr lvlDtl);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_get_face_count")]
        internal static extern ulong polygon_get_face_count(IntPtr lvlDtl);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_get_triangle_count")]
        internal static extern ulong polygon_get_triangle_count(IntPtr lvlDtl);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_get_quad_count")]
        internal static extern ulong polygon_get_quad_count(IntPtr lvlDtl);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "polygon_get_bounding_box")]
        internal static extern Boolean polygon_get_bounding_box(IntPtr lvlDtl, IntPtr max);

        // Sphere interface
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "spheres_reserve")]
        internal static extern Boolean spheres_reserve(IntPtr lvlDtl, ulong count);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "spheres_request_attribute")]
        internal static extern SphereAttributeHdl spheres_request_attribute(IntPtr lvlDtl, IntPtr name, GeomAttributeType type);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "spheres_add_sphere")]
        internal static extern Int32 spheres_add_sphere(IntPtr lvlDtl, Vec3 point, float radius);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "spheres_add_sphere_material")]
        internal static extern Int32 spheres_add_sphere_material(IntPtr lvlDtl, Vec3 point, float radius, UInt16 idx);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "spheres_add_sphere_bulk")]
        internal static extern Int32 spheres_add_sphere_bulk(IntPtr lvlDtl, IntPtr readSpheres);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "spheres_set_attribute")]
        internal static extern Boolean spheres_set_attribute(IntPtr lvlDtl, IntPtr value);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "spheres_set_material_idx")]
        internal static extern Boolean spheres_set_material_idx(IntPtr lvlDtl, Int32 sphere, UInt16 idx);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "spheres_set_attribute_bulk")]
        internal static extern ulong spheres_set_attribute_bulk(IntPtr lvlDtl, IntPtr stream);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "spheres_set_material_idx_bulk")]
        internal static extern ulong spheres_set_material_idx_bulk(IntPtr lvlDtl, IntPtr stream);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "spheres_get_sphere_count")]
        internal static extern ulong spheres_get_sphere_count(IntPtr lvlDtl);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "spheres_get_bounding_box")]
        internal static extern Boolean spheres_get_bounding_box(IntPtr lvlDtl, IntPtr max);

        // Object interface
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_create_object")]
        private static extern IntPtr world_create_object_(IntPtr instHdl, IntPtr name, ObjectFlags flags);
        internal static IntPtr world_create_object(string name, ObjectFlags flags) { return world_create_object_(muffInstHdl, StringUtil.ToNativeUtf8(name), flags); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_object")]
        private static extern IntPtr world_get_object_(IntPtr instHdl, IntPtr name);
        internal static IntPtr world_get_object(string name) { return world_get_object_(muffInstHdl, StringUtil.ToNativeUtf8(name)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_object_name")]
        private static extern IntPtr world_get_object_name_(IntPtr obj);
        internal static string world_get_object_name(IntPtr obj) { return StringUtil.FromNativeUTF8(world_get_object_name_(obj)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "object_has_lod")]
        internal static extern Boolean object_has_lod(IntPtr obj, UInt32 level);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "object_add_lod")]
        internal static extern IntPtr object_add_lod(IntPtr obj, UInt32 level, UInt32 asReduced);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "object_get_id")]
        internal static extern Boolean object_get_id(IntPtr obj, IntPtr id);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_reserve_objects_instances")]


        // Instance interface
        private static extern void world_reserve_objects_instances_(IntPtr instHdl, UInt32 objects, UInt32 instances);
        internal static void world_reserve_objects_instances(UInt32 objects, UInt32 instances) { world_reserve_objects_instances_(muffInstHdl, objects, instances); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_create_instance")]
        private static extern IntPtr world_create_instance_(IntPtr instHdl, IntPtr obj, UInt32 animationFrame);
        internal static IntPtr world_create_instance(IntPtr obj, UInt32 animationFrame) { return world_create_instance_(muffInstHdl, obj, animationFrame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_instance_count")]
        private static extern UInt32 world_get_instance_count_(IntPtr instHdl, UInt32 frame);
        internal static UInt32 world_get_instance_count(UInt32 frame) { return world_get_instance_count_(muffInstHdl, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_instance_by_index")]
        private static extern IntPtr world_get_instance_by_index_(IntPtr instHdl, UInt32 index, UInt32 animationFrame);
        internal static IntPtr world_get_instance_by_index(UInt32 index, UInt32 animationFrame) { return world_get_instance_by_index_(muffInstHdl, index, animationFrame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_apply_instance_transformation")]
        private static extern Boolean world_apply_instance_transformation_(IntPtr instHdl, IntPtr inst);
        internal static Boolean world_apply_instance_transformation(IntPtr inst) { return world_apply_instance_transformation_(muffInstHdl, inst); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "instance_set_transformation_matrix")]
        private static extern Boolean instance_set_transformation_matrix_(IntPtr instHdl, IntPtr inst, IntPtr mat, Boolean isWorldToInst);
        internal static Boolean instance_set_transformation_matrix(IntPtr inst, IntPtr mat, Boolean isWorldToInst) { return instance_set_transformation_matrix_(muffInstHdl, inst, mat, isWorldToInst); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "instance_get_transformation_matrix")]
        private static extern Boolean instance_get_transformation_matrix_(IntPtr instHdl, IntPtr inst, IntPtr mat);
        internal static Boolean instance_get_transformation_matrix(IntPtr inst, IntPtr mat) { return instance_get_transformation_matrix_(muffInstHdl, inst, mat); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "instance_get_bounding_box")]
        private static extern Boolean instance_get_bounding_box_(IntPtr instHdl, IntPtr inst, IntPtr max, UInt32 lod);
        internal static Boolean instance_get_bounding_box(IntPtr inst, IntPtr max, UInt32 lod) { return instance_get_bounding_box_(muffInstHdl, inst, max, lod); }

        // Animation interface
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_frame_current")]
        private static extern Boolean world_set_frame_current_(IntPtr instHdl, UInt32 animationFrame);
        internal static Boolean world_set_frame_current(UInt32 animationFrame) { return world_set_frame_current_(muffInstHdl, animationFrame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_frame_current")]
        private static extern Boolean world_get_frame_current_(IntPtr instHdl, out UInt32 animationFrame);
        internal static Boolean world_get_frame_current(out UInt32 animationFrame) { return world_get_frame_current_(muffInstHdl, out animationFrame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_frame_count")]
        private static extern Boolean world_get_frame_count_(IntPtr instHdl, out UInt32 frameCount);
        internal static Boolean world_get_frame_count(out UInt32 frameCount) { return world_get_frame_count_(muffInstHdl, out frameCount); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_highest_instance_frame")]
        private static extern UInt32 world_get_highest_instance_frame_(IntPtr instHdl);
        internal static UInt32 world_get_highest_instance_frame() { return world_get_highest_instance_frame_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_reserve_animation")]
        private static extern void world_reserve_animation_(IntPtr instHdl, UInt32 numBones, UInt32 frameCount);
        internal static void world_reserve_animation(UInt32 numBones, UInt32 frameCount) { world_reserve_animation_(muffInstHdl, numBones, frameCount); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_bone")]
        private static extern void world_set_bone_(IntPtr instHdl, UInt32 boneIndex, IntPtr transformation);
        internal static void world_set_bone(UInt32 boneIndex, IntPtr transformation) { world_set_bone_(muffInstHdl, boneIndex, transformation); }

        // Tessellation interface
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_tessellation_level")]
        private static extern void world_set_tessellation_level_(IntPtr instHdl, float maxTessLevel);
        internal static void world_set_tessellation_level(float maxTessLevel) { world_set_tessellation_level_(muffInstHdl, maxTessLevel); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_tessellation_level")]
        private static extern float world_get_tessellation_level_(IntPtr instHdl);
        internal static float world_get_tessellation_level() { return world_get_tessellation_level_(muffInstHdl); }

        // Material interface
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_add_material")]
        private static extern IntPtr world_add_material_(IntPtr instHdl, IntPtr name, IntPtr mat);
        internal static IntPtr world_add_material(string name, IntPtr mat) { return world_add_material_(muffInstHdl, StringUtil.ToNativeUtf8(name), mat); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_material_count")]
        private static extern Int32 world_get_material_count_(IntPtr instHdl);
        internal static Int32 world_get_material_count() { return world_get_material_count_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_material")]
        private static extern IntPtr world_get_material_(IntPtr instHdl, Int32 index);
        internal static IntPtr world_get_material(Int32 index) { return world_get_material_(muffInstHdl, index); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_material_size")]
        internal static extern ulong world_get_material_size(IntPtr material);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_material_name")]
        private static extern IntPtr world_get_material_name_(IntPtr material);
        internal static string world_get_material_name(IntPtr material) { return StringUtil.FromNativeUTF8(world_get_material_name_(material)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_material_data")]
        private static extern Boolean world_get_material_data_(IntPtr instHdl, IntPtr material, IntPtr buffer);
        internal static Boolean world_get_material_data(IntPtr material, IntPtr buffer) { return world_get_material_data_(muffInstHdl, material, buffer); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_texture")]

        // Texture interface
        private static extern IntPtr world_get_texture_(IntPtr instHdl, IntPtr path);
        internal static IntPtr world_get_texture(string path) { return world_get_texture_(muffInstHdl, StringUtil.ToNativeUtf8(path)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_add_texture")]
        private static extern IntPtr world_add_texture_(IntPtr instHdl, IntPtr path, TextureSampling sampling, MipmapType type, TextureCallback callback, IntPtr userParams);
        internal static IntPtr world_add_texture(string path, TextureSampling sampling, MipmapType type, TextureCallback callback, IntPtr userParams) {
            return world_add_texture_(muffInstHdl, StringUtil.ToNativeUtf8(path), sampling, type, callback, userParams);
        }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_add_texture_converted")]
        private static extern IntPtr world_add_texture_converted_(IntPtr instHdl, IntPtr path, IntPtr userParams);
        internal static IntPtr world_add_texture_converted(string path, IntPtr userParams) { return world_add_texture_converted_(muffInstHdl, StringUtil.ToNativeUtf8(path), userParams); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_add_texture_value")]
        private static extern IntPtr world_add_texture_value_(IntPtr instHdl, IntPtr value, int num, TextureSampling sampling);
        internal static IntPtr world_add_texture_value(IntPtr value, int num, TextureSampling sampling) { return world_add_texture_value_(muffInstHdl, value, num, sampling); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_add_displacement_map")]
        private static extern Boolean world_add_displacement_map_(IntPtr instHdl, IntPtr path, IntPtr hdlMips);
        internal static Boolean world_add_displacement_map(string path, IntPtr hdlMips) { return world_add_displacement_map_(muffInstHdl, StringUtil.ToNativeUtf8(path), hdlMips); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_texture_name")]
        private static extern IntPtr world_get_texture_name_(IntPtr hdl);
        internal static string world_get_texture_name(IntPtr hdl) { return StringUtil.FromNativeUTF8(world_get_texture_name_(hdl)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_texture_size")]
        internal static extern Boolean world_get_texture_size(IntPtr hdl, IntPtr size);

        // Camera interface
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_add_pinhole_camera")]
        private static extern IntPtr world_add_pinhole_camera_(IntPtr instHdl, IntPtr name, IntPtr up, UInt32 pathCount, float near, float far, float vFov);
        internal static IntPtr world_add_pinhole_camera(string name, IntPtr up, UInt32 pathCount, float near, float far, float vFov) {
            return world_add_pinhole_camera_(muffInstHdl, StringUtil.ToNativeUtf8(name), up, pathCount, near, far, vFov);
        }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_add_focus_camera")]
        private static extern IntPtr world_add_focus_camera_(IntPtr instHdl, IntPtr name, IntPtr up, UInt32 pathCount, float near, float far, float focalLength, float focusDistance, float lensRad, float chipHeight);
        internal static IntPtr world_add_focus_camera(string name, IntPtr up, UInt32 pathCount, float near, float far, float focalLength, float focusDistance, float lensRad, float chipHeight) {
            return world_add_focus_camera_(muffInstHdl, StringUtil.ToNativeUtf8(name), up, pathCount, near, far, focalLength, focusDistance, lensRad, chipHeight);
        }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_remove_camera")]
        private static extern Boolean world_remove_camera_(IntPtr instHdl, IntPtr hdl);
        internal static Boolean world_remove_camera(IntPtr hdl) { return world_remove_camera_(muffInstHdl, hdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_count")]
        private static extern ulong world_get_camera_count_(IntPtr instHdl);
        internal static ulong world_get_camera_count() { return world_get_camera_count_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera")]
        private static extern IntPtr world_get_camera_(IntPtr instHdl, IntPtr name);
        internal static IntPtr world_get_camera(string name) { return world_get_camera_(muffInstHdl, StringUtil.ToNativeUtf8(name)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_by_index")]
        private static extern IntPtr world_get_camera_by_index_(IntPtr instHdl, ulong index);
        internal static IntPtr world_get_camera_by_index(ulong index) { return world_get_camera_by_index_(muffInstHdl, index); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_type")]
        internal static extern CameraType world_get_camera_type(IntPtr cam);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_name")]
        private static extern IntPtr world_get_camera_name_(IntPtr cam);
        internal static string world_get_camera_name(IntPtr cam) { return StringUtil.FromNativeUTF8(world_get_camera_name_(cam)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_path_segment_count")]
        internal static extern Boolean world_get_camera_path_segment_count(IntPtr cam, out UInt32 segments);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_position")]
        internal static extern Boolean world_get_camera_position(IntPtr cam, out Vec3 pos, UInt32 pathIndex);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_current_position")]
        private static extern Boolean world_get_camera_current_position_(IntPtr instHdl, IntPtr cam, out Vec3 pos);
        internal static Boolean world_get_camera_current_position(IntPtr cam, out Vec3 pos) { return world_get_camera_current_position_(muffInstHdl, cam, out pos); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_direction")]
        internal static extern Boolean world_get_camera_direction(IntPtr cam, out Vec3 dir, UInt32 pathIndex);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_current_direction")]
        private static extern Boolean world_get_camera_current_direction_(IntPtr instHdl, IntPtr cam, out Vec3 dir);
        internal static Boolean world_get_camera_current_direction(IntPtr cam, out Vec3 dir) { return world_get_camera_current_direction_(muffInstHdl, cam, out dir); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_up")]
        internal static extern Boolean world_get_camera_up(IntPtr cam, out Vec3 up, UInt32 pathIndex);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_current_up")]
        private static extern Boolean world_get_camera_current_up_(IntPtr instHdl, IntPtr cam, out Vec3 up);
        internal static Boolean world_get_camera_current_up(IntPtr cam, out Vec3 up) { return world_get_camera_current_up_(muffInstHdl, cam, out up); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_near")]
        internal static extern Boolean world_get_camera_near(IntPtr cam, out float near);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_far")]
        internal static extern Boolean world_get_camera_far(IntPtr cam, out float far);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_camera_position")]
        internal static extern Boolean world_set_camera_position(IntPtr cam, Vec3 pos, UInt32 pathIndex);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_camera_current_position")]
        private static extern Boolean world_set_camera_current_position_(IntPtr instHdl, IntPtr cam, Vec3 pos);
        internal static Boolean world_set_camera_current_position(IntPtr cam, Vec3 pos) { return world_set_camera_current_position_(muffInstHdl, cam, pos); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_camera_direction")]
        internal static extern Boolean world_set_camera_direction(IntPtr cam, Vec3 dir, Vec3 up, UInt32 pathIndex);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_camera_current_direction")]
        private static extern Boolean world_set_camera_current_direction_(IntPtr instHdl, IntPtr cam, Vec3 dir, Vec3 up);
        internal static Boolean world_set_camera_current_direction(IntPtr cam, Vec3 dir, Vec3 up) { return world_set_camera_current_direction_(muffInstHdl, cam, dir, up); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_camera_near")]
        internal static extern Boolean world_set_camera_near(IntPtr cam, float near);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_camera_far")]
        internal static extern Boolean world_set_camera_far(IntPtr cam, float far);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_pinhole_camera_fov")]
        internal static extern Boolean world_get_pinhole_camera_fov(IntPtr cam, out float vFov);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_pinhole_camera_fov")]
        internal static extern Boolean world_set_pinhole_camera_fov(IntPtr cam, float vFov);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_focus_camera_focal_length")]
        internal static extern Boolean world_get_focus_camera_focal_length(IntPtr cam, out float focalLength);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_focus_camera_focus_distance")]
        internal static extern Boolean world_get_focus_camera_focus_distance(IntPtr cam, out float focusDistance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_focus_camera_sensor_height")]
        internal static extern Boolean world_get_focus_camera_sensor_height(IntPtr cam, out float sensorHeight);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_focus_camera_aperture")]
        internal static extern Boolean world_get_focus_camera_aperture(IntPtr cam, out float aperture);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_focus_camera_focal_length")]
        internal static extern Boolean world_set_focus_camera_focal_length(IntPtr cam, float focalLength);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_focus_camera_focus_distance")]
        internal static extern Boolean world_set_focus_camera_focus_distance(IntPtr cam, float focusDistance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_focus_camera_sensor_height")]
        internal static extern Boolean world_set_focus_camera_sensor_height(IntPtr cam, float sensorHeight);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_focus_camera_aperture")]
        internal static extern Boolean world_set_focus_camera_aperture(IntPtr cam, float aperture);

        // Scenario interface
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_reserve_scenarios")]
        private static extern void world_reserve_scenarios_(IntPtr instHdl, UInt32 scenarios);
        internal static void world_reserve_scenarios(UInt32 scenarios) { world_reserve_scenarios_(muffInstHdl, scenarios); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_create_scenario")]
        private static extern IntPtr world_create_scenario_(IntPtr instHdl, IntPtr name);
        internal static IntPtr world_create_scenario(string name) { return world_create_scenario_(muffInstHdl, StringUtil.ToNativeUtf8(name)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_find_scenario")]
        private static extern IntPtr world_find_scenario_(IntPtr instHdl, IntPtr name);
        internal static IntPtr world_find_scenario(string name) { return world_find_scenario_(muffInstHdl, StringUtil.ToNativeUtf8(name)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_scenario_count")]
        private static extern UInt32 world_get_scenario_count_(IntPtr instHdl);
        internal static UInt32 world_get_scenario_count() { return world_get_scenario_count_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_scenario_by_index")]
        private static extern IntPtr world_get_scenario_by_index_(IntPtr instHdl, UInt32 index);
        internal static IntPtr world_get_scenario_by_index(UInt32 index) { return world_get_scenario_by_index_(muffInstHdl, index); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_current_scenario")]
        private static extern IntPtr world_get_current_scenario_(IntPtr instHdl);
        internal static IntPtr world_get_current_scenario() { return world_get_current_scenario_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_finalize_scenario")]
        private static extern Boolean world_finalize_scenario_(IntPtr instHdl, IntPtr scenario, IntPtr msg);
        internal static Boolean world_finalize_scenario(IntPtr scenario, IntPtr msg) { return world_finalize_scenario_(muffInstHdl, scenario, msg); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_load_scenario")]
        private static extern IntPtr world_load_scenario_(IntPtr instHdl, IntPtr scenario);
        internal static IntPtr world_load_scenario(IntPtr scenario) { return world_load_scenario_(muffInstHdl, scenario); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_name")]
        private static extern IntPtr scenario_get_name_(IntPtr scenario);
        internal static string scenario_get_name(IntPtr scenario) { return StringUtil.FromNativeUTF8(scenario_get_name_(scenario)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_global_lod_level")]
        internal static extern UInt32 scenario_get_global_lod_level(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_set_global_lod_level")]
        internal static extern Boolean scenario_set_global_lod_level(IntPtr scenario, UInt32 level);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_resolution")]
        internal static extern Boolean scenario_get_resolution(IntPtr scenario, out UInt32 width, out UInt32 height);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_set_resolution")]
        internal static extern Boolean scenario_set_resolution(IntPtr scenario, UInt32 width, UInt32 height);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_camera")]
        internal static extern IntPtr scenario_get_camera(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_set_camera")]
        private static extern Boolean scenario_set_camera_(IntPtr instHdl, IntPtr scenario, IntPtr cam);
        internal static Boolean scenario_set_camera(IntPtr scenario, IntPtr cam) { return scenario_set_camera_(muffInstHdl, scenario, cam); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_is_object_masked")]
        internal static extern Boolean scenario_is_object_masked(IntPtr scenario, IntPtr obj);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_mask_object")]
        internal static extern Boolean scenario_mask_object(IntPtr scenario, IntPtr obj);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_mask_instance")]
        internal static extern Boolean scenario_mask_instance(IntPtr scenario, IntPtr inst);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_set_object_tessellation_level")]
        internal static extern Boolean scenario_set_object_tessellation_level(IntPtr scenario, IntPtr hdl, float level);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_set_object_adaptive_tessellation")]
        internal static extern Boolean scenario_set_object_adaptive_tessellation(IntPtr scenario, IntPtr hdl, Boolean value);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_set_object_phong_tessellation")]
        internal static extern Boolean scenario_set_object_phong_tessellation(IntPtr scenario, IntPtr hdl, Boolean value);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_has_object_tessellation_info")]
        internal static extern Boolean scenario_has_object_tessellation_info(IntPtr scenario, IntPtr value);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_object_tessellation_level")]
        private static extern Boolean scenario_get_object_tessellation_level_(IntPtr instHdl, IntPtr scenario, out float level);
        internal static Boolean scenario_get_object_tessellation_level(IntPtr scenario, out float level) { return scenario_get_object_tessellation_level_(muffInstHdl, scenario, out level); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_object_adaptive_tessellation")]
        internal static extern Boolean scenario_get_object_adaptive_tessellation(IntPtr scenario, IntPtr value);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_object_phong_tessellation")]
        internal static extern Boolean scenario_get_object_phong_tessellation(IntPtr scenario, IntPtr value);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_object_lod")]
        internal static extern UInt32 scenario_get_object_lod(IntPtr scenario, IntPtr obj);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_set_object_lod")]
        internal static extern Boolean scenario_set_object_lod(IntPtr scenario, IntPtr obj, UInt32 level);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_set_instance_lod")]
        internal static extern Boolean scenario_set_instance_lod(IntPtr scenario, IntPtr inst, UInt32 level);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_point_light_count")]
        internal static extern Int32 scenario_get_point_light_count(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_spot_light_count")]
        internal static extern Int32 scenario_get_spot_light_count(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_dir_light_count")]
        internal static extern Int32 scenario_get_dir_light_count(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_has_envmap_light")]
        private static extern Boolean scenario_has_envmap_light_(IntPtr instHdl, IntPtr scenario);
        internal static Boolean scenario_has_envmap_light(IntPtr scenario) { return scenario_has_envmap_light_(muffInstHdl, scenario); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_light_handle")]
        internal static extern UInt32 scenario_get_light_handle(IntPtr scenario, Int32 index, LightType type);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_add_light")]
        private static extern Boolean scenario_add_light_(IntPtr instHdl, IntPtr scenario, UInt32 hdl);
        internal static Boolean scenario_add_light(IntPtr scenario, UInt32 hdl) { return scenario_add_light_(muffInstHdl, scenario, hdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_remove_light")]
        private static extern Boolean scenario_remove_light_(IntPtr instHdl, IntPtr scenario, UInt32 hdl);
        internal static Boolean scenario_remove_light(IntPtr scenario, UInt32 hdl) { return scenario_remove_light_(muffInstHdl, scenario, hdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_reserve_material_slots")]
        internal static extern void scenario_reserve_material_slots(IntPtr scenario, ulong count);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_reserve_custom_object_properties")]
        internal static extern void scenario_reserve_custom_object_properties(IntPtr scenario, ulong objects);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_reserve_custom_instance_properties")]
        internal static extern void scenario_reserve_custom_instance_properties(IntPtr scenario, ulong instances);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_declare_material_slot")]
        internal static extern UInt16 scenario_declare_material_slot(IntPtr scenario, IntPtr name, ulong nameLength);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_material_slot")]
        internal static extern UInt16 scenario_get_material_slot(IntPtr scenario, IntPtr name, ulong nameLength);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_material_slot_name")]
        private static extern IntPtr scenario_get_material_slot_name_(IntPtr scenario, UInt16 slot);
        internal static string scenario_get_material_slot_name(IntPtr scenario, UInt16 slot) { return StringUtil.FromNativeUTF8(scenario_get_material_slot_name_(scenario, slot)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_material_slot_count")]
        internal static extern ulong scenario_get_material_slot_count(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_assigned_material")]
        internal static extern IntPtr scenario_get_assigned_material(IntPtr scenario, UInt16 index);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_assign_material")]
        internal static extern Boolean scenario_assign_material(IntPtr scenario, UInt16 index, IntPtr handle);

        // Scene interface
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_current_scene")]
        private static extern IntPtr world_get_current_scene_(IntPtr instHdl);
        internal static IntPtr world_get_current_scene() { return world_get_current_scene_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scene_get_bounding_box")]
        internal static extern Boolean scene_get_bounding_box(IntPtr scene, out Vec3 min, out Vec3 max);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scene_get_camera")]
        internal static extern IntPtr scene_get_camera(IntPtr scene);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scene_move_active_camera")]
        private static extern Boolean scene_move_active_camera_(IntPtr instHdl, float x, float y, float z);
        internal static Boolean scene_move_active_camera(float x, float y, float z) { return scene_move_active_camera_(muffInstHdl, x, y, z); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scene_rotate_active_camera")]
        private static extern Boolean scene_rotate_active_camera_(IntPtr instHdl, float x, float y, float z);
        internal static Boolean scene_rotate_active_camera(float x, float y, float z) { return scene_rotate_active_camera_(muffInstHdl, x, y, z); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scene_is_sane")]
        private static extern Boolean scene_is_sane_(IntPtr instHdl);
        internal static Boolean scene_is_sane() { return scene_is_sane_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scene_request_retessellation")]
        private static extern Boolean scene_request_retessellation_(IntPtr instHdl);
        internal static Boolean scene_request_retessellation() { return scene_request_retessellation_(muffInstHdl); }

        // Light interface
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_add_light")]
        private static extern UInt32 world_add_light_(IntPtr instHdl, IntPtr name, LightType type, UInt32 count);
        internal static UInt32 world_add_light(string name, LightType type, UInt32 count) { return world_add_light_(muffInstHdl, StringUtil.ToNativeUtf8(name), type, count); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_add_background_light")]
        private static extern UInt32 world_add_background_light_(IntPtr instHdl, IntPtr name, BackgroundType type);
        internal static UInt32 world_add_background_light(string name, BackgroundType type) { return world_add_background_light_(muffInstHdl, StringUtil.ToNativeUtf8(name), type); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_light_name")]
        private static extern Boolean world_set_light_name_(IntPtr instHdl, UInt32 hdl, IntPtr newName);
        internal static Boolean world_set_light_name(UInt32 hdl, string newName) { return world_set_light_name_(muffInstHdl, hdl, StringUtil.ToNativeUtf8(newName)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_remove_light")]
        private static extern Boolean world_remove_light_(IntPtr instHdl, UInt32 hdl);
        internal static Boolean world_remove_light(UInt32 hdl) { return world_remove_light_(muffInstHdl, hdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_find_light")]
        private static extern Boolean world_find_light_(IntPtr instHdl, IntPtr name, IntPtr hdl);
        internal static Boolean world_find_light(string name, IntPtr hdl) { return world_find_light_(muffInstHdl, StringUtil.ToNativeUtf8(name), hdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_point_light_count")]
        private static extern ulong world_get_point_light_count_(IntPtr instHdl);
        internal static ulong world_get_point_light_count() { return world_get_point_light_count_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_spot_light_count")]
        private static extern ulong world_get_spot_light_count_(IntPtr instHdl);
        internal static ulong world_get_spot_light_count() { return world_get_spot_light_count_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_dir_light_count")]
        private static extern ulong world_get_dir_light_count_(IntPtr instHdl);
        internal static ulong world_get_dir_light_count() { return world_get_dir_light_count_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_env_light_count")]
        private static extern ulong world_get_env_light_count_(IntPtr instHdl);
        internal static ulong world_get_env_light_count() { return world_get_env_light_count_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_light_handle")]
        internal static extern UInt32 world_get_light_handle(ulong index, LightType type);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_light_type")]
        internal static extern LightType world_get_light_type(UInt32 hdl);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_env_light_type")]
        private static extern BackgroundType world_get_env_light_type_(IntPtr instHdl, UInt32 hdl);
        internal static BackgroundType world_get_env_light_type(UInt32 hdl) { return world_get_env_light_type_(muffInstHdl, hdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_light_name")]
        private static extern IntPtr world_get_light_name_(IntPtr instHdl, UInt32 hdl);
        internal static string world_get_light_name(UInt32 hdl) { return StringUtil.FromNativeUTF8(world_get_light_name_(muffInstHdl, hdl)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_point_light_position")]
        private static extern Boolean world_get_point_light_position_(IntPtr instHdl, UInt32 hdl, out Vec3 pos, UInt32 frame);
        internal static Boolean world_get_point_light_position(UInt32 hdl, out Vec3 pos, UInt32 frame) { return world_get_point_light_position_(muffInstHdl, hdl, out pos, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_point_light_intensity")]
        private static extern Boolean world_get_point_light_intensity_(IntPtr instHdl, UInt32 hdl, out Vec3 intensity, UInt32 frame);
        internal static Boolean world_get_point_light_intensity(UInt32 hdl, out Vec3 intensity, UInt32 frame) { return world_get_point_light_intensity_(muffInstHdl, hdl, out intensity, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_point_light_path_segments")]
        private static extern Boolean world_get_point_light_path_segments_(IntPtr instHdl, UInt32 hdl, out UInt32 segments);
        internal static Boolean world_get_point_light_path_segments(UInt32 hdl, out UInt32 segments) { return world_get_point_light_path_segments_(muffInstHdl, hdl, out segments); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_point_light_position")]
        private static extern Boolean world_set_point_light_position_(IntPtr instHdl, UInt32 hdl, Vec3 pos, UInt32 frame);
        internal static Boolean world_set_point_light_position(UInt32 hdl, Vec3 pos, UInt32 frame) { return world_set_point_light_position_(muffInstHdl, hdl, pos, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_point_light_intensity")]
        private static extern Boolean world_set_point_light_intensity_(IntPtr instHdl, UInt32 hdl, Vec3 intensity, UInt32 frame);
        internal static Boolean world_set_point_light_intensity(UInt32 hdl, Vec3 intensity, UInt32 frame) { return world_set_point_light_intensity_(muffInstHdl, hdl, intensity, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_spot_light_path_segments")]
        private static extern Boolean world_get_spot_light_path_segments_(IntPtr instHdl, UInt32 hdl, out UInt32 segments);
        internal static Boolean world_get_spot_light_path_segments(UInt32 hdl, out UInt32 segments) { return world_get_spot_light_path_segments_(muffInstHdl, hdl, out segments); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_spot_light_position")]
        private static extern Boolean world_get_spot_light_position_(IntPtr instHdl, UInt32 hdl, out Vec3 pos, UInt32 frame);
        internal static Boolean world_get_spot_light_position(UInt32 hdl, out Vec3 pos, UInt32 frame) { return world_get_spot_light_position_(muffInstHdl, hdl, out pos, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_spot_light_intensity")]
        private static extern Boolean world_get_spot_light_intensity_(IntPtr instHdl, UInt32 hdl, out Vec3 intensity, UInt32 frame);
        internal static Boolean world_get_spot_light_intensity(UInt32 hdl, out Vec3 intensity, UInt32 frame) { return world_get_spot_light_intensity_(muffInstHdl, hdl, out intensity, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_spot_light_direction")]
        private static extern Boolean world_get_spot_light_direction_(IntPtr instHdl, UInt32 hdl, out Vec3 direction, UInt32 frame);
        internal static Boolean world_get_spot_light_direction(UInt32 hdl, out Vec3 direction, UInt32 frame) { return world_get_spot_light_direction_(muffInstHdl, hdl, out direction, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_spot_light_angle")]
        private static extern Boolean world_get_spot_light_angle_(IntPtr instHdl, UInt32 hdl, out float angle, UInt32 frame);
        internal static Boolean world_get_spot_light_angle(UInt32 hdl, out float angle, UInt32 frame) { return world_get_spot_light_angle_(muffInstHdl, hdl, out angle, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_spot_light_falloff")]
        private static extern Boolean world_get_spot_light_falloff_(IntPtr instHdl, UInt32 hdl, out float falloff, UInt32 frame);
        internal static Boolean world_get_spot_light_falloff(UInt32 hdl, out float falloff, UInt32 frame) { return world_get_spot_light_falloff_(muffInstHdl, hdl, out falloff, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_spot_light_position")]
        private static extern Boolean world_set_spot_light_position_(IntPtr instHdl, UInt32 hdl, Vec3 pos, UInt32 frame);
        internal static Boolean world_set_spot_light_position(UInt32 hdl, Vec3 pos, UInt32 frame) { return world_set_spot_light_position_(muffInstHdl, hdl, pos, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_spot_light_intensity")]
        private static extern Boolean world_set_spot_light_intensity_(IntPtr instHdl, UInt32 hdl, Vec3 intensity, UInt32 frame);
        internal static Boolean world_set_spot_light_intensity(UInt32 hdl, Vec3 intensity, UInt32 frame) { return world_set_spot_light_intensity_(muffInstHdl, hdl, intensity, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_spot_light_direction")]
        private static extern Boolean world_set_spot_light_direction_(IntPtr instHdl, UInt32 hdl, Vec3 direction, UInt32 frame);
        internal static Boolean world_set_spot_light_direction(UInt32 hdl, Vec3 direction, UInt32 frame) { return world_set_spot_light_direction_(muffInstHdl, hdl, direction, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_spot_light_angle")]
        private static extern Boolean world_set_spot_light_angle_(IntPtr instHdl, UInt32 hdl, float angle, UInt32 frame);
        internal static Boolean world_set_spot_light_angle(UInt32 hdl, float angle, UInt32 frame) { return world_set_spot_light_angle_(muffInstHdl, hdl, angle, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_spot_light_falloff")]
        private static extern Boolean world_set_spot_light_falloff_(IntPtr instHdl, UInt32 hdl, float falloff, UInt32 frame);
        internal static Boolean world_set_spot_light_falloff(UInt32 hdl, float falloff, UInt32 frame) { return world_set_spot_light_falloff_(muffInstHdl, hdl, falloff, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_dir_light_path_segments")]
        private static extern Boolean world_get_dir_light_path_segments_(IntPtr instHdl, UInt32 hdl, out UInt32 segments);
        internal static Boolean world_get_dir_light_path_segments(UInt32 hdl, out UInt32 segments) { return world_get_dir_light_path_segments_(muffInstHdl, hdl, out segments); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_dir_light_direction")]
        private static extern Boolean world_get_dir_light_direction_(IntPtr instHdl, UInt32 hdl, out Vec3 direction, UInt32 frame);
        internal static Boolean world_get_dir_light_direction(UInt32 hdl, out Vec3 direction, UInt32 frame) { return world_get_dir_light_direction_(muffInstHdl, hdl, out direction, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_dir_light_irradiance")]
        private static extern Boolean world_get_dir_light_irradiance_(IntPtr instHdl, UInt32 hdl, out Vec3 irradiance, UInt32 frame);
        internal static Boolean world_get_dir_light_irradiance(UInt32 hdl, out Vec3 irradiance, UInt32 frame) { return world_get_dir_light_irradiance_(muffInstHdl, hdl, out irradiance, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_dir_light_direction")]
        private static extern Boolean world_set_dir_light_direction_(IntPtr instHdl, UInt32 hdl, Vec3 direction, UInt32 frame);
        internal static Boolean world_set_dir_light_direction(UInt32 hdl, Vec3 direction, UInt32 frame) { return world_set_dir_light_direction_(muffInstHdl, hdl, direction, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_dir_light_irradiance")]
        private static extern Boolean world_set_dir_light_irradiance_(IntPtr instHdl, UInt32 hdl, Vec3 irradiance, UInt32 frame);
        internal static Boolean world_set_dir_light_irradiance(UInt32 hdl, Vec3 irradiance, UInt32 frame) { return world_set_dir_light_irradiance_(muffInstHdl, hdl, irradiance, frame); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_env_light_map")]
        private static extern IntPtr world_get_env_light_map_(IntPtr instHdl, UInt32 hdl);
        internal static string world_get_env_light_map(UInt32 hdl) { return StringUtil.FromNativeUTF8(world_get_env_light_map_(muffInstHdl, hdl)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_env_light_scale")]
        private static extern Boolean world_get_env_light_scale_(IntPtr instHdl, UInt32 hdl, out Vec3 color);
        internal static Boolean world_get_env_light_scale(UInt32 hdl, out Vec3 color) { return world_get_env_light_scale_(muffInstHdl, hdl, out color); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_sky_light_turbidity")]
        private static extern Boolean world_get_sky_light_turbidity_(IntPtr instHdl, UInt32 hdl, out float turbidity);
        internal static Boolean world_get_sky_light_turbidity(UInt32 hdl, out float turbidity) { return world_get_sky_light_turbidity_(muffInstHdl, hdl, out turbidity); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_sky_light_turbidity")]
        private static extern Boolean world_set_sky_light_turbidity_(IntPtr instHdl, UInt32 hdl, float turbidity);
        internal static Boolean world_set_sky_light_turbidity(UInt32 hdl, float turbidity) { return world_set_sky_light_turbidity_(muffInstHdl, hdl, turbidity); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_sky_light_albedo")]
        private static extern Boolean world_get_sky_light_albedo_(IntPtr instHdl, UInt32 hdl, out float albedo);
        internal static Boolean world_get_sky_light_albedo(UInt32 hdl, out float albedo) { return world_get_sky_light_albedo_(muffInstHdl, hdl, out albedo); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_sky_light_albedo")]
        private static extern Boolean world_set_sky_light_albedo_(IntPtr instHdl, UInt32 hdl, float albedo);
        internal static Boolean world_set_sky_light_albedo(UInt32 hdl, float albedo) { return world_set_sky_light_albedo_(muffInstHdl, hdl, albedo); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_sky_light_solar_radius")]
        private static extern Boolean world_get_sky_light_solar_radius_(IntPtr instHdl, UInt32 hdl, out float radius);
        internal static Boolean world_get_sky_light_solar_radius(UInt32 hdl, out float radius) { return world_get_sky_light_solar_radius_(muffInstHdl, hdl, out radius); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_sky_light_solar_radius")]
        private static extern Boolean world_set_sky_light_solar_radius_(IntPtr instHdl, UInt32 hdl, float radius);
        internal static Boolean world_set_sky_light_solar_radius(UInt32 hdl, float radius) { return world_set_sky_light_solar_radius_(muffInstHdl, hdl, radius); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_sky_light_sun_direction")]
        private static extern Boolean world_get_sky_light_sun_direction_(IntPtr instHdl, UInt32 hdl, out Vec3 sunDir);
        internal static Boolean world_get_sky_light_sun_direction(UInt32 hdl, out Vec3 sunDir) { return world_get_sky_light_sun_direction_(muffInstHdl, hdl, out sunDir); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_sky_light_sun_direction")]
        private static extern Boolean world_set_sky_light_sun_direction_(IntPtr instHdl, UInt32 hdl, Vec3 sunDir);
        internal static Boolean world_set_sky_light_sun_direction(UInt32 hdl, Vec3 sunDir) { return world_set_sky_light_sun_direction_(muffInstHdl, hdl, sunDir); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_env_light_color")]
        private static extern Boolean world_get_env_light_color_(IntPtr instHdl, UInt32 hdl, out Vec3 color);
        internal static Boolean world_get_env_light_color(UInt32 hdl, out Vec3 color) { return world_get_env_light_color_(muffInstHdl, hdl, out color); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_env_light_color")]
        private static extern Boolean world_set_env_light_color_(IntPtr instHdl, UInt32 hdl, Vec3 color);
        internal static Boolean world_set_env_light_color(UInt32 hdl, Vec3 color) { return world_set_env_light_color_(muffInstHdl, hdl, color); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_env_light_map")]
        private static extern Boolean world_set_env_light_map_(IntPtr instHdl, UInt32 hdl, IntPtr tex);
        internal static Boolean world_set_env_light_map(UInt32 hdl, IntPtr tex) { return world_set_env_light_map_(muffInstHdl, hdl, tex); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_set_env_light_scale")]
        private static extern Boolean world_set_env_light_scale_(IntPtr instHdl, UInt32 hdl, Vec3 color);
        internal static Boolean world_set_env_light_scale(UInt32 hdl, Vec3 color) { return world_set_env_light_scale_(muffInstHdl, hdl, color); }

        // Render interfaces
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_get_renderer_count")]
        private static extern UInt32 render_get_renderer_count_(IntPtr instHdl);
        internal static UInt32 render_get_renderer_count() { return render_get_renderer_count_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_get_renderer_variations")]
        private static extern UInt32 render_get_renderer_variations_(IntPtr instHdl, UInt32 index);
        internal static UInt32 render_get_renderer_variations(UInt32 index) { return render_get_renderer_variations_(muffInstHdl, index); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_get_renderer_name")]
        private static extern IntPtr render_get_renderer_name_(IntPtr instHdl, UInt32 index);
        internal static string render_get_renderer_name(UInt32 index) { return StringUtil.FromNativeUTF8(render_get_renderer_name_(muffInstHdl, index)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_get_renderer_short_name")]
        private static extern IntPtr render_get_renderer_short_name_(IntPtr instHdl, UInt32 index);
        internal static string render_get_renderer_short_name(UInt32 index) { return StringUtil.FromNativeUTF8(render_get_renderer_short_name_(muffInstHdl, index)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_get_renderer_devices")]
        private static extern RenderDevice render_get_renderer_devices_(IntPtr instHdl, UInt32 index, UInt32 variation);
        internal static RenderDevice render_get_renderer_devices(UInt32 index, UInt32 variation) { return render_get_renderer_devices_(muffInstHdl, index, variation); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_enable_renderer")]
        private static extern Boolean render_enable_renderer_(IntPtr instHdl, UInt32 index, UInt32 variation);
        internal static Boolean render_enable_renderer(UInt32 index, UInt32 variation) { return render_enable_renderer_(muffInstHdl, index, variation); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_iterate")]
        private static extern Boolean render_iterate_(IntPtr instHdl, out ProcessTime iterateTime, out ProcessTime preTime, out ProcessTime postTime);
        internal static Boolean render_iterate(out ProcessTime iterateTime, out ProcessTime preTime, out ProcessTime postTime) { return render_iterate_(muffInstHdl, out iterateTime, out preTime, out postTime); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_get_current_iteration")]
        private static extern UInt32 render_get_current_iteration_(IntPtr instHdl);
        internal static UInt32 render_get_current_iteration() { return render_get_current_iteration_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_reset")]
        private static extern Boolean render_reset_(IntPtr instHdl);
        internal static Boolean render_reset() { return render_reset_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_save_screenshot")]
        private static extern Boolean render_save_screenshot_(IntPtr instHdl, IntPtr filename, IntPtr targetName, Boolean variance);
        internal static Boolean render_save_screenshot(string filename, string targetName, Boolean variance) {
            return render_save_screenshot_(muffInstHdl, StringUtil.ToNativeUtf8(filename), StringUtil.ToNativeUtf8(targetName), variance);
        }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_save_denoised_radiance")]
        private static extern Boolean render_save_denoised_radiance_(IntPtr instHdl, IntPtr filename);
        internal static Boolean render_save_denoised_radiance(string filename) { return render_save_denoised_radiance_(muffInstHdl, StringUtil.ToNativeUtf8(filename)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_get_render_target_count")]
        private static extern UInt32 render_get_render_target_count_(IntPtr instHdl);
        internal static UInt32 render_get_render_target_count() { return render_get_render_target_count_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_get_render_target_name")]
        private static extern IntPtr render_get_render_target_name_(IntPtr instHdl, UInt32 index);
        internal static string render_get_render_target_name(UInt32 index) { return StringUtil.FromNativeUTF8(render_get_render_target_name_(muffInstHdl, index)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_enable_render_target")]
        private static extern Boolean render_enable_render_target_(IntPtr instHdl, IntPtr target, Boolean variance);
        internal static Boolean render_enable_render_target(string target, Boolean variance) { return render_enable_render_target_(muffInstHdl, StringUtil.ToNativeUtf8(target), variance); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_disable_render_target")]
        private static extern Boolean render_disable_render_target_(IntPtr instHdl, IntPtr target, Boolean variance);
        internal static Boolean render_disable_render_target(string target, Boolean variance) { return render_disable_render_target_(muffInstHdl, StringUtil.ToNativeUtf8(target), variance); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_is_render_target_enabled")]
        private static extern Boolean render_is_render_target_enabled_(IntPtr instHdl, IntPtr name, Boolean variance);
        internal static Boolean render_is_render_target_enabled(string name, Boolean variance) { return render_is_render_target_enabled_(muffInstHdl, StringUtil.ToNativeUtf8(name), variance); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_is_render_target_required")]
        private static extern Boolean render_is_render_target_required_(IntPtr instHdl, IntPtr name, Boolean variance);
        internal static Boolean render_is_render_target_required(string name, Boolean variance) { return render_is_render_target_required_(muffInstHdl, StringUtil.ToNativeUtf8(name), variance); }

        // Renderer parameter interface
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_num_parameters")]
        private static extern UInt32 renderer_get_num_parameters_(IntPtr instHdl);
        internal static UInt32 renderer_get_num_parameters() { return renderer_get_num_parameters_(muffInstHdl); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_desc")]
        private static extern IntPtr renderer_get_parameter_desc_(IntPtr instHdl, UInt32 idx, out ParameterType type);
        internal static string renderer_get_parameter_desc(UInt32 idx, out ParameterType type) { return StringUtil.FromNativeUTF8(renderer_get_parameter_desc_(muffInstHdl, idx, out type)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_set_parameter_int")]
        private static extern Boolean renderer_set_parameter_int_(IntPtr instHdl, IntPtr name, Int32 value);
        internal static Boolean renderer_set_parameter_int(string name, Int32 value) { return renderer_set_parameter_int_(muffInstHdl, StringUtil.ToNativeUtf8(name), value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_int")]
        private static extern Boolean renderer_get_parameter_int_(IntPtr instHdl, IntPtr name, out Int32 value);
        internal static Boolean renderer_get_parameter_int(string name, out Int32 value) { return renderer_get_parameter_int_(muffInstHdl, StringUtil.ToNativeUtf8(name), out value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_set_parameter_float")]
        private static extern Boolean renderer_set_parameter_float_(IntPtr instHdl, IntPtr name, float value);
        internal static Boolean renderer_set_parameter_float(string name, float value) { return renderer_set_parameter_float_(muffInstHdl, StringUtil.ToNativeUtf8(name), value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_float")]
        private static extern Boolean renderer_get_parameter_float_(IntPtr instHdl, IntPtr name, out float value);
        internal static Boolean renderer_get_parameter_float(string name, out float value) { return renderer_get_parameter_float_(muffInstHdl, StringUtil.ToNativeUtf8(name), out value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_set_parameter_bool")]
        private static extern Boolean renderer_set_parameter_bool_(IntPtr instHdl, IntPtr name, Boolean value);
        internal static Boolean renderer_set_parameter_bool(string name, Boolean value) { return renderer_set_parameter_bool_(muffInstHdl, StringUtil.ToNativeUtf8(name), value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_bool")]
        private static extern Boolean renderer_get_parameter_bool_(IntPtr instHdl, IntPtr name, out UInt32 value);
        internal static Boolean renderer_get_parameter_bool(string name, out bool value) {
            UInt32 val;
            var res = renderer_get_parameter_bool_(muffInstHdl, StringUtil.ToNativeUtf8(name), out val);
            value = val != 0;
            return res;
        }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_set_parameter_enum")]
        private static extern Boolean renderer_set_parameter_enum_(IntPtr instHdl, IntPtr name, int value);
        internal static Boolean renderer_set_parameter_enum(string name, int value) { return renderer_set_parameter_enum_(muffInstHdl, StringUtil.ToNativeUtf8(name), value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_enum")]
        private static extern Boolean renderer_get_parameter_enum_(IntPtr instHdl, IntPtr name, out int value);
        internal static Boolean renderer_get_parameter_enum(string name, out int value) { return renderer_get_parameter_enum_(muffInstHdl, StringUtil.ToNativeUtf8(name), out value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_enum_count")]
        private static extern Boolean renderer_get_parameter_enum_count_(IntPtr instHdl, IntPtr param, out UInt32 count);
        internal static Boolean renderer_get_parameter_enum_count(string param, out UInt32 count) { return renderer_get_parameter_enum_count_(muffInstHdl, StringUtil.ToNativeUtf8(param), out count); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_enum_value_from_index")]
        private static extern Boolean renderer_get_parameter_enum_value_from_index_(IntPtr instHdl, IntPtr param, UInt32 index, out int value);
        internal static Boolean renderer_get_parameter_enum_value_from_index(string param, UInt32 index, out int value) { return renderer_get_parameter_enum_value_from_index_(muffInstHdl, StringUtil.ToNativeUtf8(param), index, out value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_enum_value_from_name")]
        private static extern Boolean renderer_get_parameter_enum_value_from_name_(IntPtr instHdl, IntPtr param, IntPtr valueName, out int value);
        internal static Boolean renderer_get_parameter_enum_value_from_name(string param, string valueName, out int value) {
            return renderer_get_parameter_enum_value_from_name_(muffInstHdl, StringUtil.ToNativeUtf8(param), StringUtil.ToNativeUtf8(valueName), out value);
        }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_enum_index_from_value")]
        private static extern Boolean renderer_get_parameter_enum_index_from_value_(IntPtr instHdl, IntPtr param, int value, out UInt32 index);
        internal static Boolean renderer_get_parameter_enum_index_from_value(string param, int value, out UInt32 index) { return renderer_get_parameter_enum_index_from_value_(muffInstHdl, StringUtil.ToNativeUtf8(param), value, out index); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_enum_name")]
        private static extern Boolean renderer_get_parameter_enum_name_(IntPtr instHdl, IntPtr param, int value, out IntPtr name);
        internal static Boolean renderer_get_parameter_enum_name(string param, int value, out string name) {
            IntPtr namePtr;
            Boolean res = renderer_get_parameter_enum_name_(muffInstHdl, StringUtil.ToNativeUtf8(param), value, out namePtr);
            name = StringUtil.FromNativeUTF8(namePtr);
            return res;
        }


        // Profiling interface
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_enable")]
        internal static extern void profiling_enable();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_disable")]
        internal static extern void profiling_disable();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_set_level")]
        internal static extern Boolean profiling_set_level(ProfilingLevel level);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_save_current_state")]
        internal static extern Boolean profiling_save_current_state(IntPtr path);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_save_snapshots")]
        internal static extern Boolean profiling_save_snapshots(IntPtr path);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_save_total_and_snapshots")]
        internal static extern Boolean profiling_save_total_and_snapshots(IntPtr path);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_current_state")]
        private static extern IntPtr profiling_get_current_state_();
        internal static string profiling_get_current_state() { return StringUtil.FromNativeUTF8(profiling_get_current_state_()); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_snapshots")]
        private static extern IntPtr profiling_get_snapshots_();
        internal static string profiling_get_snapshots() { return StringUtil.FromNativeUTF8(profiling_get_snapshots_()); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_total")]
        private static extern IntPtr profiling_get_total_();
        internal static string profiling_get_total() { return StringUtil.FromNativeUTF8(profiling_get_total_()); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_total_and_snapshots")]
        private static extern IntPtr profiling_get_total_and_snapshots_();
        internal static string profiling_get_total_and_snapshots() { return StringUtil.FromNativeUTF8(profiling_get_total_and_snapshots_()); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_reset")]
        internal static extern void profiling_reset();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_total_cpu_memory")]
        internal static extern ulong profiling_get_total_cpu_memory();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_free_cpu_memory")]
        internal static extern ulong profiling_get_free_cpu_memory();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_used_cpu_memory")]
        internal static extern ulong profiling_get_used_cpu_memory();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_total_gpu_memory")]
        internal static extern ulong profiling_get_total_gpu_memory();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_free_gpu_memory")]
        internal static extern ulong profiling_get_free_gpu_memory();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_used_gpu_memory")]
        internal static extern ulong profiling_get_used_gpu_memory();
    }
}
