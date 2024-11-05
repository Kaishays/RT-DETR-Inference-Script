using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Text;
using static System.Formats.Asn1.AsnWriter;
using System.Windows;
using System.Xml.Linq;
using System.Windows.Media.Media3D;
namespace OnnxRuntime.ResNet.Template.utils
{
    public static class ModelHelper
    {
        private static InferenceSession session;
        private static List<NamedOnnxValue> inputsParametersForOnnxModel;
        public static void InitializeSession(string modelFilePath)
        {
            if (session == null)
            {
                using var gpuSessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(0);
                session = new InferenceSession(modelFilePath, gpuSessionOptions);
            }
        }
        public static IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run_FP16_ModelInference(Tensor<Float16> inputFrameTensor, Tensor<long> origTargetSizes)
        {
            inputsParametersForOnnxModel = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor<Float16>("images", inputFrameTensor),
                NamedOnnxValue.CreateFromTensor<long>("orig_target_sizes", origTargetSizes)
            };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputsParametersForOnnxModel);
            return results;
        }
        public static IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run_FP32_ModelInference(Tensor<float> inputFrameTensor, Tensor<long> origTargetSizes)
        {
            inputsParametersForOnnxModel = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor<float>("images", inputFrameTensor),
                NamedOnnxValue.CreateFromTensor<long>("orig_target_sizes", origTargetSizes)
            };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputsParametersForOnnxModel);
            return results;
        }
    }
}