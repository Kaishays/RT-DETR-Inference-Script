using AForge.Video;
using AForge.Video.DirectShow;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using OnnxRuntime.ResNet.Template.utils;
using System.Diagnostics;
using System.Drawing;
using objectDetectionV1Library;
using OpenCvSharp.Extensions;
using OpenCvSharp;
using System.Windows.Media.Media3D;
using System.Windows.Media.Imaging;
namespace objectDetectionV1
{
    public static class ManageModelRuntime
    {
        private static VideoCaptureDevice videoSource { get; set; }
        private static Thread InferenceWorkerThread { get; set; }
        public static ConfigureRuntimeSettings configureRuntimeSettings { get; set; }
        private static int FrameCount { get; set; } = 0;
        private static Tensor<long> originTargetSize { get; set; }
        public static void RunInference(ConfigureRuntimeSettings _configureRuntimeSettings)
        {
            configureRuntimeSettings = _configureRuntimeSettings;

            ModelHelper.InitializeSession(configureRuntimeSettings.pathToOnnxModel);
            originTargetSize = ProcessImage.PreProcess.CreateOrginTargetSizeTensor(configureRuntimeSettings.inputSubjectPixelHeight, configureRuntimeSettings.inputSubjectPixelWidth);
            if (configureRuntimeSettings.filePath == null)
            {
                ManageWebcamInference.StartWebCamStream();
            }
            else
            {
                ManageFileInference.StartFileStream();
            }
        }
        private static void HandleNewFrame(Bitmap bitmap)
        {
            bitmap = ProcessImage.PreProcess.ConvertBitmapToTargetSize(bitmap, configureRuntimeSettings.inputSubjectPixelWidth, configureRuntimeSettings.inputSubjectPixelHeight);
            if (configureRuntimeSettings.modelType == "FP16")
            {
                HandleFP16Frame(bitmap);
            }
            else if (configureRuntimeSettings.modelType == "FP32")
            {
                HandleFP32Frame(bitmap);
            }
            FrameCount++;
            Debug.WriteLine("frame" + FrameCount);
        }
        private static void HandleFP16Frame(Bitmap bitmap)
        {
            Tensor<Float16> inputFrameTensor = ProcessImage.PreProcess.ImageToTensorFP16(bitmap);
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = ModelHelper.Run_FP16_ModelInference(inputFrameTensor, originTargetSize);
            Bitmap imageWithInference = ProcessImage.PostProcess.FP16.ProcessOnnxModelOutput(results, bitmap, configureRuntimeSettings.confidenceThreshold);
            MainWindow.MainWindowLibrary.DisplayBitmapToImageControl(imageWithInference);
        }
        private static void HandleFP32Frame(Bitmap bitmap)
        {
            Tensor<float> inputFrameTensor = ProcessImage.PreProcess.ImageToTensorFP32(bitmap);
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = ModelHelper.Run_FP32_ModelInference(inputFrameTensor, originTargetSize);
            Bitmap imageWithInference = ProcessImage.PostProcess.FP32.ProcessOnnxModelOutput(results, bitmap, configureRuntimeSettings.confidenceThreshold);
            MainWindow.MainWindowLibrary.DisplayBitmapToImageControl(imageWithInference);
        }
        public static class ManageWebcamInference
        {
            public static void StartWebCamStream()
            {
                InferenceWorkerThread = new Thread(() =>
                {
                    var videoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
                    if (videoDevices.Count == 0)
                    {
                        throw new Exception("No video capture device found.");
                    }
                    videoSource = new VideoCaptureDevice(videoDevices[0].MonikerString);
                    videoSource.NewFrame += new NewFrameEventHandler(OnNewFrame);
                    videoSource.Start();
                });
                InferenceWorkerThread.Start();
            }
            private static void OnNewFrame(object sender, NewFrameEventArgs eventArgs)
            {
                Bitmap bitmap = eventArgs.Frame;
                HandleNewFrame(bitmap);
            }
        }
        public static class ManageFileInference
        {
            public static void StartFileStream()
            {
                InferenceWorkerThread = new Thread(() =>
                {
                    using (var videoCapture = new VideoCapture(configureRuntimeSettings.filePath))
                    {
                        Mat frame = new Mat();

                        while (videoCapture.Read(frame))
                        {
                            if (frame.Empty())
                                break;

                            Bitmap bitmap = BitmapConverter.ToBitmap(frame);
                            HandleNewFrame(bitmap);
                        }
                    }
                });
                InferenceWorkerThread.Start();
            }
        }
    }
}