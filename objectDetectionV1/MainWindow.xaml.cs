using System.Windows;
using System.Windows.Media.Imaging;
using OnnxRuntime.ResNet.Template;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using System.IO;
using System.Drawing;
using objectDetectionV1Library;
namespace objectDetectionV1
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private static ConfigureRuntimeSettings rtdetr_r18vd_dec3_6x_coco_from_paddle_optimized_fp16 = new ConfigureRuntimeSettings(
            @"C:\Git\ml\models\OnnxModels\COCO\rtdetr_r18vd_dec3_6x_coco_from_paddle_optimized_fp16.onnx",
            640,
            640,
            "FP16"
        );
        private static ConfigureRuntimeSettings rtdetr_r50vd_6x_coco_from_paddle = new ConfigureRuntimeSettings(
            @"C:\Git\ml\models\OnnxModels\COCO\rtdetr_r50vd_6x_coco_from_paddle.onnx",
            640,
            640,
            "FP32"
        );
        private static ConfigureRuntimeSettings rtdetr_r101vd_2x_coco_objects365_from_paddle = new ConfigureRuntimeSettings(
           @"C:\Git\ml\models\OnnxModels\COCO\rtdetr_r101vd_2x_coco_objects365_from_paddle.onnx",
           640,
           640,
           "FP32"
        );
        private static ConfigureRuntimeSettings rtdetr_r101vd_2x_coco_objects365_from_paddle_Video = new ConfigureRuntimeSettings(
           @"I:\RT-DETR\rtdetr_pytorch\model.onnx",
           @"path/to/video1.ts",
           640,
           640,
           "FP32"
        );
        private static ConfigureRuntimeSettings rtdetr_r18vd_2x_coco_objects365_from_paddle_Video = new ConfigureRuntimeSettings(
           @"I:\RT-DETR\Models\18vd_cp56.onnx",
           @"path/to/video2.ts",
           640,
           640,
           "FP32"
        );
        public static MainWindow current;
        public MainWindow()
        {
            current = this;
            InitializeComponent();
        }
        private void Start_Click(object sender, RoutedEventArgs e)
        {
            ManageModelRuntime.RunInference(rtdetr_r18vd_2x_coco_objects365_from_paddle_Video);
        }
        public static class MainWindowLibrary
        {
            public static void DisplayBitmapToImageControl(Bitmap bitmap)
            {
                current.Dispatcher.Invoke(() => {
                    BitmapImage bitmapImage = ConverBitmapToBitmapImage(bitmap);
                    current.ImageControl.Source = bitmapImage;
                });
            }
            private static BitmapImage ConverBitmapToBitmapImage(Bitmap bitmap)
            {
                using (MemoryStream memoryStream = new MemoryStream())
                {
                    bitmap.Save(memoryStream, System.Drawing.Imaging.ImageFormat.Bmp);
                    memoryStream.Position = 0;
                    BitmapImage bitmapImage = new BitmapImage();
                    bitmapImage.BeginInit();
                    bitmapImage.StreamSource = memoryStream;
                    bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                    bitmapImage.EndInit();
                    return bitmapImage;
                }
            }
        }
    }
}