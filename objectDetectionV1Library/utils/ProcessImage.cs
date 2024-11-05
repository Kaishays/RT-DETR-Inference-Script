using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxRuntime.ResNet.Template;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
public static class ProcessImage
{
    public static class PreProcess
    {
        private static int batchSize = 1;
        private static int channelSize_RGB = 3;
        public static Bitmap ConvertBitmapToTargetSize(Bitmap bitMap, int targetWidth, int targetHeight)
        {
            int originalWidth = bitMap.Width;
            int originalHeight = bitMap.Height;

            if (originalWidth > targetWidth || originalHeight > targetHeight)
            {
                bitMap = CropBitmap(bitMap, targetWidth, targetHeight);
            } else if (originalWidth < targetWidth || originalHeight < targetHeight) 
            {
                // Implement Add To Bitmap
            }
            return bitMap;
        }
        private static Bitmap CropBitmap(Bitmap originalImage, int targetWidth, int targetHeight)
        {
            int originalWidth = originalImage.Width;
            int originalHeight = originalImage.Height;

            int xDimensionStartingPoint = (originalWidth - targetWidth) / 2;
            int yDimensionStartingPoint = (originalHeight - targetHeight) / 2;

            Rectangle areaToExtract = new Rectangle(xDimensionStartingPoint, yDimensionStartingPoint, 640, 640);

            Bitmap croppedBitmap = originalImage.Clone(areaToExtract, originalImage.PixelFormat);
            return croppedBitmap;
        }
        public static byte[] BitmapToByteArray(Bitmap bitmap)
        {
            BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                                                 ImageLockMode.ReadOnly, bitmap.PixelFormat);

            int bytes = Math.Abs(bmpData.Stride) * bitmap.Height;
            byte[] pixelData = new byte[bytes];

            System.Runtime.InteropServices.Marshal.Copy(bmpData.Scan0, pixelData, 0, bytes);

            bitmap.UnlockBits(bmpData);

            return pixelData;
        }
        public static Tensor<Float16> ImageToTensorFP16(Bitmap bitmap)
        {
            int height = bitmap.Height; 
            int width = bitmap.Width;
            byte[] byteArray = BitmapToByteArray(bitmap);
            int tensorSize = batchSize * channelSize_RGB * height * width;
            Float16[] data = new Float16[tensorSize];

            int channelSize = width * height;
            int redChannel = 0;
            int greenChannel = channelSize;
            int blueChannel = channelSize * 2;

            Parallel.For(0, height, y =>
            {
                int rowStart = y * width * 3;
                int rowWidth = width * 3;

                var row = new Span<byte>(byteArray, rowStart, rowWidth);

                for (int i = 0; i < row.Length; i += 3)
                {
                    byte red = row[i];
                    byte green = row[i + 1];
                    byte blue = row[i + 2];

                    var index = i / 3 + y * width;

                    data[redChannel + index] = (Float16)(red / 255f);
                    data[greenChannel + index] = (Float16)(green / 255f);
                    data[blueChannel + index] = (Float16)(blue / 255f);
                }
            });
            return new DenseTensor<Float16>(data, new[] { batchSize, channelSize_RGB, height, width });
        }
        public static Tensor<float> ImageToTensorFP32(Bitmap bitmap)
        {
            int height = bitmap.Height;
            int width = bitmap.Width;
            byte[] pixels = BitmapToByteArray(bitmap);
            int tensorSize = 1 * 3 * height * width;

            float[] data = new float[tensorSize];

            var channelSize = width * height;
            var redChannel = 0;
            var greenChannel = channelSize;
            var blueChannel = channelSize * 2;

            Parallel.For(0, height, y =>
            {
                int rowStart = y * width * 3;
                int rowWidth = width * 3;

                var row = new Span<byte>(pixels, rowStart, rowWidth);

                for (int i = 0; i < row.Length; i += 3)
                {
                    byte red = row[i];
                    byte green = row[i + 1];
                    byte blue = row[i + 2];

                    var index = i / 3 + y * width;

                    data[redChannel + index] = (float)(red / 255f);
                    data[greenChannel + index] = (float)(green / 255f);
                    data[blueChannel + index] = (float)(blue / 255f);
                }
            });
            return new DenseTensor<float>(data, new[] { 1, 3, height, width });
        }
        public static Tensor<long> CreateOrginTargetSizeTensor(int orginHeight, int orginWidth)
        {
            Tensor<long> orig_target_sizes = new DenseTensor<long>(new[] { batchSize, 2 });
            orig_target_sizes[0, 0] = orginHeight;
            orig_target_sizes[0, 1] = orginWidth;
            return orig_target_sizes;
        }
    }
    public static class PostProcess
    {
        public static class FP16
        {
            public static Bitmap ProcessOnnxModelOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> onnxModelOutputTensors, Bitmap bitmap, double confidenseThreshold)
            {
                List<string> labels = new List<string>();
                List<Rectangle> boundingBoxes = new List<Rectangle>();
                List<float> scores = new List<float>();

                List<string> labels_MeetConfidenseThreshold = new List<string>();
                List<Rectangle> boundingBoxes_MeetConfidenseThreshold = new List<Rectangle>();
                List<float> scores_MeetConfidenseThreshold = new List<float>();

                foreach (DisposableNamedOnnxValue result in onnxModelOutputTensors)
                {
                    if (result.Name == "labels")
                    {
                        Tensor<long> labelsTensor = result.AsTensor<long>();
                        labels = HandleLabelsTensor(labelsTensor);
                    }
                    else if (result.Name == "boxes")
                    {
                        Tensor<Float16> boxesTensor = result.AsTensor<Float16>();
                        boundingBoxes = HandelBoxesTensor(boxesTensor);
                    }
                    else if (result.Name == "scores")
                    {
                        Tensor<Float16> scoresTensor = result.AsTensor<Float16>();
                        scores = HandleScoresTensor(scoresTensor);
                    }
                }

                int totalPredictions = labels.Count;
                for (int i = 0; i < totalPredictions; i++)
                {
                    if (scores[i] >= confidenseThreshold)
                    {
                        labels_MeetConfidenseThreshold.Add(labels[i]);
                        boundingBoxes_MeetConfidenseThreshold.Add(boundingBoxes[i]);
                        scores_MeetConfidenseThreshold.Add(scores[i]);
                    }
                }
                Bitmap imageWithInference = AddInferenceToBitmap(bitmap, labels_MeetConfidenseThreshold, boundingBoxes_MeetConfidenseThreshold, scores_MeetConfidenseThreshold);
                return imageWithInference;
            }
            private static List<Rectangle> HandelBoxesTensor(Tensor<Float16> boxesTensor)
            {
                int batchSize = boxesTensor.Dimensions[0];
                int numberOfPredictions = boxesTensor.Dimensions[1];
                int boundingBoxCoordinatesCount = boxesTensor.Dimensions[2];

                List<Rectangle> boundingBoxes = new List<Rectangle>();
                for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)
                {
                    for (int predictionIndex = 0; predictionIndex < numberOfPredictions; predictionIndex++)
                    {
                        float top_Left_X = (float)boxesTensor[batchIndex, predictionIndex, 0];
                        float top_Left_Y = (float)boxesTensor[batchIndex, predictionIndex, 1];
                        float bottom_Right_X = (float)boxesTensor[batchIndex, predictionIndex, 2];
                        float bottom_Right_Y = (float)boxesTensor[batchIndex, predictionIndex, 3];
                        Rectangle boundingBox = CreateRectangleFromBoxesTensor(top_Left_X, top_Left_Y, bottom_Right_X, bottom_Right_Y);
                        boundingBoxes.Add(boundingBox);
                    }
                }
                return boundingBoxes;
            }
            private static List<float> HandleScoresTensor(Tensor<Float16> scoresTensor)
            {
                int batchSize = scoresTensor.Dimensions[0];
                int numberOfScores = scoresTensor.Dimensions[1];

                List<float> scores = new List<float>();
                for (int i = 0; i < batchSize; i++)
                {
                    for (int j = 0; j < numberOfScores; j++)
                    {
                        float label = (float)scoresTensor[i, j];
                        float roundedValue = (float)Math.Round(label, 2);
                        scores.Add(roundedValue);
                    }
                }
                return scores;
            }
        }
        public static class FP32
        {
            public static Bitmap ProcessOnnxModelOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> onnxModelOutputTensors, Bitmap bitmap, double confidenseThreshold)
            {
                List<string> labels = new List<string>();
                List<Rectangle> boundingBoxes = new List<Rectangle>();
                List<float> scores = new List<float>();

                List<string> labels_MeetConfidenseThreshold = new List<string>();
                List<Rectangle> boundingBoxes_MeetConfidenseThreshold = new List<Rectangle>();
                List<float> scores_MeetConfidenseThreshold = new List<float>();

                foreach (DisposableNamedOnnxValue result in onnxModelOutputTensors)
                {
                    if (result.Name == "labels")
                    {
                        Tensor<long> labelsTensor = result.AsTensor<long>();
                        labels = HandleLabelsTensor(labelsTensor);
                    }
                    else if (result.Name == "boxes")
                    {
                        Tensor<float> boxesTensor = result.AsTensor<float>();
                        boundingBoxes = HandelBoxesTensor(boxesTensor);
                    }
                    else if (result.Name == "scores")
                    {
                        Tensor<float> scoresTensor = result.AsTensor<float>();
                        scores = HandleScoresTensor(scoresTensor);
                    }
                }

                int totalPredictions = labels.Count;
                for (int i = 0; i < totalPredictions; i++)
                {
                    if (scores[i] >= confidenseThreshold)
                    {
                        labels_MeetConfidenseThreshold.Add(labels[i]);
                        boundingBoxes_MeetConfidenseThreshold.Add(boundingBoxes[i]);
                        scores_MeetConfidenseThreshold.Add(scores[i]);
                    }
                }
                Bitmap imageWithInference = AddInferenceToBitmap(bitmap, labels_MeetConfidenseThreshold, boundingBoxes_MeetConfidenseThreshold, scores_MeetConfidenseThreshold);
                return imageWithInference;
            }
            private static List<Rectangle> HandelBoxesTensor(Tensor<float> boxesTensor)
            {
                int batchSize = boxesTensor.Dimensions[0];
                int numberOfPredictions = boxesTensor.Dimensions[1];
                int boundingBoxCoordinatesCount = boxesTensor.Dimensions[2];

                List<Rectangle> boundingBoxes = new List<Rectangle>();
                for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)
                {
                    for (int predictionIndex = 0; predictionIndex < numberOfPredictions; predictionIndex++)
                    {
                        float top_Left_X = (float)boxesTensor[batchIndex, predictionIndex, 0];
                        float top_Left_Y = (float)boxesTensor[batchIndex, predictionIndex, 1];
                        float bottom_Right_X = (float)boxesTensor[batchIndex, predictionIndex, 2];
                        float bottom_Right_Y = (float)boxesTensor[batchIndex, predictionIndex, 3];
                        Rectangle boundingBox = CreateRectangleFromBoxesTensor(top_Left_X, top_Left_Y, bottom_Right_X, bottom_Right_Y);
                        boundingBoxes.Add(boundingBox);
                    }
                }
                return boundingBoxes;
            }
            private static List<float> HandleScoresTensor(Tensor<float> scoresTensor)
            {
                int batchSize = scoresTensor.Dimensions[0];
                int numberOfScores = scoresTensor.Dimensions[1];

                List<float> scores = new List<float>();
                for (int i = 0; i < batchSize; i++)
                {
                    for (int j = 0; j < numberOfScores; j++)
                    {
                        float label = (float)scoresTensor[i, j];
                        float roundedValue = (float)Math.Round(label, 2);
                        scores.Add(roundedValue);
                    }
                }
                return scores;
            }
        }
        private static List<string> HandleLabelsTensor(Tensor<long> labelsTensor)
        {
            int batchSize = labelsTensor.Dimensions[0];
            int numberOfPredictions = labelsTensor.Dimensions[1];

            List<string> labels = new List<string>();
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < numberOfPredictions; j++)
                {
                    long label = labelsTensor[i, j];
                    labels.Add(LabelMap.label_06_V7[label]);
                }
            }
            return labels;
        }
        private static Rectangle CreateRectangleFromBoxesTensor(float top_Left_X, float top_Left_Y, float bottom_Right_X, float bottom_Right_Y)
        {
            int top_Left_X_Int = (int)top_Left_X;
            int top_Left_Y_Int = (int)top_Left_Y;

            int width = (int)bottom_Right_X - top_Left_X_Int;
            int height = (int)bottom_Right_Y - top_Left_Y_Int;

            Rectangle boundingBox = new Rectangle(top_Left_X_Int, top_Left_Y_Int, width, height);
            return boundingBox;
        }
        public static Bitmap AddInferenceToBitmap(Bitmap bitmap, List<string> labels, List<Rectangle> boundingBoxes, List<float> scores)
        {
            using (Graphics graphics = Graphics.FromImage(bitmap))
            {
                int marginForBoundingBoxHeader = 20;
                System.Drawing.FontFamily fontFamily = new System.Drawing.FontFamily("Arial");
                Font font = new Font(fontFamily, 14);
                Brush brush = System.Drawing.Brushes.Yellow;

                for (int i = 0; i < labels.Count; i++)
                {
                    string boundingBoxHeader = labels[i] + " " + scores[i];
                    Rectangle boundingBox = boundingBoxes[i];
                    PointF labelPoint = new PointF((float)boundingBox.X, (float)boundingBox.Y - marginForBoundingBoxHeader);

                    graphics.DrawString(boundingBoxHeader, font, brush, labelPoint);
                    graphics.DrawRectangle(Pens.Red, boundingBox);
                }
            }
            return bitmap;
        }
    }
}