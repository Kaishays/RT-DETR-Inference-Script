using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace objectDetectionV1Library
{
    public class ConfigureRuntimeSettings
    {
        public ConfigureRuntimeSettings(string pathToOnnxModel, int inputSubjectPixelHeight, int inputSubjectPixelWidth, string modelType)
        {
            this.pathToOnnxModel = pathToOnnxModel;
            this.inputSubjectPixelHeight = inputSubjectPixelHeight;
            this.inputSubjectPixelWidth = inputSubjectPixelWidth;
            this.modelType = modelType;
        }
        public ConfigureRuntimeSettings(string pathToOnnxModel, string pathToVideoFile, int inputSubjectPixelHeight, int inputSubjectPixelWidth, string modelType)
        {
            this.pathToOnnxModel = pathToOnnxModel;
            this.inputSubjectPixelHeight = inputSubjectPixelHeight;
            this.inputSubjectPixelWidth = inputSubjectPixelWidth;
            this.filePath = pathToVideoFile;
            this.modelType = modelType;
        }

        public int batchSize { get; set; } = 1;
        public double confidenceThreshold { get; set; } = 0.6;
        public string pathToOnnxModel {  get; set; }
        public int inputSubjectPixelHeight { get; set; }
        public int inputSubjectPixelWidth { get; set; }
        public string filePath { get; set; }
        public string modelType { get; set; }
    }
}
