using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace AICalculator
{
    public class SQRTModelInput
    {
        [LoadColumn(0)]
        public float Value { get; set; }

        [LoadColumn(1)]
        public float Result { get; set; }
    }

    public class SQRTModelOutput
    {
        [ColumnName("Score")]
        public float Result { get; set; }
    }

    public partial class SQRTmodel
    {
        public static void Train(string outputModelPath, string inputDataFilePath)
        {
            var mlContext = new MLContext();

            string inputDataFullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, inputDataFilePath);
            string outputModelFullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, outputModelPath);

            var data = mlContext.Data.LoadFromTextFile<SQRTModelInput>(
                path: inputDataFullPath,
                separatorChar: ',',
                hasHeader: true
            );

            var pipeline = mlContext.Transforms.ReplaceMissingValues(@"Value", @"Value")
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new[] { @"Value" }))
                                    .Append(mlContext.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options() { NumberOfLeaves = 4, NumberOfIterations = 9010, MinimumExampleCountPerLeaf = 20, LearningRate = 0.13032366932211, LabelColumnName = @"Result", FeatureColumnName = @"Features", Booster = new GradientBooster.Options() { SubsampleFraction = 0.0143717246714535, FeatureFraction = 0.99999999, L1Regularization = 2.44447777021566E-10, L2Regularization = 0.999999776672986 }, MaximumBinCountPerFeature = 256 }));


            var model = pipeline.Fit(data);

            mlContext.Model.Save(model, data.Schema, outputModelFullPath);

            System.Diagnostics.Debug.WriteLine($"Model saved to {outputModelFullPath}");
        }
    }
}
