using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;

namespace AICalculator
{
    public class COSModelInput
    {
        [LoadColumn(0)]
        public float Value { get; set; }

        [LoadColumn(1)]
        public float Result { get; set; }
    }

    public class COSModelOutput
    {
        [ColumnName("Score")]
        public float Result { get; set; }
    }

    public partial class COSmodel
    {
        public static void Train(string outputModelPath, string inputDataFilePath)
        {
            var mlContext = new MLContext();

            string inputDataFullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, inputDataFilePath);
            string outputModelFullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, outputModelPath);

            var data = mlContext.Data.LoadFromTextFile<COSModelInput>(
                path: inputDataFullPath,
                separatorChar: ',',
                hasHeader: true
            );

            var pipeline = mlContext.Transforms.ReplaceMissingValues(@"Value", @"Value")
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new[] { @"Value" }))
                                    .Append(mlContext.Regression.Trainers.FastTree(new FastTreeRegressionTrainer.Options() { NumberOfLeaves = 19, MinimumExampleCountPerLeaf = 2, NumberOfTrees = 547, MaximumBinCountPerFeature = 1023, FeatureFraction = 0.99999999, LearningRate = 0.0109346776966098, LabelColumnName = @"Result", FeatureColumnName = @"Features", DiskTranspose = false }));


            var model = pipeline.Fit(data);

            mlContext.Model.Save(model, data.Schema, outputModelFullPath);

            System.Diagnostics.Debug.WriteLine($"Model saved to {outputModelFullPath}");
        }
    }
}
