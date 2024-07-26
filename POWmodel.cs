using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;

namespace AICalculator
{
    public class POWModelInput
    {
        [LoadColumn(0)]
        public float Value1 { get; set; }

        [LoadColumn(1)]
        public float Value2 { get; set; }

        [LoadColumn(2)]
        public float Result { get; set; }
    }

    public class POWModelOutput
    {
        [ColumnName("Score")]
        public float Result { get; set; }
    }

    public partial class POWmodel
    {
        public static void Train(string outputModelPath, string inputDataFilePath)
        {
            var mlContext = new MLContext();

            string inputDataFullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, inputDataFilePath);
            string outputModelFullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, outputModelPath);

            var data = mlContext.Data.LoadFromTextFile<POWModelInput>(
                path: inputDataFullPath,
                separatorChar: ',',
                hasHeader: true
            );

            var pipeline = mlContext.Transforms.ReplaceMissingValues(new[] { new InputOutputColumnPair(@"Value1", @"Value1"), new InputOutputColumnPair(@"Value2", @"Value2") })
                                     .Append(mlContext.Transforms.Concatenate(@"Features", new[] { @"Value1", @"Value2" }))
                                     .Append(mlContext.Regression.Trainers.FastTree(new FastTreeRegressionTrainer.Options() { NumberOfLeaves = 329, MinimumExampleCountPerLeaf = 2, NumberOfTrees = 1069, MaximumBinCountPerFeature = 304, FeatureFraction = 0.933689231422323, LearningRate = 0.0686028658549272, LabelColumnName = @"Result", FeatureColumnName = @"Features", DiskTranspose = false }));
            var model = pipeline.Fit(data);

            mlContext.Model.Save(model, data.Schema, outputModelFullPath);

            System.Diagnostics.Debug.WriteLine($"Model saved to {outputModelFullPath}");
        }
    }
}
