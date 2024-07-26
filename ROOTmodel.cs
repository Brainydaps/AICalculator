using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;

namespace AICalculator
{
    public class ROOTModelInput
    {
        [LoadColumn(0)]
        public float Value1 { get; set; }

        [LoadColumn(1)]
        public float Value2 { get; set; }

        [LoadColumn(2)]
        public float Result { get; set; }
    }

    public class ROOTModelOutput
    {
        [ColumnName("Score")]
        public float Result { get; set; }
    }

    public partial class ROOTmodel
    {
        public static void Train(string outputModelPath, string inputDataFilePath)
        {
            var mlContext = new MLContext();

            string inputDataFullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, inputDataFilePath);
            string outputModelFullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, outputModelPath);

            var data = mlContext.Data.LoadFromTextFile<ROOTModelInput>(
                path: inputDataFullPath,
                separatorChar: ',',
                hasHeader: true
            );

            var pipeline = mlContext.Transforms.ReplaceMissingValues(new[] { new InputOutputColumnPair(@"Value1", @"Value1"), new InputOutputColumnPair(@"Value2", @"Value2") })
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new[] { @"Value1", @"Value2" }))
                                    .Append(mlContext.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options() { 
                                        NumberOfLeaves = 1429, 
                                        NumberOfIterations = 400, 
                                        MinimumExampleCountPerLeaf = 20, 
                                        LearningRate = 0.543228216647901, 
                                        LabelColumnName = @"Result", 
                                        FeatureColumnName = @"Features", 
                                        Booster = new GradientBooster.Options() { 
                                            SubsampleFraction = 0.597069228920116, 
                                            FeatureFraction = 0.924946100334807, 
                                            L1Regularization = 2.0984862382559E-10, 
                                            L2Regularization = 0.999999776672986 }, 
                                        MaximumBinCountPerFeature = 296 }));
            var model = pipeline.Fit(data);

            mlContext.Model.Save(model, data.Schema, outputModelFullPath);

            System.Diagnostics.Debug.WriteLine($"Model saved to {outputModelFullPath}");
        }
    }
}
