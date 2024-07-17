using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace AICalculator
{
    public class ADDModelInput
    {
        [LoadColumn(0)]
        public float Value1 { get; set; }

        [LoadColumn(1)]
        public float Value2 { get; set; }

        [LoadColumn(2)]
        public float Result { get; set; }
    }

    public class ADDModelOutput
    {
        [ColumnName("Score")]
        public float Result { get; set; }
    }

    public partial class ADDmodel
    {
        public static void Train(string outputModelPath, string inputDataFilePath)
        {
            var mlContext = new MLContext();

            string inputDataFullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, inputDataFilePath);
            string outputModelFullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, outputModelPath);

            var data = mlContext.Data.LoadFromTextFile<ADDModelInput>(
                path: inputDataFullPath,
                separatorChar: ',',
                hasHeader: true
            );

            var pipeline = mlContext.Transforms.ReplaceMissingValues(new[] { new InputOutputColumnPair("Value1", "Value1"), new InputOutputColumnPair("Value2", "Value2") })
                .Append(mlContext.Transforms.Concatenate("Features", new[] { "Value1", "Value2" }))
                .Append(mlContext.Regression.Trainers.Sdca(new SdcaRegressionTrainer.Options() { 
                    L1Regularization = 0.2849315F, 
                    L2Regularization = 0.9969157F, 
                    LabelColumnName = "Result", 
                    FeatureColumnName = "Features" }));
            var model = pipeline.Fit(data);

            mlContext.Model.Save(model, data.Schema, outputModelFullPath);

            System.Diagnostics.Debug.WriteLine($"Model saved to {outputModelFullPath}");
        }
    }
}
