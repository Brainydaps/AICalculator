using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace AICalculator
{
    public class DIVModelInput
    {
        [LoadColumn(0)]
        public float Value1 { get; set; }

        [LoadColumn(1)]
        public float Value2 { get; set; }

        [LoadColumn(2)]
        public float Result { get; set; }
    }

    public class DIVModelOutput
    {
        [ColumnName("Score")]
        public float Result { get; set; }
    }

    public partial class DIVmodel
    {
        public const string RetrainFilePath = "training_dataDIV.csv";
        public const char RetrainSeparatorChar = ',';
        public const bool RetrainHasHeader = true;
        public const bool RetrainAllowQuoting = false;

        public static void Train(string outputModelPath = "DIVmodel.zip", string inputDataFilePath = RetrainFilePath, char separatorChar = RetrainSeparatorChar, bool hasHeader = RetrainHasHeader, bool allowQuoting = RetrainAllowQuoting)
        {
            var mlContext = new MLContext();

            var data = LoadIDataViewFromFile(mlContext, inputDataFilePath, separatorChar, hasHeader, allowQuoting);
            var model = RetrainModel(mlContext, data);
            SaveModel(mlContext, model, data, outputModelPath);
        }

        public static IDataView LoadIDataViewFromFile(MLContext mlContext, string inputDataFilePath, char separatorChar, bool hasHeader, bool allowQuoting)
        {
            string fullInputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, inputDataFilePath);
            System.Diagnostics.Debug.WriteLine($"Loading data from {fullInputPath}");

            return mlContext.Data.LoadFromTextFile<DIVModelInput>(fullInputPath, separatorChar, hasHeader, allowQuoting: allowQuoting);
        }

        public static void SaveModel(MLContext mlContext, ITransformer model, IDataView data, string modelSavePath)
        {
            string fullModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, modelSavePath);
            System.Diagnostics.Debug.WriteLine($"Saving model to {fullModelPath}");

            DataViewSchema dataViewSchema = data.Schema;

            using (var fs = File.Create(fullModelPath))
            {
                mlContext.Model.Save(model, dataViewSchema, fs);
            }
        }

        public static ITransformer RetrainModel(MLContext mlContext, IDataView trainData)
        {
            var pipeline = BuildPipeline(mlContext);
            var model = pipeline.Fit(trainData);
            return model;
        }

        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {
            var pipeline = mlContext.Transforms.ReplaceMissingValues(new[] { new InputOutputColumnPair("Value1", "Value1"), new InputOutputColumnPair("Value2", "Value2") })
                                    .Append(mlContext.Transforms.Concatenate("Features", new[] { "Value1", "Value2" }))
                     
                                    .Append(mlContext.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options() { 
                                        NumberOfLeaves = 4, 
                                        NumberOfIterations = 2981, 
                                        MinimumExampleCountPerLeaf = 20, 
                                        LearningRate = 0.999999776672986, 
                                        LabelColumnName = @"Result", 
                                        FeatureColumnName = @"Features", 
                                        Booster = new GradientBooster.Options() { 
                                            SubsampleFraction = 0.977159784466191, 
                                            FeatureFraction = 0.955921410836799, 
                                            L1Regularization = 1.08597194480575E-09, 
                                            L2Regularization = 0.180834720077925 }, 
                                        MaximumBinCountPerFeature = 196 }));


            return pipeline;
        }
    }
}
