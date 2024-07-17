using System;
using System.IO;
using Microsoft.ML;

namespace AICalculator
{
    public class ModelCall
    {
        private readonly MLContext mlContext;
        private PredictionEngine<ADDModelInput, ADDModelOutput> addPredictionEngine;
        private PredictionEngine<SUBModelInput, SUBModelOutput> subPredictionEngine;
        private PredictionEngine<MULModelInput, MULModelOutput> mulPredictionEngine;
        private PredictionEngine<DIVModelInput, DIVModelOutput> divPredictionEngine;

        public ModelCall()
        {
            mlContext = new MLContext();

            // Train models before loading them
            TrainModels();

            // Load models after training
            addPredictionEngine = LoadModel<ADDModelInput, ADDModelOutput>("ADDmodel.zip");
            subPredictionEngine = LoadModel<SUBModelInput, SUBModelOutput>("SUBmodel.zip");
            mulPredictionEngine = LoadModel<MULModelInput, MULModelOutput>("MULmodel.zip");
            divPredictionEngine = LoadModel<DIVModelInput, DIVModelOutput>("DIVmodel.zip");
        }

        private void TrainModels()
        {
            try
            {
                // Train and save the ADD model
                ADDmodel.Train("ADDmodel.zip", "training_dataADD.csv");
                System.Diagnostics.Debug.WriteLine("ADD model trained and saved successfully.");

                // Train and save the SUB model
                SUBmodel.Train("SUBmodel.zip", "training_dataSUB.csv");
                System.Diagnostics.Debug.WriteLine("SUB model trained and saved successfully.");

                // Train and save the MUL model
                MULmodel.Train("MULmodel.zip", "training_dataMUL.csv");
                System.Diagnostics.Debug.WriteLine("MUL model trained and saved successfully.");

                // Train and save the DIV model
                DIVmodel.Train("DIVmodel.zip", "training_dataDIV.csv");
                System.Diagnostics.Debug.WriteLine("DIV model trained and saved successfully.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error training models: {ex.Message}");
            }
        }

        private PredictionEngine<TInput, TOutput> LoadModel<TInput, TOutput>(string modelPath)
            where TInput : class
            where TOutput : class, new()
        {
            string fullModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, modelPath);
            ITransformer model = mlContext.Model.Load(fullModelPath, out _);
            return mlContext.Model.CreatePredictionEngine<TInput, TOutput>(model);
        }

        public float Predict(string operation, float number1, float number2)
        {
            switch (operation)
            {
                case "+":
                    var addInput = new ADDModelInput { Value1 = number1, Value2 = number2 };
                    return addPredictionEngine.Predict(addInput).Result;

                case "−":
                    var subInput = new SUBModelInput { Value1 = number1, Value2 = number2 };
                    return subPredictionEngine.Predict(subInput).Result;

                case "×":
                    var mulInput = new MULModelInput { Value1 = number1, Value2 = number2 };
                    return mulPredictionEngine.Predict(mulInput).Result;

                case "÷":
                    var divInput = new DIVModelInput { Value1 = number1, Value2 = number2 };
                    return divPredictionEngine.Predict(divInput).Result;

                default:
                    throw new InvalidOperationException("Unsupported operation");
            }
        }
    }
}
