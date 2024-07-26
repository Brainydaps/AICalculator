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
        private PredictionEngine<SQRModelInput, SQRModelOutput> sqrPredictionEngine;
        private PredictionEngine<SQRTModelInput, SQRTModelOutput> sqrtPredictionEngine;
        private PredictionEngine<POWModelInput, POWModelOutput> powPredictionEngine;
        private PredictionEngine<ROOTModelInput, ROOTModelOutput> rootPredictionEngine;
        private PredictionEngine<COSModelInput, COSModelOutput> cosPredictionEngine;
        private PredictionEngine<SINModelInput, SINModelOutput> sinPredictionEngine;
        private PredictionEngine<TANModelInput, TANModelOutput> tanPredictionEngine;

        public ModelCall()
        {
            mlContext = new MLContext();

            // Train models if needed and then load them
            TrainModels();

            // Load models after training
            addPredictionEngine = LoadModel<ADDModelInput, ADDModelOutput>("ADDmodel.zip");
            subPredictionEngine = LoadModel<SUBModelInput, SUBModelOutput>("SUBmodel.zip");
            mulPredictionEngine = LoadModel<MULModelInput, MULModelOutput>("MULmodel.zip");
            divPredictionEngine = LoadModel<DIVModelInput, DIVModelOutput>("DIVmodel.zip");
            sqrPredictionEngine = LoadModel<SQRModelInput, SQRModelOutput>("SQRmodel.zip");
            sqrtPredictionEngine = LoadModel<SQRTModelInput, SQRTModelOutput>("SQRTmodel.zip");
            powPredictionEngine = LoadModel<POWModelInput, POWModelOutput>("POWmodel.zip");
            rootPredictionEngine = LoadModel<ROOTModelInput, ROOTModelOutput>("ROOTmodel.zip");
            cosPredictionEngine = LoadModel<COSModelInput, COSModelOutput>("COSmodel.zip");
            sinPredictionEngine = LoadModel<SINModelInput, SINModelOutput>("SINmodel.zip");
            tanPredictionEngine = LoadModel<TANModelInput, TANModelOutput>("TANmodel.zip");
        }

        private void TrainModels()
        {
            try
            {
                TrainModelIfNotExists("ADDmodel.zip", "training_dataADD.csv", (modelPath, dataPath) => ADDmodel.Train(modelPath, dataPath));
                TrainModelIfNotExists("SUBmodel.zip", "training_dataSUB.csv", (modelPath, dataPath) => SUBmodel.Train(modelPath, dataPath));
                TrainModelIfNotExists("MULmodel.zip", "training_dataMUL.csv", (modelPath, dataPath) => MULmodel.Train(modelPath, dataPath));
                TrainModelIfNotExists("DIVmodel.zip", "training_dataDIV.csv", (modelPath, dataPath) => DIVmodel.Train(modelPath, dataPath));
                TrainModelIfNotExists("SQRmodel.zip", "training_dataSQR.csv", (modelPath, dataPath) => SQRmodel.Train(modelPath, dataPath));
                TrainModelIfNotExists("SQRTmodel.zip", "training_dataSQRT.csv", (modelPath, dataPath) => SQRTmodel.Train(modelPath, dataPath));
                TrainModelIfNotExists("POWmodel.zip", "training_dataPOW.csv", (modelPath, dataPath) => POWmodel.Train(modelPath, dataPath));
                TrainModelIfNotExists("ROOTmodel.zip", "training_dataROOT.csv", (modelPath, dataPath) => ROOTmodel.Train(modelPath, dataPath));
                TrainModelIfNotExists("COSmodel.zip", "training_dataCOS.csv", (modelPath, dataPath) => COSmodel.Train(modelPath, dataPath));
                TrainModelIfNotExists("SINmodel.zip", "training_dataSIN.csv", (modelPath, dataPath) => SINmodel.Train(modelPath, dataPath));
                TrainModelIfNotExists("TANmodel.zip", "training_dataTAN.csv", (modelPath, dataPath) => TANmodel.Train(modelPath, dataPath));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error training models: {ex.Message}");
                System.Diagnostics.Debug.WriteLine($"Error training models: {ex.Message}");
            }
        }

        private void TrainModelIfNotExists(string modelPath, string dataPath, Action<string, string> trainAction)
        {
            string fullModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, modelPath);
            if (!File.Exists(fullModelPath))
            {
                Console.WriteLine($"Training {modelPath}...");
                trainAction(modelPath, dataPath);
                Console.WriteLine($"{modelPath} trained and saved successfully.");
                System.Diagnostics.Debug.WriteLine($"{modelPath} trained and saved successfully.");
            }
            else
            {
                Console.WriteLine($"{modelPath} already exists. Skipping training.");
                System.Diagnostics.Debug.WriteLine($"{modelPath} already exists. Skipping training.");
            }
        }

        private PredictionEngine<TInput, TOutput> LoadModel<TInput, TOutput>(string modelPath)
            where TInput : class
            where TOutput : class, new()
        {
            string fullModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, modelPath);
            Console.WriteLine($"Loading model from path: {fullModelPath}");
            System.Diagnostics.Debug.WriteLine($"Loading model from path: {fullModelPath}");

            if (!File.Exists(fullModelPath))
            {
                Console.WriteLine($"Model file not found: {fullModelPath}");
                System.Diagnostics.Debug.WriteLine($"Model file not found: {fullModelPath}");
                throw new FileNotFoundException($"Model file not found: {fullModelPath}");
            }

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

                case "x²":
                    var sqrInput = new SQRModelInput { Value = number1 };
                    return sqrPredictionEngine.Predict(sqrInput).Result;

                case "√":
                    var sqrtInput = new SQRTModelInput { Value = number1 };
                    return sqrtPredictionEngine.Predict(sqrtInput).Result;

                case "xʸ":
                    var powInput = new POWModelInput { Value1 = number1, Value2 = number2 };
                    return powPredictionEngine.Predict(powInput).Result;

                case "ʸ√x":
                    var rootInput = new ROOTModelInput { Value1 = number1, Value2 = number2 };
                    return rootPredictionEngine.Predict(rootInput).Result;

                case "cos":
                    var cosInput = new COSModelInput { Value = number1 };
                    return cosPredictionEngine.Predict(cosInput).Result;

                case "sin":
                    var sinInput = new SINModelInput { Value = number1 };
                    return sinPredictionEngine.Predict(sinInput).Result;

                case "tan":
                    var tanInput = new TANModelInput { Value = number1 };
                    return tanPredictionEngine.Predict(tanInput).Result;

                default:
                    throw new InvalidOperationException("Unsupported operation");
            }
        }
    }
}
