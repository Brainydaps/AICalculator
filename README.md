
# AI Calculator

## Overview

So I decided to develop the AI version of my calculator app(check my repos), we would call it AI Calculator. It utilizes a combination of different machine learning models to perform basic arithmetic operations such as addition, subtraction, multiplication, and division. The application integrates different machine learning algorithms with different tunings to enhance calculation accuracy and efficiency. It is my way of demonstrating that a relatively accurate calculator can be made without being hard coded into the app using math libraries. This just demonstrates the power of AI.

## Features

- Addition, subtraction, multiplication, and division operations are powered by different machine learning models.
- Trained models using the LightGBM algorithm for multiplication and division, and SDCA for addition and subtraction.
- Support for basic arithmetic operations and scientific functions(these ones use the normal math libraries provided by C#, i would see if i can create machine learning models for them also in future updates) such as trigonometric calculations, square, and square root.
- User-friendly interface designed using Microsoft MAUI for cross-platform compatibility.

  ## What's New in v1.2 of AI Calculator

- **Updated Training Data**: Enhanced the training datasets for multiplication (MUL) and division (DIV) models with more relevant and comprehensive data to improve pattern recognition.
- **Algorithm Improvements**: Updated the algorithms for MUL and DIV models with optimizations and tunings made using AutoML from ML.NET.
- **Increased Accuracy**: Multiplication and division calculations are now more accurate than in previous versions, providing users with more reliable results.

These improvements ensure that the AI Calculator continues to deliver high-quality performance and better accuracy for your arithmetic operations.


![Screenshot 2024-07-17 145819](https://github.com/user-attachments/assets/9da55a29-1c28-45bb-983d-083029f1b595)

[YouTube Demonstration](https://youtu.be/65KieyJofJE)

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License. See the [LICENSE](LICENSE) file for details.

## Structure

The project is structured as follows:

- **Model Training**: Machine learning models (ADD, SUB, MUL, DIV) are trained using training data located in separate CSV files (`training_dataADD.csv`, `training_dataSUB.csv`, `training_dataMUL.csv`, `training_dataDIV.csv`).
- **Model Implementation**: Each operation (ADD, SUB, MUL, DIV) is encapsulated within its respective model class (`ADDmodel.cs`, `SUBmodel.cs`, `MULmodel.cs`, `DIVmodel.cs`).
- **Model Invocation**: The `ModelCall.cs` class manages the invocation of models based on user operations, ensuring that each arithmetic operation is handled by the appropriate machine learning model.
- **User Interface**: The calculator UI is implemented using Microsoft MAUI, with XAML files (`MainPage.xaml`) defining the layout and C# files (`MainPage.xaml.cs`) handling user interactions and calculations.
- **Integration**: Machine learning models are seamlessly integrated into the calculator logic to provide accurate results for each arithmetic operation.

## Usage

To use the AI Calculator:

1. Clone the repository: `git clone https://github.com/Brainydaps/AICalculator.git`
2. Open the project in Visual Studio or your preferred IDE.
3. Build and run the application on your desired platform (Windows or macOS, hopefully in future ML.NET would support mobile OS).

## `ModelCall` Class

The `ModelCall.cs` class is responsible for managing the machine learning models and their invocation based on user input:

- **Model Initialization**: Initializes instances of each machine learning model (ADD, SUB, MUL, DIV) during application startup.
- **Operation Routing**: Routes user input (arithmetic operations) to the respective model for processing.
- **Prediction Handling**: Uses the trained models to predict results for addition, subtraction, multiplication, and division operations.
- **Error Handling**: Manages exceptions and errors that may occur during model invocation or prediction.

## Contribution

Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

## Acknowledgments

- Thanks to Microsoft MAUI for providing a robust framework for developing cross-platform applications.
- Special thanks to the creators and contributors of LightGBM and SDCA machine learning algorithms.

```
