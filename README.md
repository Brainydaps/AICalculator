

# AI Calculator

## Overview

AI Calculator is now a pure Artificial Intelligence calculator. All mathematical operations are predicted by dedicated machine learning models working behind. It utilizes a combination of different machine learning models to perform basic arithmetic operations such as addition, subtraction, multiplication, and division. The application integrates different machine learning algorithms with various tunings to enhance calculation accuracy and efficiency. This demonstrates that a relatively accurate calculator can be made without being hard-coded into the app using math libraries. This just demonstrates the power of AI.

## Features

- Addition and subtraction operations remain unchanged, powered by their respective machine learning models.
- Multiplication and division operations are now handled by the FastTree algorithm.
- Root and square root operations are handled by LightGBM.
- Power and square operations are handled by the FastTree algorithm.
- All trigonometric operations (sin, cos, tan) are also handled by the FastTree algorithm, each with their own specialized hyperparameters.
- User-friendly interface designed using Microsoft MAUI for cross-platform compatibility.

## What's New in v2.0 of AI Calculator

- **AI-Driven Operations**: All mathematical operations are now predicted by dedicated machine learning models.
- **Enhanced Algorithms**: Multiplication and division operations are handled by the FastTree algorithm.
- **Advanced Handling for Scientific Operations**: Root and square root operations are managed by LightGBM, while power and square operations are handled by the FastTree algorithm.
- **Specialized Trigonometric Functions**: All trigonometric operations are managed by the FastTree algorithm with specialized hyperparameters.


![Screenshot 2024-07-17 145819](https://github.com/user-attachments/assets/9da55a29-1c28-45bb-983d-083029f1b595)

[YouTube Demonstration](https://youtu.be/65KieyJofJE)

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License. See the [LICENSE](LICENSE) file for details.

## Structure

The project is structured as follows:

- **Model Training**: Machine learning models for various operations (ADD, SUB, MUL, DIV, ROOT, SQRT, POW, SQR, SIN, COS, TAN) are trained using training data located in separate CSV files (e.g., `training_dataADD.csv`, `training_dataMUL.csv`).
- **Model Implementation**: Each operation is encapsulated within its respective model class (e.g., `ADDmodel.cs`, `MULmodel.cs`, `SINmodel.cs`).

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


- **Model Initialization**: Initializes instances of each machine learning model (ADD, SUB, MUL, DIV, ROOT, SQRT, POW, SQR, SIN, COS, TAN) during application startup.
- **Operation Routing**: Routes user input (arithmetic operations) to the respective model for processing.
- **Prediction Handling**: Uses the trained models to predict results for addition, subtraction, multiplication, division, and scientific operations.

- **Error Handling**: Manages exceptions and errors that may occur during model invocation or prediction.

## Contribution

Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

## Acknowledgments

- Thanks to Microsoft MAUI for providing a robust framework for developing cross-platform applications.

- Special thanks to the creators and contributors of LightGBM and FastTree machine learning algorithms.

