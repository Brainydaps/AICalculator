using System;
using Microsoft.Maui.Controls;

namespace AICalculator
{
    public partial class MainPage : ContentPage
    {
        string displayText = "";
        string tempDisplay = "";
        double number1 = 0;
        double number2 = 0;
        string operation = "";
        ModelCall modelCall;

        public string DisplayText
        {
            get { return displayText; }
            set
            {
                displayText = value;
                OnPropertyChanged(nameof(DisplayText));
            }
        }

        public string TempDisplay
        {
            get { return tempDisplay; }
            set
            {
                tempDisplay = value;
                OnPropertyChanged(nameof(TempDisplay));
            }
        }

        public MainPage()
        {
            InitializeComponent();
            BindingContext = this;

            modelCall = new ModelCall();
        }

        void Button_Click(object sender, EventArgs e)
        {
            Button button = (Button)sender;
            string text = button.Text;

            switch (text)
            {
                case "=":
                    CalculateResult();
                    break;
                case "÷":
                case "×":
                case "−":
                case "+":
                case "xʸ":
                case "ʸ√x":
                    SetOperation(text);
                    break;
                case "cos":
                case "sin":
                case "tan":
                    PerformTrigonometricOperation(text);
                    break;
                case "x²":
                    CalculateSquare();
                    break;
                case "√":
                    CalculateSquareRoot();
                    break;
                case "C":
                    ClearDisplay();
                    break;
                default:
                    AppendToDisplay(text);
                    break;
            }
        }

        void SetOperation(string op)
        {
            if (double.TryParse(DisplayText, out number1))
            {
                operation = op;
                TempDisplay = DisplayText + " " + op + " ";
                DisplayText = "";
            }
        }

        void PerformTrigonometricOperation(string op)
        {
            if (double.TryParse(DisplayText, out number1))
            {
                double result = modelCall.Predict(op, (float)number1, 0);
                DisplayText = result.ToString();
                TempDisplay = "";
            }
        }

        void CalculateSquare()
        {
            if (double.TryParse(DisplayText, out number1))
            {
                double result = modelCall.Predict("x²", (float)number1, 0);
                DisplayText = result.ToString();
                TempDisplay = "";
            }
        }

        void CalculateSquareRoot()
        {
            if (double.TryParse(DisplayText, out number1))
            {
                double result = modelCall.Predict("√", (float)number1, 0);
                DisplayText = result.ToString();
                TempDisplay = "";
            }
        }

        void ClearDisplay()
        {
            DisplayText = "";
            TempDisplay = "";
            number1 = 0;
            number2 = 0;
            operation = "";
        }

        void AppendToDisplay(string text)
        {
            DisplayText += text;
        }

        void CalculateResult()
        {
            if (double.TryParse(DisplayText, out number2))
            {
                double result = modelCall.Predict(operation, (float)number1, (float)number2);
                DisplayText = result.ToString();
                number1 = result; // Store result for consecutive operations
                TempDisplay = "";
                operation = "";
            }
        }
    }
}
