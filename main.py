import importlib.util
from importlib.machinery import SourceFileLoader


project_path_dict = {1: "./projects/humidity/", 2: "./projects/mobility/"}


def run_project(TransformationObject):
    exit = False
    while not exit:
        print("Select the function to perform")
        print("1. Get Transformed (Pre-Processed) Dataset")
        print("2. Get Model")
        print("3. Fetch Predictions")
        print("4. Retrain Model [Optional]")
        print("5. Exit")
        choice = int(input("Enter Choice (1-5) : "))
        if choice == 1:
            x = TransformationObject.transform()
            print(x)
        if choice == 2:
            x = TransformationObject.load_model()
            print(x)
        if choice == 3:
            x = TransformationObject.get_predictions()
            print(x)
        if choice == 4:
            try:
                TransformationObject.train_model()
                print("Model Trained Successfully")
            except Exception as ex:
                print(ex)
        if choice == 5:
            break


if __name__ == "__main__":
    print("Select a Project to Run")
    print("List of Available Projects")
    print("1. Humidity Prediction using XGBoost")
    print("2. Google Mobility Trends")
    ch = int(input("Which Project you would like to select : "))
    if ch == 1:
        class_path = project_path_dict[ch] + "Transformation.py"

        obj = (
            SourceFileLoader("Transformation", class_path)
            .load_module()
            .Transformation()
        )
