import importlib.util
from importlib.machinery import SourceFileLoader


project_path_dict = {1: "./projects/humidity/", 2: "./projects/mobility/"}

if __name__ == "__main__":
    print("Select a Project to Run")
    print("List of Available Projects")
    print("1. Humidity Prediction using XGBoost")
    print("2. Google Mobility Trends")
    # choice = int(input("Which Project you would like to select : "))
    choice = 2
    if choice == 1:
        class_path = project_path_dict[choice] + "Transformation.py"

        obj = (
            SourceFileLoader("Transformation", class_path)
            .load_module()
            .Transformation()
        )
