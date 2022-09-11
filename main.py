import importlib.util
from importlib.machinery import SourceFileLoader


project_path_dict = {1: "./projects/humidity/"}

if __name__ == "__main__":
    print("Select a Project to Run")
    print("List of Available Projects")
    print("1. Humidity Prediction using XGBoost")
    # choice = int(input("Which Project you would like to select : "))
    choice = 1
    if choice == 1:
        class_path = project_path_dict[choice] + "Transformation.py"

        obj = (
            SourceFileLoader("Transformation", class_path)
            .load_module()
            .Transformation()
        )
