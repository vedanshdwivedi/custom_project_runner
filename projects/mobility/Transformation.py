import pip
import importlib


class Transformation:
    def __init__(self):
        self.project_id = 2
        # Mention Required Packages Here
        self.cols_to_drop = ["area_name"]
        self.ac_to_num_dict = {
            "E09000001": 1,
            "E09000033": 2,
            "E09000002": 3,
            "E09000003": 4,
            "E09000004": 5,
            "E09000005": 6,
            "E09000006": 7,
            "E09000007": 8,
            "E09000008": 9,
            "E09000009": 10,
            "E09000010": 11,
            "E09000012": 12,
            "E09000013": 13,
            "E09000014": 14,
            "E09000015": 15,
            "E09000016": 16,
            "E09000017": 17,
            "E09000018": 18,
            "E09000019": 19,
            "E09000022": 20,
            "E09000023": 21,
            "E09000024": 22,
            "E09000025": 23,
            "E09000026": 24,
            "E09000027": 25,
            "E09000028": 26,
            "E09000029": 27,
            "E09000030": 28,
            "E09000031": 29,
            "E09000032": 30,
            "E09000011": 31,
            "E09000020": 32,
            "E09000021": 33,
        }
        self.num_to_ac_dict = {
            1: "E09000001",
            2: "E09000033",
            3: "E09000002",
            4: "E09000003",
            5: "E09000004",
            6: "E09000005",
            7: "E09000006",
            8: "E09000007",
            9: "E09000008",
            10: "E09000009",
            11: "E09000010",
            12: "E09000012",
            13: "E09000013",
            14: "E09000014",
            15: "E09000015",
            16: "E09000016",
            17: "E09000017",
            18: "E09000018",
            19: "E09000019",
            20: "E09000022",
            21: "E09000023",
            22: "E09000024",
            23: "E09000025",
            24: "E09000026",
            25: "E09000027",
            26: "E09000028",
            27: "E09000029",
            28: "E09000030",
            29: "E09000031",
            30: "E09000032",
            31: "E09000011",
            32: "E09000020",
            33: "E09000021",
        }
        self.requirements = [
            "pandas",
            "numpy",
            "matplotlib",
            "warnings",
            "os",
            "datetime",
            "xgboost",
            "skforecast",
            "joblib",
        ]
        self.install_requirements()
        self.project_metadata = {
            "col_map": {
                "date": "date_and_time",
                "retail_and_recreation_percent_change_from_baseline": "retail",
                "grocery_and_pharmacy_percent_change_from_baseline": "grocery",
                "parks_percent_change_from_baseline": "parks",
                "transit_stations_percent_change_from_baseline": "transit",
                "workplaces_percent_change_from_baseline": "workplace",
                "residential_percent_change_from_baseline": "residential",
            },
            "cols_to_keep": ["date_and_time", "area_code", "movement"],
            "label": ["movement"],
            "exog_vars": ["area_code", "day", "month", "year", "weekday"],
        }
        print("Mobility Class Imported")

    def install_requirements(self) -> None:
        for package in self.requirements:
            try:
                importlib.import_module(package)
            except ImportError:
                pip.main(["install", package])
                importlib.import_module(package)

    @staticmethod
    def load_model():
        import joblib

        model = joblib.load("model")
        return model

    def get_train_val_test_data(self, df):
        train_data = df[: int(len(df) * 0.7)]
        val_data = df[int(len(df) * 0.7) : int(len(df) * 0.85)]
        test_data = df[int(len(df) * 0.85) :]
        return train_data, val_data, test_data

    def train_model(self):
        import pandas as pd
        import joblib
        from skforecast.model_selection import grid_search_forecaster
        from skforecast.ForecasterAutoreg import ForecasterAutoreg
        from xgboost import XGBRegressor

        param_grid = {
            "n_estimators": [500],
            "max_depth": [10],
            "learning_rate": [0.01],
        }

        # Lags used as predictors
        lags_grid = [[1, 2, 3, 23, 24, 25, 71, 72, 73]]
        forecaster = ForecasterAutoreg(
            regressor=XGBRegressor(random_state=123), lags=24
        )

        train_data, val_data, test_data = self.get_train_val_test_data()
        df3 = pd.concat([train_data, val_data], sort=True)
        label = self.project_metadata["label"]
        exog_vars = self.project_metadata["exog_vars"]

        grid_search_forecaster(
            forecaster=forecaster,
            y=df3[label],  # Train and validation data
            exog=df3[exog_vars],
            param_grid=param_grid,
            lags_grid=lags_grid,
            steps=36,
            refit=False,
            metric="mean_squared_error",
            initial_train_size=int(
                len(train_data)
            ),  # Model is trained with trainign data
            return_best=True,
            verbose=False,
        )

        print("Model Trained, Saving Model : ")
        joblib.dump(forecaster, "model")

    def load_dataset(self):
        import pandas as pd

        df = pd.read_csv("./dataset.csv")
        return df

    def transform(self):
        import pandas as pd

        df = self.load_dataset()
        df = df.drop(columns=self.cols_to_drop)
        df = df.rename(columns=self.project_metadata["col_map"])
        df["date_and_time"] = pd.to_datetime(df["date_and_time"])
        for cause in self.project_metadata["movement_causes"]:
            df[cause] = df[cause].fillna(0)
        df["area_code"] = df["area_code"].apply(lambda x: self.ac_to_num_dict[x])
        df["movement"] = df[self.project_metadata["movement_causes"]].sum(axis=1)
        df = df[self.project_metadata["cols_to_keep"]]
        df = df.sort_values(by=["date_and_time", "area_code"]).reset_index(drop=True)
        df["day"] = df["date_and_time"].dt.day
        df["year"] = df["date_and_time"].dt.year
        df["month"] = df["date_and_time"].dt.month
        df["week"] = df["date_and_time"].dt.week
        df["weekday"] = df["date_and_time"].dt.weekday
        df.drop(columns=["date_and_time"], inplace=True)
        return df

    def get_predictions(self):
        from datetime import datetime

        _, _, test_df = self.get_train_val_test_data()
        model = self.load_model()
        label = self.project_metadata["label"]
        preds = model.predict_interval(
            steps=len(test_df), exog=test_df.drop(columns=label)
        )
        test_df["predictions"] = preds["pred"].tolist()
        test_df["date_and_time"] = test_df.apply(
            lambda row: datetime(int(row["year"]), int(row["month"]), int(row["day"])),
            axis=1,
        )
        df = test_df[["date_and_time", "area_code", "movement", "pred"]].rename(
            columns={"pred": "predicted"}
        )
        df["area_code"] = df["area_code"].apply(lambda x: self.num_to_ac_dict[x])
        return df

    def test_fun(self):
        print(f"The Project ID is {self.project_id}")
