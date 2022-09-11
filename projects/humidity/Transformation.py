import pip
import importlib


class Transformation:
    def __init__(self):
        self.project_id = 1
        # Mention Required Packages Here
        self.requirements = [
            "pandas",
            "numpy",
            "matplotlib",
            "plotly",
            "warnings",
            "statsmodels",
            "os",
            "datetime",
            "xgboost",
            "skforecast",
            "joblib",
            "seaborn",
        ]
        self.install_requirements()
        self.project_metadata = {
            "col_map": {
                "CloudOpacity": "opacity",
                "Ghi": "ghi",
                "GtiFixedTilt": "gii_seasonal",
                "GtiTracking": "gii_tracker",
                "PrecipitableWater": "water",
                "RelativeHumidity": "humidity",
                "SurfacePressure": "pressure",
                "WindSpeed10m": "wind_speed",
            },
            "cols_to_keep": [
                "opacity",
                "ghi",
                "gii_seasonal",
                "gii_tracker",
                "water",
                "pressure",
                "wind_speed",
                "humidity",
            ],
            "features": [
                "opacity",
                "ghi",
                "gii_seasonal",
                "gii_tracker",
                "water",
                "pressure",
                "wind_speed",
            ],
            "label": ["humidity"],
            "exog_vars": [
                "opacity",
                "gii_seasonal",
                "water",
                "water_opacity",
                "water_pressure",
                "all_gi",
            ],
        }
        print("Humidity Class Imported")

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

    def get_train_val_test_data(self):
        from datetime import datetime

        df = self.transform()
        train_split_date = datetime(2021, 5, 1)
        val_split_date = datetime(2021, 9, 22)
        train_data = df[df.index < train_split_date]
        test_data = df[df.index >= val_split_date]
        val_data = df[(df.index >= train_split_date) & (df.index < val_split_date)]
        return train_data, val_data, test_data

    def train_model(self):
        import pandas as pd
        import joblib
        from skforecast.model_selection import grid_search_forecaster
        from skforecast.ForecasterAutoreg import ForecasterAutoreg
        from xgboost import XGBRegressor

        # param_grid = {
        #     "n_estimators": [100, 500],
        #     "max_depth": [3, 5, 10],
        #     "learning_rate": [0.01, 0.1, 0.02, 0.05, 0.2],
        # }

        # We know the best values for different hyperparameters
        param_grid = {
            "n_estimators": [500],
            "max_depth": [10],
            "learning_rate": [0.02],
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

    @staticmethod
    def load_dataset():
        import pandas as pd

        df = pd.read_csv("./dataset.csv")
        return df

    def transform(self):
        import pandas as pd

        df = self.load_dataset()
        df.drop(columns=["PeriodStart", "Period"], inplace=True)
        df.rename(columns={"PeriodEnd": "date_and_time"}, inplace=True)
        df["date_and_time"] = pd.to_datetime(df["date_and_time"])
        df["date_and_time"] = df["date_and_time"].dt.tz_localize(None)
        df = df.sort_values(by="date_and_time")
        print(f"Start Date : {df['date_and_time'][0]}")
        print(f"end Date : {df['date_and_time'][len(df) - 1]}")
        df = df.set_index("date_and_time")
        col_map = self.project_metadata.get("col_map")
        cols_to_keep = self.project_metadata.get("cols_to_keep")
        if col_map is not None:
            df.rename(columns=col_map, inplace=True)
        if cols_to_keep is not None:
            df = df[cols_to_keep]
        df = self.feature_engineering(df)
        exog_vars = self.project_metadata.get("exog_vars")
        if exog_vars is not None:
            label = self.project_metadata["label"]
            df = df[exog_vars + label]
        return df

    def get_predictions(self):
        _, _, test_df = self.get_train_val_test_data()
        model = self.load_model()
        label = self.project_metadata["label"]
        preds = model.predict_interval(
            steps=len(test_df), exog=test_df.drop(columns=label)
        )
        test_df["predictions"] = preds["pred"].tolist()
        df = test_df.reset_index()[["date_and_time", "humidity", "pred"]].rename(
            columns={"pred": "predicted"}
        )
        return df

    @staticmethod
    def feature_engineering(df):
        df["water_opacity"] = df["water"] * df["opacity"]
        df["water_pressure"] = df["water"] * df["pressure"]
        df["all_gi"] = (df["ghi"] / df["gii_seasonal"]) * df["water_pressure"]
        return df

    def test_fun(self):
        print(f"The Project ID is {self.project_id}")
