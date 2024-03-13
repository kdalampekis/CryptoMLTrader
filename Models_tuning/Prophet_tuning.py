from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
import pandas as pd
import pickle


def train_and_evaluate_prophet(df, target_column):
    # Define a comprehensive parameter grid including new parameters
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'n_changepoints': [20, 25, 30],  # Additional parameter
        # Consider other parameters as per your dataset characteristics
    }

    grid = list(ParameterGrid(param_grid))
    results = []

    for params in grid:
        m = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays_prior_scale=params['holidays_prior_scale'],
            seasonality_mode=params['seasonality_mode'],
            n_changepoints=params['n_changepoints']
        )

        # Add custom seasonality and holidays if applicable
        # m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        # m.add_country_holidays(country_name='US')

        # Prepare the dataset for the specified target
        df_prophet = df[['ds', target_column]].rename(columns={target_column: 'y'})
        m.fit(df_prophet)

        # Cross-validation and performance metrics
        df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='60 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)

        # Collect and store results
        results.append({
            'params': params,
            'rmse': df_p['rmse'].mean(),
            'mae': df_p['mae'].mean(),
            'mape': df_p['mape'].mean()
        })

    # Identify and return the best parameters based on RMSE
    best_params = sorted(results, key=lambda x: x['rmse'])[0]
    print(best_params)
    best_params = sorted(results, key=lambda x: x['rmse'])[0]['params']

    # Retrain the best model
    best_model = Prophet(
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        holidays_prior_scale=best_params['holidays_prior_scale'],
        seasonality_mode=best_params['seasonality_mode'],
        n_changepoints=best_params['n_changepoints']
    )
    df_prophet = df[['ds', target_column]].rename(columns={target_column: 'y'})
    best_model.fit(df_prophet)

    return best_model, best_params


def main():
    # Load your dataset
    df_prophet = pd.read_csv('../Data/df_csv_minmax.csv')
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    # Train and evaluate for 'high' target
    best_model_high, best_params_high = train_and_evaluate_prophet(df_prophet, 'y')
    print("Best Parameters for High:", best_params_high)

    # Save the best model for 'high' target
    with open('../Trained_Models/best_prophet_high.pkl', 'wb') as pkl:
        pickle.dump(best_model_high, pkl)

    # Train and evaluate for 'low' target
    best_model_low, best_params_low = train_and_evaluate_prophet(df_prophet, 'low')
    print("Best Parameters for Low:", best_params_low)

    with open('../Trained_Models/best_prophet_low.pkl', 'wb') as pkl:
        pickle.dump(best_model_low, pkl)


if __name__ == '__main__':
    main()
