"""
Project 2: Commercial Fleet Predictive Maintenance & Remaining Useful Life (RUL) Forecasting
---------------------------------------------------------------------------------------------
Module V: IoT Analytics
---------------------------------------------------------------------------------------------
Architecture:
  [ Fleet Sensor Logs ] ──> [ Feature Engineering (Rolling Stats) ]
                                              │
  [ Maintenance Schedule ] <──[ RUL Forecast ] ─── [ Random Forest Regressor ]
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# =====================================================================
# 1. SYNTHETIC FLEET TELEMETRY GENERATOR
# =====================================================================
def generate_fleet_telemetry(num_engines=25):
    """Simulates multi-sensor wear degradation sequences across fleet vehicles."""
    records = []
    np.random.seed(42)

    for engine_id in range(1, num_engines + 1):
        total_lifespan_cycles = np.random.randint(130, 260)
        base_temp = 450.0       # Standard exhaust gas temperature (°F)
        base_vibration = 0.02   # Standard vibration baseline (g)

        for cycle in range(1, total_lifespan_cycles + 1):
            # Accelerated wear factor as cycles approach engine failure
            degradation = (cycle / total_lifespan_cycles) ** 2

            sensor_temp = base_temp + (degradation * 85.0) + np.random.normal(0, 2.5)
            sensor_vib = base_vibration + (degradation * 0.45) + np.random.normal(0, 0.01)
            remaining_useful_life = total_lifespan_cycles - cycle

            records.append({
                "engine_id": engine_id,
                "cycle": cycle,
                "sensor_temp_f": round(sensor_temp, 2),
                "sensor_vibration_g": round(sensor_vib, 4),
                "RUL": remaining_useful_life
            })

    return pd.DataFrame(records)


# =====================================================================
# 2. FEATURE ENGINEERING & ANALYTICS PIPELINE
# =====================================================================
if __name__ == "__main__":
    print("==========================================================================")
    print("     MODULE V: FLEET PREDICTIVE MAINTENANCE & RUL FORECASTING SYSTEM     ")
    print("==========================================================================\n")

    print("📊 Generating Engine Degradation Historical Telemetry Data...")
    df = generate_fleet_telemetry(num_engines=30)
    print(f"✅ Generated {len(df)} telemetry logs across 30 fleet engines.")

    # Rolling window features to capture degradation momentum over time
    print("\n⚙️ Calculating Time-Series Rolling Window Indicators...")
    df['temp_rolling_avg'] = (
        df.groupby('engine_id')['sensor_temp_f']
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    df['vib_rolling_std'] = (
        df.groupby('engine_id')['sensor_vibration_g']
        .transform(lambda x: x.rolling(5, min_periods=1).std())
        .fillna(0)
    )

    print("\nProcessed Data Sample (First 5 Cycles):")
    print(df[['engine_id', 'cycle', 'sensor_temp_f', 'sensor_vibration_g', 'temp_rolling_avg', 'RUL']].head(5))

    # Features and Target selection
    feature_columns = ['cycle', 'sensor_temp_f', 'sensor_vibration_g', 'temp_rolling_avg', 'vib_rolling_std']
    X = df[feature_columns]
    y = df['RUL']

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Model Training
    print("\n🤖 Training Random Forest Regressor Model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model Training Complete.")

    # Evaluation Metrics
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--------------------------------------------------------------------------")
    print("                     MODEL PERFORMANCE EVALUATION                         ")
    print("--------------------------------------------------------------------------")
    print(f" Root Mean Squared Error (RMSE) : {rmse:.2f} cycles")
    print(f" R² Accuracy Score              : {r2 * 100:.2f}%")
    print("--------------------------------------------------------------------------")

    # Real-Time Operational Prediction Test
    print("\n🔮 INFERENCE TEST ON LIVE FIELD ENGINE DATA:")
    live_engine_data = pd.DataFrame([{
        'cycle': 115,
        'sensor_temp_f': 512.4,
        'sensor_vibration_g': 0.22,
        'temp_rolling_avg': 508.1,
        'vib_rolling_std': 0.031
    }])

    predicted_rul = model.predict(live_engine_data)[0]
    print(f" Engine Status Inputs  : Cycle=115 | Temp=512.4°F | Vib=0.22g")
    print(f" 🎯 Predicted Remaining Useful Life (RUL): {predicted_rul:.1f} Cycles")

    if predicted_rul < 20:
        print(" 🚨 STATUS ACTION: SCHEDULE IMMEDIATE MAINTENANCE WORKSHOP VISIT!")
    else:
        print(" ✅ STATUS ACTION: OPERATIONAL WITHIN SAFE DEGRADATION LIMITS.")
    print("==========================================================================\n")