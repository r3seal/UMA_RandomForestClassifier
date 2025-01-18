import pandas as pd
from sklearn.model_selection import train_test_split

def gym_data():
    # Step 1: Load the dataset
    data_path = 'data/gym_members_exercise_tracking.csv'  # Replace with the actual file path
    df = pd.read_csv(data_path)

    # Step 2: Encode categorical variables manually
    # Gender: Male = 0, Female = 1
    df['Gender'] = df['Gender'].apply(lambda x: 0 if x == 'Male' else 1)

    # Workout_Type: Assign numeric values to each workout type
    workout_type_mapping = {workout: idx for idx, workout in enumerate(df['Workout_Type'].unique())}
    df['Workout_Type'] = df['Workout_Type'].apply(lambda x: workout_type_mapping[x])

    # Step 3: Define features (X) and target (y)
    X = df.drop('Experience_Level', axis=1).values
    y = df['Experience_Level'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test