import pandas as pd

def create_submission_csv(model, df, filename, scaler=None, imputer=None):
    # SUBMISSION
    test_df = df.copy()

    test_df.drop(['ID_code'], axis=1, inplace=True)

    if scaler is not None:
        test_df = scaler.transform(test_df)

    if imputer is not None:
        test_df = imputer.transform(test_df)

    predictions = model.predict(test_df)
    
    submission = pd.DataFrame({'ID_code': df['ID_code'].values, 'target': predictions})
    submission.to_csv(filename, index=False)

    print('Saved file: ' + filename)