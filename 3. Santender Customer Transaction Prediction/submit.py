import pandas as pd

def create_submission_csv(model, df, filename, scaler=None, imputer=None):
    # SUBMISSION
    test_df = df.copy()

    test_df.drop(['ID_code'], axis=1, inplace=True)

    features_to_drop = [129,29,182,14,84,98,41,61,79,100,183,158,46,47,126,42,160,7,38,73,185,30,10,27,103,124,136,17,117,39,161,96]
    for feature in features_to_drop:
        feature_name = f'var_{str(feature)}'
        test_df.drop([feature_name], axis=1, inplace=True)

    if scaler is not None:
        test_df = scaler.transform(test_df)

    if imputer is not None:
        test_df = imputer.transform(test_df)

    predictions = model.predict(test_df)
    
    submission = pd.DataFrame({'ID_code': df['ID_code'].values, 'target': predictions})
    submission.to_csv(filename, index=False)

    print('Saved file: ' + filename)