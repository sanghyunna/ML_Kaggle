def preprocess(df):
    features_to_drop = [129,29,182,14,84,98,41,61,79,100,183,158,46,47,126,42,160,7,38,73,185,30,10,27,103,124,136,17,117,39,161,96]
    for feature in features_to_drop:
        feature_name = f'var_{str(feature)}'
        df.drop([feature_name], axis=1, inplace=True)
    return df