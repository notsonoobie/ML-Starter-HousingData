# ML-Starter-HousingData
A Starter ML Project containing Housing DataSet for Prediction of Housing Prices (Mean) using various Algorithms like Linear Regression, Decision Tree Regressor, Random Forest Regressor. 

Housing Data Set : [Github Source](https://github.com/ageron/handson-ml/tree/master/datasets/housing)
Features : 
  1. Automatically Fetches the data from the source and extracts the Data.
  2. Pandas for DataFrame
  3. Matplotlib for Data Visualization and Analysis
  4. Scikit Learn for MachineLeaning
  4. SciKit ML Library Features used in the project:
        1. [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) for Efficently Splitting Dataset into Training(80%) and Testing(20%) Sets.
        2. [StratifiedShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) for Unbiased Splitting and shuffling of Data Sets.
        3. [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) for Dealing with missing values/Nan's.
        4. [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html), [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), and most importantly [LabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html) for Labelling Categorical Data.
        5. [TransformerMixin](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html) to get fit_transform() method for our custom Class/Transformers.
        6. [BaseEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html) to get set_params() and get_params() methods for our custom Class/Transformers.
        7. [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) for making various preprocessing steps easier.
        8. [FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) for Joining Pipelines.
        9. [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) for Capping Data Values to prevent ML Algorithms to fall into trap of Numbers.
        10. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) for Capping Data Values to prevent ML Algorithms to fall into trap of Numbers.
        11. [mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) for finding errors in the expected predictions and predictions made by the model.
        12. [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) for Cross Validation of our ML Model.
        13. [joblib](https://joblib.readthedocs.io/en/latest/) for serializing our Model Instances into memory.
        14. [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) for Fine-Tuning i.e finding which instances/attributes are good for predictions.
  5. Algorthms Used :
        1. [___LinearRegression___](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
        2. [___DecisionTressRegressor___](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
        3. [___RandomForestRegressor___](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
        
