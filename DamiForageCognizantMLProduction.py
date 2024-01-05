# DamiForageCognizantAI

#Importing packages 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

#Global Constraints 
K = 10 

# Split defines the % of data that will be used in the training sample
# 1 - SPLIT = the % used for testing
SPLIT = 0.75

class dataset:
    def load(self, file_path):
        #This function is created to load the data pathfile and 
        #empty null values. It returns the data in pd format 
        data = pd.read_csv(file_path)
        data.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
        return data
       
    def var_mod(self, data: pd.DataFrame = None, target: str = "estimated_stock_pct"):
        #This is my variable modification function created to set our  
        #target and feature variables as X and Y
        
        # Check to see if the target variable is present in the data
        if target not in data.columns:
            raise Exception(target ," is not present in the data.")
        
        X = data.drop(columns = [target]) #feature
        y = data[target] #target
        return X,y
        
    def train(self, X: pd.DataFrame = None, y: pd.Series = None):
        # Create a list that will store the accuracies of each fold
        accuracy = []

        # Enter a loop to run K folds of cross-validation
        for fold in range(0, K):
            # Instantiate algorithm and scaler
            model = RandomForestRegressor()
            scaler = StandardScaler()
            
            # Create training and test samples
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)
            
            # Scale X data, we scale the data because it helps the algorithm to converge
            # and helps the algorithm to not be greedy with large values
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
            # Train model
            trained_model = model.fit(X_train, y_train)
    
            # Generate predictions on test sample
            y_pred = trained_model.predict(X_test)
    
            # Compute accuracy, using mean absolute error
            mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
            accuracy.append(mae)
            print(f"Fold {fold + 1}: MAE = {mae:.3f}")
            
        # Finish by computing the average MAE across all folds
        print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")

sales_df = "/content/drive/MyDrive/sales.csv"
stock_df = "/content/drive/MyDrive/sensor_stock_levels.csv"
temp_df = "/content/drive/MyDrive/sensor_storage_temperature.csv"
merged_df = "/content/drive/MyDrive/merged_df.csv"

#Main Function 
# Create an instance of the class
ins = Dataset()

# Call a method of the class
ins.load(merged_df)

# Now split the data into predictors and target variables
data = pd.read_csv(merged_df)
tar = "estimated_stock_pct"
a, b = ins.var_mod(data,tar)
#print("Number of samples in X:", data)#len(data))
#print("Number of samples in y:", tar)#len(tar))


# Finally, train the machine learning model
ins.train(a, b)