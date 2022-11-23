### [ PREDICTION OF MOST SUITABLE SURFBOARD TYPE FOR ANY KIND OF SURFER ]
### Author: Manex Ormazabal Arregi
### Date: 29/01/2021
## Description of the program: In this Python program we develop a Machine Learning (AI) program to predict the most suitable surfboard type for each kind of surfer.
 ## For that, we feed the machine with some features (Surfer level,age,weight,height...) and we get 4 different targets (Surfboard Volume, Real volume, Size and type).
  ## As we have 4 different targets (Volume, Real Volume and Surfboard shape), we divide our program in 4 different sections and we make a different prediction with each one:
   # 1. Predict the surfboard standard volume based on surfer´s weight and surfing level (skill)
   # 2. Predict the real surfboard volume (more accurate) using the previous surfer´s standard volume, Age and Fitness.
   # 3. Predict the surfboard size (Length) based on the surfer level, weight, height and the previously calculated real volume.
   # 4. Predict the surfboard shape (model) based on the size of waves, surfer level, surfboard real volume and surfboard size.
 ## To make predictions, we will use different tools or methods for Regression along the three approaches, like KNeighbors (KNN), Random Forest or MLP Regressor.
   # Furthermore, in the first approach we will plot in a graph the variation of Train and Test predictions accuracy by using different number of neighbors (value of K).

## START THE PROGRAM:
# STEP 1:
# Import all the libraries that we will use:
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split  # Training algorithm
from sklearn.neighbors import KNeighborsRegressor     # KNN regressor algorithm
from sklearn.neural_network import MLPRegressor       # MLP Regressor algorithm
from sklearn.ensemble import RandomForestClassifier   # Random forest classifier algorithm

# STEP 2:
# Read the Excel file in order to import the data that we will use:

data = pd.read_excel(r'C:\Users\elo\Desktop\ROBOTICS BENG (LHU)\PROJECTS (Courseworks)\ML_Surfboard Prediction\Python program_ML_Surfing\Surfing_Dataset.xlsx')
data = data.replace(np.nan,"0") # Replace any values as "nan" by zeros
df = pd.DataFrame(data) # Create a dataframe with the imported data
print(df)

# STEP 3:
#Create a dictionary for "Surfboard shape" data:

surfboardShape_dict = dict(zip(data.Surfboard_Shape_N.unique(),data.Surfboard_Shape.unique())) # Create a dictionary in order to show up the surfboard type names and not the numbers associated to each type (Surfboard_Type_N).
print(surfboardShape_dict)

print(data['Surfboard_Generic_Volume'].value_counts())

# STEP 4:
### APPROACH 1: PREDICT THE SURFBOARD STANDARD VOLUME. Plot the accuracy variation of Train and Test accuracy by taking different values of K (neighbors)
print("||| PREDICT SURFBOARD VOLUME (using K-NEIGHBORS REGRESSOR) |||")

# Define variables for Features (X) and Targets (y):
X = data[['Surfing_Level_N','Surfer_Weight']] # Define the features that we are going to use. In this case we have 2 variables of features (Column "Surfer_Weight" and "Surfing_Level_N" of the Excel file)
y = data['Surfboard_Generic_Volume']  # The target in this case is the column "Surfboard_Volume" of the Excel file.

print(X)
print(y)
# Train the data:
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.describe()) # Show how many features and targets we keep for training the algorithm

# STEP 5:
# Make the prediction by using different values of K (number of neighbors) and plot the accuracy results of each of them:
neighbors = np.arange(1,9)  # Define how many neighbors we will take (in this case 8. From 1 to 9). We will test how the accuracy of the prediction changes as we use less or more neighbors.
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate (neighbors):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train,y_train)   # Fit the Features with their respective targets
    train_accuracy[i] = knn.score(X_train,y_train)  # Get the train accuracy
    test_accuracy[i] = knn.score(X_test,y_test)    #Get the test accuracy

# Ilustrate and compare the accuracy of Training and Testing accuracy in a graph:
plt.title('Accuracy variation in KNN')  # Title of the graph
plt.plot(neighbors,train_accuracy,label='Training accuracy')   # Represent the line of Training accuracy values
plt.plot(neighbors,test_accuracy,label='Testing accuracy')     # Represent the line of Testing accuracy values
plt.legend()  # Show the legend
plt.xlabel('Number of Neighbors')  # Name of the X axis
plt.ylabel('Accuracy')            # Name of the Y axis
plt.show()

# Make the normal prediction with KNN using 6 neighbors and print the accuracy:
print("Prediction of Surfboard Volume using KNN")
knn = KNeighborsRegressor(n_neighbors=6) # We are specifying that we will use 6 neighbors,since we see in the graph that it gets a high accuracy with this number.
knn.fit(X_train,y_train)  # Fit the training features with the respective training targets
predAccuracy = knn.score(X_test,y_test)  # Print the accuracy of prediction of the machine, using testing data
print("The prediction accuracy of the machine using KNN algorithm was: ",predAccuracy)

# Make the prediction entering any feature and check if the machine predicts correctly and gives the correct target:
surferLevel = input("Enter the surfer Level [1 to 5] : ")
surferWeight = input("Enter surfer Weight (kg) : ")
surfboard_std_volume = knn.predict([[surferLevel,surferWeight]])  ## Here we enter the value of Surfer level and Weight in order to predict the surfboard volume.
print("*** Based on the entered surfer Level and Weight, the suitable surfboard standard volume for this surfer is: ",surfboard_std_volume, "Litres ***")

# STEP 5:
### APPROACH 2: CALCULATE THE REAL SURFBOARD VOLUME.
# We will calculate the Real Volume multiplying the obtained surfboard volume by the coefficients of surfer Age and Fitness.

print("||| Calculate de Real Surfboard Volume for the surfer |||")

surferAge = int(input("Enter the surfer Age : "))
surferFitness = int(input("Enter surfer Fitness [1 to 4] : "))

# Multiply the corresponding Age coefficients:
if surferAge <= 30:                            # First threshold: Age below 30

    resultAge = surfboard_std_volume * 1

elif surferAge >= 31 and surferAge <= 50:      # Second threshold: Age between 31 and 50

    resultAge = surfboard_std_volume * 1.08

elif surferAge >= 51 and surferAge <= 60:      # Third threshold: Age between 51 and 60

    resultAge = surfboard_std_volume *  1.20

elif surferAge >= 61:                          # Fourth threshold: Age above 61

    resultAge = surfboard_std_volume *  1.30

print("The result of volume multiplied by the Age coefficient is : ", resultAge)

# STEP 6:
# Multiply the corresponding Fitness coefficient:

if surferFitness == 1:
    surfboard_real_volume = resultAge * 1.20

elif surferFitness == 2:
    surfboard_real_volume = resultAge * 1.10

elif surferFitness == 3:
    surfboard_real_volume = resultAge * 1.05

elif surferFitness == 4:
    surfboard_real_volume = resultAge * 1

print("*** The real surfboard volume for this surfer is : ", surfboard_real_volume, "Litres ***")

# STEP 7:
### APPROACH 3: PREDICT THE SURFBOARD LENGTH.

print("||| PREDICT THE SURFBOARD LENGTH (using MLP REGRESSOR) |||")

# Define the features (X1) and targets (y1):
X1 = data[['Surfer_Height','Surfing_Level_N','Surfer_Weight','Surfboard_Real_Volume']]
y1 = data['Surfboard_Length']

while True:

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1)

    #  1.2 Prediccion:

    mlr = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), max_iter=400, random_state=3)
    mlr.fit(X1_train, y1_train)
    print("Prediction accuracy : ",mlr.score(X1_train, y1_train))
    if mlr.score(X1_train, y1_train) > 0.80:
      break

# Make the prediction with MLPR:

# we only need to enter a new feature "Surfer´s Height", because the rest of the features we already entered before. We will use the Weight, Level and Volume that we entered in the previous part.
surferHeight = input("Enter the surfer Height (cm) : ")
surfboard_length = mlr.predict([[surferHeight,surferLevel,surferWeight,surfboard_real_volume]])  ## Here we enter the value of Surfer level and Weight in order to predict the surfboard volume.
print("*** Based on the surfer's Level, Weight, Height and the real surfboard volume, the most suitable surfboard Length for this surfer is: ",surfboard_length, "Feet (ft) *** ")

### APPROACH 4: PREDICT THE SURFBOARD SHAPE.
# We will use as features the surfboard volume and length previously obtained, and also surfer Level and the size of wave.

print("||| PREDICT THE SURFBOARD SHAPE (using RANDOM FOREST CLASSIFIER) |||")
X2 = data[['Surfboard_Real_Volume','Surfboard_Length','Surfing_Level_N','Wave_Size']]
y2 = data['Surfboard_Shape_N']

# Train the data:
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,test_size=0.2,random_state=42)

# Make the prediction with Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X2_train,y2_train)
print("The Prediction accuracy with RANDOM FOREST was: ",rfc.score(X2_test,y2_test)) #  print the prediction accuracy

waveSize = input("Enter the wave size [1 to 12 (ft)] : ")
surfboard_shape = rfc.predict([[surfboard_real_volume,surfboard_length,surferLevel,waveSize]])  ## Here we enter the value of Surfer level and Weight in order to predict the surfboard volume.
surfboard_shape_name = surfboardShape_dict[surfboard_shape[0]]  # Access the dictionary created previously, and extract the corresponding name of the surfboard.
print("*** Based on the surfer's Level, surfboard volume, surfboard length and the size of the wave, the surfboard Shape for this surfer is: ",surfboard_shape_name," ***")


# Print the final result of the ideal Surfboard for the surfer:

print ("Based on all made predictions and calculations, the ideal surfboard for this surfer is: ")
print("-Surfboard Volume: ",surfboard_real_volume ," Litres (L)")
print("-Surfboard Length: ",surfboard_length," Feet (ft)")
print("-Surfboard Shape or Model: ",surfboard_shape_name)

if surfboard_shape_name =="NOT SUITABLE":
    print("The size of wave is too big for the level of this person. This person can not surf such a wave")