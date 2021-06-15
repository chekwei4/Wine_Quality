# Red Wine Quality
The dataset is related to red variant of the Portuguese "Vinho Verde" wine. For more details, consult the reference [Cortez et al., 2009]. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc)

# Data source
https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

https://archive.ics.uci.edu/ml/datasets/wine+quality

# Introduction
Data set contains physicochemical of 1599 records of red wines. Initiially, dataset included a dependent feature [quality], which is a score between 0 to 10. The higher the score, the better the wine. However, most of the scores are concentrated between 5 or 6, 

Project has turned this into a classification problem, by setting an arbitrary cutoff for [quality] at 7 or higher getting classified as 'good/1' and the remainder as 'bad/0'.

# Objective
Aim to detect the few good wines among an imbalanced dataset using various sampling methods belows and observe which yield better recall score for minority class. 

1. Random Over Sampling

2. SMOTE + Tomek Links

3. SMOTE with random under sampling of majority class

4. ADASYN

Logistic regression model is used throughout the project.  

# Exploratory Data Analysis

Key analysis: 

1. Quality is dominated by values 5 and 6

![image info](./images/quality_cnts.png)

2. Features are in different scales, hence scaling is required. 

![image info](./images/scale_1.png)

3. Good correlation between alcohol and quality

![image info](./images/alcohol_quality.png)

4. High values of sulphates, alchohol, citric acid do lead to better quality  

![image info](./images/barplot_1.png)

5. Better quality wines are mostly above 0.2, 0.4 citric acid, with lower pH and higher fixed acidity

![image info](./images/newplot.png)
