# League-of-Legends-data-analysis
A data analysis using League of Legends e-sports match data from 2022, a final project for EECS 398 at the University of Michigan.

## Introduction
In this project, we are utilizing League of Legends esports match data from the year 2022, which is a comprehensive collection of player-level statistics from professional matches around the world. Our project attempts to answer the question: Does getting the first blood increase a team’s odds of winning a match?
In a high-stakes competition like LoL esports, momentum can significantly influence the match results. First blood is one of the earliest and most impactful events in a match, and understanding its correlation with victory can offer insight into whether early aggression is a reliable path to winning. 
After filtering the dataset to remove missing values and keep only relevant early-game features, our final working dataset contains 12,872 rows and 10 columns, which provide figures for in-game statistics. The most important one amongst these, and the focus of our analysis, is firstblood, which is a binary value of true or false based on whether or not the players teams achieved the first kill in the match.  

## Data Cleaning and Exploratory Data Analysis:
We cleaned the dataset by focusing on a subset of early-game performance columns relevant to our question. Since we were interested in predicting win/loss based on early game dynamics, we selected 10 key columns such as firstblood, killsat10, and golddiffat10.
Upon inspection, some rows had missing values, likely due to incomplete or truncated game recordings. We removed these rows to avoid introducing bias through imputation or incorrect assumptions. This left us with a clean dataset of 12,872 rows and no missing values.

The histogram below shows how often teams secured first blood. The distribution is clearly skewed, since most players in the dataset were on teams that did not get first blood. This suggests that first blood is a relatively rare event and reinforces its potential importance as a high-impact moment in a game.

<iframe 
src="assets/fig1_firstblood.html" 
width="800" 
height="600" 
frameborder="0"
></iframe>

We also visualized the distribution of golddiffat10, which shows a roughly symmetric bell shape around 0. This suggests that most teams are close in gold at the 10-minute mark, but some start to pull ahead or fall behind:

<iframe 
src="assets/fig2_gold_diff.html" 
width="800" height="600" 
frameborder="0"
></iframe>

The chart below shows the relationship between securing first blood and winning the match. While teams that did not get first blood still won frequently, teams that did get first blood were more likely to win than lose. This supports our hypothesis that early kills can tilt a game’s momentum, though it is not a guaranteed path to victory.

<iframe 
src="assets/fig3_firstblood_vs_result.html" 
width="800" 
height="600" 
frameborder="0"
></iframe>

The box plot below shows that teams who won had higher average gold differences at 10 minutes than those who lost. This suggests that even by 10 minutes, a gold lead is a good indicator of future success: 

<iframe 
src="assets/fig4_gold_diff_by_result.html" 
width="800" 
height="600" 
frameborder="0"
></iframe>

The table below shows the average win rate for teams based on whether they secured first blood:

|   First Blood |   Average Win Rate |
|--------------:|-------------------:|
|             0 |           0.456228 |
|             1 |           0.609333 |

Teams that earned first blood won approximately 61% of their matches, compared to only 46% for those that didn’t. This supports the intuition that first blood provides a significant early-game advantage.

After inspecting the early-game columns relevant to our analysis, we found that some rows contained missing values. Rather than imputing these values, we chose to drop rows with any missing data using dropna(). This decision was based on the fact that the dataset is large (over 150k rows originally), and we retained over 12,000 clean observations after filtering. We avoided imputation because early-game statistics like firstblood, golddiffat10, and killsat10 are context-dependent and not safe to estimate. Imputing these values might introduce bias or noise into our classification model, especially since these stats are tied closely to in-game events that don't always occur consistently (e.g., some players never place wards or secure kills by 10 minutes).
As a result, no imputation plots are necessary, and our cleaned dataset contains only rows with fully observed early-game metrics.

##  Framing a Prediction Problem:

The prediction problem we address is: "Can we predict whether a professional League of Legends team will win a match using only early-game statistics available by the 10-minute mark?" This is a binary classification problem where the response variable is:
result: 1 if the player's team won the match, 0 if they lost.
We selected this variable because it directly reflects the competitive outcome of the game, and understanding what leads to victory is highly relevant to players, analysts, and fans. The features we used for prediction (e.g., firstblood, golddiffat10, killsat10, assistsat10) are all known within the first 10 minutes of gameplay — ensuring the model does not "peek into the future."
We evaluated model performance using accuracy, which is an appropriate metric in our case because the dataset is perfectly balanced between wins and losses (50/50). Precision and recall are not as critical here since both false positives and false negatives carry similar consequences in this esports context.
This model could hypothetically be used mid-game to assess win probability based on early dynamics, which helps analysts understand what factors actually tilt the odds in a team’s favor.

## Baseline Model
We trained a baseline logistic regression model using two early-game features:
firstblood (binary, nominal): Whether the player's team got the first kill.
golddiffat10 (numeric, quantitative): The gold lead or deficit at 10 minutes.
No categorical encoding was needed, since firstblood is already numeric (0 or 1), and golddiffat10 is a continuous feature. Both features were scaled using StandardScaler as part of a single scikit-learn Pipeline, which also included a LogisticRegression classifier.
We split the data 80/20 for training and testing, and evaluated performance using accuracy, since the dataset is balanced.
The baseline model achieved an accuracy of 61.7%, with a balanced precision and recall of approximately 0.61–0.62 across both win and loss classes. This confirms that early gold leads and first blood carry predictive weight, but additional features are needed to improve confidence and reliability. 

## Final Model
To improve upon the baseline, we added two new early-game features:
xpdiffat10: Experience point difference at 10 minutes — reflects which team is gaining more levels early
csdiffat10: Minion kill difference at 10 minutes — an indicator of farming efficiency and lane control
These features are relevant to predicting match outcome because they indicate map pressure, laning success, and early resource control, which often snowball into a win in professional League of Legends. We applied different preprocessing steps for different features:
golddiffat10: Scaled with StandardScaler
xpdiffat10: Transformed with QuantileTransformer to reduce skew and better fit a tree model
firstblood and csdiffat10: Left as-is (binary and small-scale numeric)
We trained a RandomForestClassifier using a Pipeline with a ColumnTransformer, and tuned hyperparameters using GridSearchCV with 5-fold cross-validation. The parameters we tuned were:
Number of trees (n_estimators)
Tree depth (max_depth)
Minimum samples to split (min_samples_split)
Our final model achieved an accuracy of 61.8%, slightly improving on the baseline’s 61.7%. While the numerical increase is small, the use of additional early-game features (xpdiffat10, csdiffat10) and hyperparameter tuning led to a better-balanced and more informed model. These features represent aspects of in-game control beyond just kills and gold, such as experience gain and farming advantage. Our best-performing model was a RandomForestClassifier tuned using GridSearchCV across three parameters. The confusion matrix for our final model shows the following:
    True Positives (Predicted Win, Actual Win): 7,957
    True Negatives (Predicted Loss, Actual Loss): 7,859
    False Positives (Predicted Win, Actual Loss): 4,929
    False Negatives (Predicted Loss, Actual Win): 4,830
This means the model correctly predicted approximately 62% of both wins and losses. While it made some incorrect predictions in both directions, the overall distribution is fairly balanced, suggesting that the model is not biased toward either outcome. This aligns with the model’s F1-scores of 0.62 for both classes and confirms that our feature set captures meaningful early-game signals for classification.
