# FEEL THE RHYTHM
## OVERVIEW
You’ll be challenged to build a model that can predict when incidents are more likely to occur, with explainability being the key!
Because explainability is so important, in this challenge you will be scored on your leaderboard position, and a report that explains your model and your predictions.

## DESCRIPTION
### Background
The teams at Western Power perform a range of tasks across their operations, and they are looking to better understand what factors contribute to an increased likelihood of safety incidents occurring.

This challenge is **specifically designed to look at the impact of body clocks, circadian rhythms and work schedules on the likelihood of incidents.**

### Challenge
In this challenge you are asked to build and submit a model that can predict when incidents are more likely to occur, and provide an explanation of your findings and predictions in a report.

> This problem has been framed as hourly classification. For a given employee, working on a given hour, did an incident occur or not?

## DATA
You are also allowed to use additional open source data (e.g. circadian rhythm models, weather data), to aid your explanations and insights, but not in training your model. *If you would like to use open source data to train your model, please contact one of our team.*

### Main Challenge Data Set
For the leaderboard, the dataset has been split into training, validation (public leaderboard) and test (private leaderboard).

The total dataset consists of 20394936 hours worked, 821 of which contain incidents.

> The dataset split was done by employees, and stratified by incidents

Please note that this dataset was not split by time, as causality is not required. Data recording and timesheet processes have also changed overtime. However there are enough incidents to make a model that predicts incidents for unseen employees, and generate model driven insights.

### Training data
12354494 records, each one representing an hour, with 490 of those hours containing incidents. The dataset consists of approximately 60% of records from the overall sample.

### Validation (Public Leaderboard)
We have withheld 3915000 sample records with 156 incidents. These records correspond to unseen employees. The dataset consists of approximately 20% of records from the overall sample.

### Validation (Private Leaderboard)
We have withheld 4125442 sample records with 175 incidents. These records correspond to unseen employees. The dataset consists of approximately 20% of records from the overall sample.

### Data Dictionnary
1. Work_DateTime: The hour for this row
2. EmpNo_Anon: Anonomised Employee identifier
3. Incident_Number: Can be used to tie incidents to the auxiliary data, for your report
4. TIME_TYPE: Overtime or normal
5. WORK_NO: Code for work type
6. WORK_DESC: Full description of work type, e.g. Admin
7. FUNC_CAT: Operational/Support/Network or Asset
8. TOT_BRK_TM: Minutes of break time, averaged per hour in shift

Target - Incident: True if an incident occurred False otherwise

## METHOD
![](https://imgur.com/tA6FTDz.png)