# CS3244

This is a team project made for the NUS CS3244 Machine Learning module, AY2020/21 Semester 1.
We look at data from 25,000+ football matches from major European leagues from 2008-2016, and attempt to predict the match results purely using attributes that would be available before the start of a match. 

How to use:
1) Download data from https://www.kaggle.com/hugomathien/soccer and convert all databases to CSV files. All the CSV files are already available here except for Match.csv, which exceeds the GitHub file size limit. Place them in the same directory as this Git repository.
2) Depending on the data cleaning process required, open "player_model.py", "player_team_model.py" or "player_model + time.py".
	"player_model.py" builds model using 2 x 11 player ratings as the attributes. 
	"player_team_model.py" builds model using 2 x 11 player ratings and 2 x 9 continuous team chracteristics as the attributes. 
	"player_model + time.py" builds model using 2 x 11 player ratings as the attributes, after preprocessing to find most recent player rating to occur before the match date.
3) When prompted, enter the supervised ML model you wish to train. Descriptions of models are given when running the program.
4) When prompted, enter the y-label to obtain after processing the data.
	"wl (win-loss)" removes all instances of draws from the data. A home win is labeled as 1 and an away win is labeled as 0.
	"wdl (win-draw-loss)" labels home win as 1, draws as 0, away win as -1.
	"gd (goal difference)" calculates (home goals scored minus away goals scored) as the label.
5) Wait for program to output results. You can adjust the "multicore" parameter within the .py files to enable/disable multicore processing when doing cross validation (disabled by default)