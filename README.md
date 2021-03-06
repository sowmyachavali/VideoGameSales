# VideoGameSales
Video Game Sales - Exploratory Data Analysis 

Since the early 1980s, video games have served as an increasingly popular source of entertainment and a fixture in modern pop culture, evolving from humble beginnings to a multibillion-dollar industry today. Throughout its history, the video game market has evolved significantly, reflecting the continuous improvements to the underlying graphics and game designs as computer technology has become more powerful.  With each successive year, more advanced consoles have spawned a plethora of video game titles and genres, with thousands of titles available and new developers cropping up to try and take a share of the market. Although I have played many video games for years, I have never thought about which platforms, publishers, and genres are responsible for the bulk of the sales and quality of the games we all love to play. So here in this analysis, I would like to take a deep dive into the three categories mentioned above along with the regional sales. I'll focus on these questions to better direct the focus, and shape of the analysis.

Research questions –
1.	What are the sales of video games per genre?
2.	What are sales of video games per platform?
3.	Visualizations of platforms’ total sales and understanding each platform’s composition.
4.	What are the top 5 publishers with highest global sales?
5.	What is the growth of the top publisher of games over years?
6.	Generate correlation table and observe the interaction between variables.
7.	Observation on Year Wise Video Game Releases
8.	Understanding the global sales pattern
9.	How are the changes in sales over time?

Dataset overview -
I analyzed data scraped from VGChartz.com, consisting of over 16,500 video game titles based either on consoles or computers and ranging from 1980-2016. The data spanned across 31 platforms, 12 genres, and over 500 publishers.  
 
Dataset:
https://www.kaggle.com/gregorut/videogamesales?select=vgsales.csv

The fields include
•	Name - The games name
•	Platform - Platform of the games release (i.e. PC,PS4, etc.)
•	Year - Year of the game's release
•	Genre - Genre of the game
•	Publisher - Publisher of the game
•	NA_Sales - Sales in North America (in millions)
•	EU_Sales - Sales in Europe (in millions)
•	JP_Sales - Sales in Japan (in millions)
•	Other_Sales - Sales in the rest of the world (in millions)
•	Global_Sales - Total worldwide sales.
