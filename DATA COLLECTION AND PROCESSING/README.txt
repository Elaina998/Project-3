1-> Run Crawl page.ipynb. result: BZ=F_15year_daily_data.csv, DX-Y.NYB_15year_daily_data.csv

2-> Run BZ=F_15 year csv_Process.ipynb. result: Final brent_processed_data.csv

3-> Run DX-Y.NYB_15year_daily_data_Process.ipynb. result: Final USDX_processed_data.csv

4-> Run Sentiment_Cleaning.ipynb. result: SENT.csv

5-> Run Data Merging.ipynb. result:SENT_process.csv

6-> Run Data Cleaning. result: processed_sentiment_data.csv

7-> Copy processed_sentiment_data.csv to BEST SENTIMENT SELECTION Folder. Run BEST SENTIMENT.ipynb. result:processed_data_best_corr_sentiment.csv

8-> Open processed_data_best_corr_sentiment.csv, Change the Csum_CrudeBERT_Plus_GT to SENT and USDX Close to USDX

9-> Run Data Visualization.ipynb.

10-> At processed_data_best_corr_sentiment.csv Change Date -> date

11-> Copy the processed_data_best_corr_sentiment.csv to the main/dataset folder 