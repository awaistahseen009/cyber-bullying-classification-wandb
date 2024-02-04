from keras.models import load_model
from data_ingest import DataIngestion
from data_preprocessing import DataPreprocessing
ingest_data=DataIngestion()
df=ingest_data.get_data('cyberbullying_tweets.csv')
print('***********Data has been ingested***********')
print("---------------------------------------------")
'''
DATA PREPROCESSING CLASS IS USED TO PREPROCESS THE DATA AND MAKE
IT READY FOR THE MODEL
'''
clean_data=DataPreprocessing(df)
clean_data.rename_cols()
clean_data.remove_null_values()
clean_data.remove_duplicates()
clean_data.preprocess_dataframe()
clean_data.encode_target()
clean_data.prepare_training_data()
clean_data.split_data(test_size=0.1)
model=load_model('model.keras')
print('***********Data has been cleaned***********')
print("---------------------------------------------")
'''
EVALUATING THE TESTING DATA TO TEST THE PERFORMANCE OF MODEL
'''
print('***********Model Evaulation has started***********')
print("---------------------------------------------")
_, x_test, _, y_test= clean_data.get_split_data()
with open('evaluate.txt', 'w') as file:
    print(f'Model Evaluation: {model.evaluate(x_test, y_test)}', file=file)
print("---------------------------------------------")
print('***********Model Evaulation has finished***********')
