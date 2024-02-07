from data_ingest import DataIngestion
from data_preprocessing import DataPreprocessing
from experiment_tracking import TrackExperiment
from model import build_model
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("API_KEY")
'''
DATA INGESTION CLASS IS USED TO GET THE DATA AND MAKE IT
READY IN DATAFRAME FORMAT
'''
ingest_data=DataIngestion()
df=ingest_data.get_data('data\cyberbullying_tweets.csv')
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
x_train, x_test, y_train, y_test =clean_data.get_split_data()
print('***********Data has been cleaned***********')
print("---------------------------------------------")
'''
SETTING UP EXPERIMENT_TRACKING
'''
tracking=TrackExperiment(api_key=api_key)
tracking.init_project(project_name='Cyberbullying-Classification-ML',version=1)
config={
'batch_size':128,
'epochs':15,
'latent_dim':64,
'embedding_dim':100,
'optimizer':'rmsprop',
}
tracking.set_wandb_config(config)
print('***********Experiment has been set***********')
print("---------------------------------------------")
'''
MODEL DEFINITION
'''
vocab_size=clean_data.get_vocab_size()
target_values=clean_data.target_count()
max_len=clean_data.max_len
model=build_model(config['embedding_dim'], config['latent_dim'], vocab_size+1 ,max_len, target_values)
print('***********Building model finished***********')
print("---------------------------------------------")
model.compile(optimizer=config['optimizer'], loss='categorical_crossentropy' , metrics=['accuracy'] )
early_s = EarlyStopping(monitor='val_loss',verbose=1,patience=10)
'''
TRAINING THE MODEL
'''
print('***********Model training starting***********')
print("---------------------------------------------")
model.fit(x_train, y_train,
          epochs=config['epochs'] , 
          batch_size=config['batch_size'],
          validation_split=0.2,
          callbacks=[early_s, tracking.return_callback()]
         )
print('***********Model training finished***********')
print("---------------------------------------------")
'''
STOPPING THE RUN
'''

print('***********WANDB DATA HAS STARTED UPLOADING***********')
print("---------------------------------------------")
tracking.stop_wandb_run()
print('***********WANDB RUN HAS BEEN FINSHED AND DATA HAS BEEN UPLOADED***********')
print("---------------------------------------------")

model.save('model.keras')