import mlflow
import wandb
import datetime
from wandb.keras import WandbCallback
class TrackExperiment:
    def __init__(self, api_key,wandb_flag=True ) -> None:
        self.wandb_api_key=api_key
        self.wandb=None
        self.mlflow = mlflow
        self.experiment_name=None
        if wandb_flag:
            self.wandb=wandb
            self.wandb.login(key=self.wandb_api_key)
        

    
    def init_project(self , project_name:str, version=None):
        if self.wandb is not None:
            if version is not None:
                wandb.init(project=project_name+str(version))
            else:
                wandb.init(project=project_name)
        else:
            print('You cannot initialize the project , First initialize the wandb')

    def set_wandb_config(self, config:dict):
        self.wandb.config=config

    def get_wandb_config(self)->dict:
        return self.wandb.config
    
    def stop_wandb_run(self):
        self.wandb.finish()
    def return_callback(self):
        return WandbCallback()
    def init_ml_run(self,experiment_name):
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f'{experiment_name}_{current_datetime}'
        self.mlflow.create_experiment(experiment_name)
        
    def run_mlflow_experiment(self):
        self.mlflow.set_experiment(experiment_name=self.experiment_name)
        self.mlflow.start_run()
    def save_log_model(self,model):
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        mlflow.sklearn.save_model(model,f"model_best_v2_resample_{current_datetime}")
        self.mlflow.keras.log_model(model, self.experiment_name)

    def evaluate_model(self,x_test,y_test):
        last_run = self.mlflow.last_active_run().info.run_id
        eval_data=x_test.copy()
        eval_data['left']=y_test
        mlflow.evaluate(
        f'runs:/{last_run}/{self.experiment_name}',
            data=eval_data,
            targets='left',
            model_type='classifier'
            )