import pandas as pd 

class DataIngestion:
    def __init__(self )->None:
        self.file_path=None
        self.data_frame=None
    
    def get_data(self, file_path:str)->pd.DataFrame:
        self.file_path=file_path
        self.data_frame=pd.read_csv(self.file_path)
        return self.data_frame

    def show_data(self, num_rows:int)->pd.DataFrame:
        return self.data_frame.head(num_rows)