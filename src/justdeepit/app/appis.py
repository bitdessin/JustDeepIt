import os
import logging
import justdeepit
from justdeepit.app import OD

logger = logging.getLogger(__name__)       


class IS(OD):
    
    def __init__(self, workspace):
        super().__init__(workspace)
        self.app = 'IS'
    

    def train_model(self, *args, **kwargs):
        return super().train_model(*args, **kwargs)

    
    def detect_objects(self, *args, **kwargs):
        return super().detect_objects(*args, **kwargs)


    def summarize_objects(self, *args, **kwargs):
        return super().summarize_objects(*args, **kwargs)
