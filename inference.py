
import json
import logging
import os, sys
import shutil
from argparse import ArgumentParser
from datetime import datetime
import pandas as pd
import requests
import time
import numpy as np

import torch
import torchvision
from torch.nn.parallel import DistributedDataParallel
from torch.nn.functional import softmax


from ignite.metrics import Accuracy, Loss

from monai.data import (
    CacheDataset,
    DataLoader
)
from monai.engines import SupervisedEvaluator
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine
)
from monai.inferers import SimpleInferer
from monai.transforms import (
    AsDiscreted,
    Compose,
    LoadImaged,
    AsChannelFirstd,
    ToTensord,
    SqueezeDimd,
    Resized,
    DataStatsd,
    AddChanneld,
    RepeatChanneld,
    EnsureTyped,
    CopyItemsd
)

import data_preparation

def getEnvVarOrDefault(var, default):
    envVar = os.environ.get(var)
    if(envVar is None):
        return default
    return envVar


class acrCOVIDinference:

    def __init__(self, args):

        self.weights = args.weights
        self.gpu = args.gpu
        self.job_id = args.job_id
        self.report_url = args.report_url

        #list for appending raw model outputs for json file
        self.raw_preds = []

        self.input_dir = '/input'
        self.output_dir = '/output'
        self.scratch_dir = '/scratch'

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(self.scratch_dir):
            os.mkdir(self.scratch_dir)

        model_name = 'acrCOVIDclassifier'
        start_time = round(time.time())
        logpath = os.path.join(self.output_dir, model_name + '_' + str(start_time) + '.log')
        file_handler = logging.FileHandler(filename=logpath)
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers = [file_handler, stdout_handler]
        logging.basicConfig(
            level=logging.DEBUG, 
            handlers=handlers
        )
        logging.info('Started: %s', start_time)


    def report_progress(self, progress):
        # print(progress)
        try:
            data = {'progress': progress ,'jobId':self.job_id }
            r = requests.post(url = self.report_url,json = data )
        except Exception as ex:
            logging.error("Exception occured while posting final progress api url: "+self.report_url	+" for job-id: "+ self.job_id + "with progress: "+ str(progress),ex)


    def main(self):

        try:
            self.report_progress(1)
            test_datalist = self.preprocess(self.input_dir, self.output_dir, self.scratch_dir)
            self.report_progress(33)
            test_engine = self.configure(test_datalist) #need to be self?
            self.report_progress(66)
            self.predict(test_engine)
            self.report_progress(99)
            logging.info('Finished')
            progress_json = {'jobId': self.job_id, 'jobStatus': 'complete'}
            #requests.post(self.reportURL, data=progress.json)


        except Exception as e:

            logging.error("Exception")
            logging.error('Critical failure, stopping container. Timestamp: %s Error: %s', time.time(), e)
            logging.error(e)
            progress_json = {'jobId': self.job_id, 'jobStatus': 'failure'}
            #requests.post(self.reportURL, data=progress.json)
        finally :
            try:
                shutil.rmtree(self.scratch_dir)    
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (self.scratch_dir, e))
         


    def ignite_accuracy_conversion(self, x):
        pred = [batch["pred"] for batch in x]
        label = [batch["label"].to(self.device) for batch in x]
        pred = torch.stack(pred)
        label = torch.stack(label)
        return (pred, label)

    def save_test_json(self, engine):

        #get raw model output and convert to softmax
        preds = sum(self.raw_preds, [])
        preds = torch.stack(preds)
        preds = softmax(preds, dim=1)
        pred_list = [preds[i][1].item() for i in range(len(preds))]

        #get study uids
        inputs = engine.state.dataloader
        studyuids = [batch["studyuid"] for batch in inputs]
        studyuids = sum(studyuids, [])

        #add data to output.json
        test_json = {
           "usecase": 'covid19-compatible-chest-radiograph-pattern',
           "modelname": 'acrCOVIDclassifier',
           "studies": []
           }
        studies = test_json['studies']
        for instance in  zip(pred_list, studyuids):
            studies.append({
                'studyInstanceUID': instance[1],
                'pred': instance[0]
                 })

        #ensure one prediction per study
        predictions_list = []
        df = pd.DataFrame(studies)
        studies = pd.unique(df['studyInstanceUID'])
        for study in studies:
            study_pred = np.max(df.loc[df['studyInstanceUID'] == study]['pred'].to_numpy())
            predictions_list.append({
                "studyInstanceUID": str(study),
                "classificationOutput": [{
                "key": "codableConcept_covid19detection", # covid19 classification key
                "output": {
                "1" : study_pred
                          }
                   }]
                })
        test_json['studies'] = predictions_list

        with open(os.path.join(self.output_dir, 'output.json'), 'w') as f:
            json.dump(test_json, f)
            f.close()


    def preprocess(self, input_dir, output_dir, scratch_dir):

        logging.info('Begining preprocessing')

        preprocessed_files = data_preparation.preprocess(input_dir, output_dir, scratch_dir)
        datalists = data_preparation.generate_data_splits(preprocessed_files, 'inference')
        test_datalist = datalists['test']

        logging.info('Finished preprocessing')

        return test_datalist

    def configure(self, test_datalist):

        logging.info('Configuring inference parameters')

        if args.gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        network = torchvision.models.resnet50()
        num_ftrs = network.fc.in_features
        network.fc = torch.nn.Linear(num_ftrs, 2)
        if self.weights:
            logging.info('WEIGHTS')
            logging.info('type weights: ', self.weights)
            logging.info('type statedict: ', type(network.state_dict()))
            network.load_state_dict(torch.load(self.weights, map_location=self.device))
            logging.info('loaded weights')
        network.to(self.device)

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Resized(keys=["image"], spatial_size=(224,224)),
                RepeatChanneld(keys=["image"], repeats=3)
            ]
        )

        post_transform = EnsureTyped(keys=["pred"], data_type="tensor", device=self.device)


        test_ds = CacheDataset(
            data=test_datalist,
            transform=test_transforms,
            cache_rate=1.0,
            num_workers=4,
        )

        test_data_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )

        test_handlers = [
                StatsHandler(
                name="test_engine",
                tag_name="test_json",
                iteration_print_logger=lambda x: self.raw_preds.append([batch["pred"] for batch in x.state.output]),
                output_transform=lambda x: None)
            ]


        test_engine = SupervisedEvaluator(
            device=self.device,
            val_data_loader=test_data_loader,
            network=network,
            inferer=SimpleInferer(),
            postprocessing=post_transform,
            val_handlers=test_handlers,
            amp=True,
        )

        return test_engine

    def predict(self, test_engine):

        logging.info('Running inference')

        test_engine.run()
        self.save_test_json(test_engine)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--weights', type=str, default=getEnvVarOrDefault('weights','model_key_metric_0.9171.pt'))
    parser.add_argument('--gpu', type=int, default=getEnvVarOrDefault('gpu',None))
    parser.add_argument('--job_id', type=str, default =getEnvVarOrDefault('job_id', 'job0'))
    parser.add_argument('--report_url', type=str, default=getEnvVarOrDefault('report_url', 'https://foo.bar'))
    args = parser.parse_args()

    inferer = acrCOVIDinference(args)
    inferer.main()
