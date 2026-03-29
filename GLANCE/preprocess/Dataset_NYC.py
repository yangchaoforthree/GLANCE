import numpy as np
import torch
import torch.utils.data

class EventData(torch.utils.data.Dataset):
    """ Spatial-temporal event dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time-location pair and other invidual features
        """
        self.unique_SUSP_RACE_list = ['ASIAN / PACIFIC ISLANDER', 'BLACK HISPANIC', 'WHITE HISPANIC', 
                                      'AMERICAN INDIAN/ALASKAN NATIVE', 'WHITE', 'BLACK']
        self.unique_SUSP_SEX_list = ['F', 'M']
        self.time = [inst['time_location_idx_pair'][0] for inst in data]
        self.precinct = [inst['time_location_idx_pair'][1] for inst in data]
        self.race = [self.unique_SUSP_RACE_list.index(inst['SUSP_RACE']) for inst in data]
        self.sex = [self.unique_SUSP_SEX_list.index(inst['SUSP_SEX']) for inst in data]

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.precinct[idx], self.race[idx], self.sex[idx]



def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dl
