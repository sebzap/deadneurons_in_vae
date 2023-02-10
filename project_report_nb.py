import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from typing import List
import numpy as np
import torch
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
import pickle


# https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


def loadTensorboard(dir:str):
    df = None
    for dirname in os.listdir(dir):
        f = os.path.join(dir, dirname)

        df_run = tflog2pandas(f)
        info = dirname.split('_')
        if 'leaky' in info:
            df_run["model_type"] = "LeakyReLU"
        else:
            df_run["model_type"] = "ReLU"
        df_run['vae_model'] = info[0]
        df_run['model_name'] = info[1]
        df_run['seed'] = info[6]

        if df is None:
            df = df_run
        else:
            df = df.append(df_run, ignore_index=True)

    return df

config = {
    "total_epochs": 50
}

class TextHandler(HandlerBase):
        def create_artists(self, legend, tup ,xdescent, ydescent,
                            width, height, fontsize,trans):
            tx = Text(width/2.,height/2,tup[0], fontsize=fontsize,
                    ha="center", va="center", color=tup[1], fontweight="bold")

            return [tx]

class DeadNeuronViz():
    def __init__(self, square_heatmap, colors, titles, tick_offset, step_modifier_dict):
        self.square_heatmap = square_heatmap
        self.colors = colors
        self.titles = titles
        self.tick_offset = tick_offset
        self.step_modifier_dict = step_modifier_dict
        self.weights = ['encoder.main.0.weight [32, 1, 4, 4]',
            'encoder.main.2.weight [32, 32, 4, 4]',
            'encoder.main.4.weight [64, 32, 4, 4]',
            'encoder.main.6.weight [64, 64, 4, 4]',
            'encoder.main.9.weight [256, 1024]',
            'encoder.main.11.weight [20, 256]',
            'decoder.main.0.weight [256, 10]',
            'decoder.main.2.weight [1024, 256]',
            'decoder.main.5.weight [64, 64, 4, 4]',
            'decoder.main.7.weight [64, 32, 4, 4]',
            'decoder.main.9.weight [32, 32, 4, 4]',
            'decoder.main.11.weight [32, 1, 4, 4]']
        self.total_epochs=50
        self.total_iterations=2160*self.total_epochs

    def load(self, dir, tag):
        with open(dir+'/deadNeurondata.'+tag+'.pkl', 'rb') as handle:
            return pickle.load(handle)


    def renderHeatmaps(self, data: List[torch.Tensor], axs: List[plt.Axes], isConv: bool):
        for j in range(3):
            d = data[j].cpu().numpy()
            ax = axs[j]

            if j == 2:
                d = d.sum(0)

            if len(d.shape) > 3:
                d = np.concatenate(d,axis=1)
                d = np.concatenate(d,axis=1)
            
            
            vmax = self.total_epochs if j == 2 else self.total_iterations
            _colors = self.colors + '_r' if j == 2 else self.colors
            sns.heatmap(d,cmap=sns.color_palette(_colors, as_cmap=True), vmin=0, vmax=vmax, ax=ax, square=self.square_heatmap)
            ax.set(title=self.titles[j])
            if isConv:
                ax.set(xlabel="Input Channels", ylabel="Output Channels")
            else:
                ax.set(xlabel="Input Dimension", ylabel="Output Dimension")
            
            if isConv:
                self.setConvHeatmapTicks(ax, d.shape)

    def setConvHeatmapTicks(self, ax, shape):
        n_ticks = shape[0]//4
        step_modifier = self.step_modifier_dict[n_ticks]
        ax.set_yticks(range(self.tick_offset,self.tick_offset+n_ticks*4,4*step_modifier))
        ax.set_yticks(range(self.tick_offset,self.tick_offset+n_ticks*4,4), minor=True)
        ax.set_yticklabels(range(0,n_ticks,step_modifier))

        n_ticks = shape[1]//4
        step_modifier = self.step_modifier_dict[n_ticks]
        ax.set_xticks(range(self.tick_offset,self.tick_offset+n_ticks*4,4*step_modifier))
        ax.set_xticks(range(self.tick_offset,self.tick_offset+n_ticks*4,4), minor=True)
        ax.set_xticklabels(range(0,n_ticks,step_modifier))

    
    def renderWeightTypeDistribution(self, data: List[torch.Tensor], ax: plt.Axes):
        types = ["never dead","never alive","died and stayed dead","started dead, revived","died then revived (once)","started dead, revived, died","other"] 
        handltext = ['ND', 'NA', 'DIED', 'REV', "REV1", "DRD", "OTH"]
        
        df = self.getWeightTypeDistribution(data)
        sns.countplot(x=df["type_shorthand"], ax=ax, order=handltext)
        ax.set(title="Weights by Training Behaviour", xlabel="Behaviour", ylabel="weight count")

        # https://stackoverflow.com/a/43591678/5168770
        t = ax.get_xticklabels()
        handles = [(h.get_text(),c.get_fc()) for h,c in zip(t,ax.patches)]
        ax.legend(handles, types, handler_map={tuple : TextHandler()}) 

        return df

    def getWeightTypeDistribution(self, data: List[torch.Tensor]):
        types = ["never dead","never alive","died and stayed dead","started dead, revived","died then revived (once)","started dead, revived, died","other"] 
        handltext = ['ND', 'NA', 'DIED', 'REV', "REV1", "DRD", "OTH"]
        
        weight_sequences = []
        weight_epoch_data = data[2].view((50,-1)).numpy().transpose()
        for weightIndex, epochData in enumerate(weight_epoch_data):
            sequences = []

            curIsDead = False
            curCount = 0
            for epoch_dead in epochData:
                if epoch_dead == curIsDead:
                    curCount += 1
                else:
                    if curCount > 0:
                        sequences.append((curIsDead, curCount))
                    curCount = 1
                    curIsDead = epoch_dead

            if curCount > 0:
                sequences.append((curIsDead, curCount))
            weight_sequences.append(sequences)

        
        df_entries = []
        for weightIndex, sequences in enumerate(weight_sequences):
            neverDead = len(sequences) == 1 and sequences[0][0] == False
            neverAlive = len(sequences) == 1 and sequences[0][0] == True
            diedAndStayedDead = len(sequences) == 2 and sequences[-1][0] == True
            startedDeadButRevived = len(sequences) == 2 and sequences[0][0] == True and sequences[1][0] == False
            revivedOnce = (len(sequences) == 2 or len(sequences) == 3) and sequences[-2][0] == True and sequences[-1][0] == False
            starteDeadRevivedButDiedAgain = len(sequences) == 3 and sequences[0][0] == True and sequences[-2][0] == False and sequences[-1][0] == True
            other = not (neverAlive or neverDead or diedAndStayedDead or startedDeadButRevived or revivedOnce or starteDeadRevivedButDiedAgain)

            type_index=np.argmax([neverDead,neverAlive,diedAndStayedDead,startedDeadButRevived,revivedOnce,starteDeadRevivedButDiedAgain,other])

            df_entries.append({
                "index": weightIndex,
                "type": types[type_index],
                "type_shorthand": handltext[type_index],
                "neverDead": neverDead,
                "neverAlive": neverAlive,
                "diedAndStayedDead": diedAndStayedDead,
                "startedDeadButRevived": startedDeadButRevived,
                "revivedOnce": revivedOnce,
                "starteDeadRevivedButDiedAgain": starteDeadRevivedButDiedAgain,
                "other": other,
                "startedDead": sequences[0][0] == True,
                "epochsUntilStateChange": sequences[0][1],
            })

        df = pd.DataFrame.from_dict(df_entries)
       
        return df

    def heatmap_iteration(self, d: np.ndarray, ax: plt.Axes, isConv:bool, transposed: bool = False, title:str = None): 
        vmax = self.total_iterations
        _colors = self.colors
        sns.heatmap(d,cmap=sns.color_palette(_colors, as_cmap=True), vmin=0, vmax=vmax, ax=ax, square=self.square_heatmap)
        ax.set_title(title)
        if isConv:
            if transposed:
                ax.set(xlabel="Output Channels", ylabel="Input Channels")
            else:
                ax.set(xlabel="Input Channels", ylabel="Output Channels")
        else:
            if transposed:
                ax.set(xlabel="Output Dimension", ylabel="Input Dimension")
            else:
                ax.set(xlabel="Input Dimension", ylabel="Output Dimension")
        
        if isConv:
            n_ticks = d.shape[0]//4
            step_modifier = self.step_modifier_dict[n_ticks]
            ax.set_yticks(range(self.tick_offset,self.tick_offset+n_ticks*4,4*step_modifier))
            ax.set_yticks(range(self.tick_offset,self.tick_offset+n_ticks*4,4), minor=True)
            ax.set_yticklabels(range(0,n_ticks,step_modifier))

            n_ticks = d.shape[1]//4
            step_modifier = self.step_modifier_dict[n_ticks]
            ax.set_xticks(range(self.tick_offset,self.tick_offset+n_ticks*4,4*step_modifier))
            ax.set_xticks(range(self.tick_offset,self.tick_offset+n_ticks*4,4), minor=True)
            ax.set_xticklabels(range(0,n_ticks,step_modifier))
