import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
cm = plt.cm.get_cmap('Spectral')


def main(dpath):
    
    d = None
    for i, fpath in enumerate(glob.glob(os.path.join(dpath, '*.objects.txt'))):
        if os.path.basename(fpath) == 'timeseries_master.objects.txt':
            continue
        if os.path.basename(fpath) == 'series_master.objects.txt':
            continue
        x = pd.read_csv(fpath, header=0, sep='\t')
        x['tray'] = int(os.path.basename(fpath).split('_')[1].replace('tray', ''))
        d = pd.concat([d, x])
    
    
    d = d.sort_values(['label', 'tray'])
    
    
    
    # area
    fig = plt.figure(figsize=(8, 4), dpi=250)
    ax = fig.add_subplot()
    for plant_id in pd.unique(d.loc[:, 'label']):
        if d.loc[d.loc[:, 'label'] == plant_id, 'tray'].shape[0] < 10:
            continue
        ax.plot(d.loc[d.loc[:, 'label'] == plant_id, 'tray'],
                np.log10(d.loc[d.loc[:, 'label'] == plant_id, 'area']),
                label=plant_id,
                color=cm(plant_id / len(pd.unique(d.loc[:, 'label']))))
    ax.set_xlabel('Tray')
    ax.set_ylabel('log10(area)')
    ax.legend(ncol=3, fontsize=8)
    fig.savefig('PPD2013_outputs_area.png', bbox_inches='tight', pad_inches=0)
    
 
    # area (individual)
    fig = plt.figure(figsize=(11, 11), dpi=250)
    for plant_id in pd.unique(d.loc[:, 'label']):
        if d.loc[d.loc[:, 'label'] == plant_id, 'tray'].shape[0] < 10:
            continue
        ax = fig.add_subplot(6, 4, plant_id)
        ax.plot(d.loc[d.loc[:, 'label'] == plant_id, 'tray'],
                np.log10(d.loc[d.loc[:, 'label'] == plant_id, 'area']),
                label=plant_id, c='k')
        ax.set_ylim(2.8, 5.5)
        ax.legend()
    fig.savefig('PPD2013_outputs_area_plant.png', bbox_inches='tight', pad_inches=0)
 

    # color
    fig = plt.figure(figsize=(8, 4), dpi=250)
    ax = fig.add_subplot()
    for plant_id in pd.unique(d.loc[:, 'label']):
        if d.loc[d.loc[:, 'label'] == plant_id, 'tray'].shape[0] < 10:
            continue
        ax.plot(d.loc[d.loc[:, 'label'] == plant_id, 'tray'],
                d.loc[d.loc[:, 'label'] == plant_id, 'HSV;1'] * 180,
                label=plant_id,
                color=cm(plant_id / len(pd.unique(d.loc[:, 'label']))))
    ax.set_xlabel('Tray')
    ax.set_ylabel('Hue')
    ax.legend(ncol=3, fontsize=8)
    fig.savefig('PPD2013_outputs_colorhue.png', bbox_inches='tight', pad_inches=0)
    
    
    # color (individual)
    fig = plt.figure(figsize=(11, 11), dpi=250)
    for plant_id in pd.unique(d.loc[:, 'label']):
        if d.loc[d.loc[:, 'label'] == plant_id, 'tray'].shape[0] < 10:
            continue
        ax = fig.add_subplot(6, 4, plant_id)
        ax.plot(d.loc[d.loc[:, 'label'] == plant_id, 'tray'],
                d.loc[d.loc[:, 'label'] == plant_id, 'HSV;1'] * 180,
                label=plant_id, c='k')
        ax.set_ylim(100, 160)
        ax.legend()
    fig.savefig('PPD2013_outputs_colorhue_plant.png', bbox_inches='tight', pad_inches=0)
 
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    
    dpath = 'detection_results'
    main(dpath)
    


