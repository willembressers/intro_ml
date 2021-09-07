# -*- coding: utf-8 -*-

# python core
import math

# 3rd party modules
import sklearn
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# use a more white template for all plotly plots
pio.templates.default = "plotly_white"

def histograms(df:pd.DataFrame, cols:int=4, height:int=600) -> None:
    # count how many rows we need (given the nr of columns)
    rows = math.ceil(len(df.columns) / cols)

    # initialize a figure
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=df.columns)

    # add a histogram as a subplot for each column
    for index, column in enumerate(df.columns):

        # get the current row & column number
        col = (index % cols) + 1
        row = (index // cols) + 1
        
        # leave a trace
        fig.add_trace(go.Histogram(x=df[column], name=column), row=row, col=col)

    # show it
    fig.update_layout(height=height, title_text=f"Column histograms", showlegend=False)
    fig.show()

def boxplots(df:pd.DataFrame, title:str, threshold:int) -> None:
    # initialize a figure
    fig = go.Figure()

    # loop over the columns
    for index, column in enumerate(df.columns):
        if df[column].max() < threshold:
            fig.add_trace(go.Box(x=df[column], name=column))

    # show it
    fig.update_layout(height=600, title_text=title, showlegend=False)
    fig.show()

def confusion_matrix(y_true, y_pred, colorscale='Blues'):
    # collect all the labels
    labels = list(np.unique(np.concatenate((y_true, y_pred))))

    # get the confusion matrices (absolute and normalized)
    cm_abs = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize=None, labels=labels)
    cm_norm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='true', labels=labels)

    # create the annotated heatmaps
    fig1 = ff.create_annotated_heatmap(cm_abs, labels, labels,  colorscale=colorscale)
    fig3 = ff.create_annotated_heatmap(cm_norm, labels, labels,  colorscale=colorscale)

    # add the heatmaps to the figure
    fig = make_subplots(subplot_titles=('Absolute values', 'Normalized values'), rows=1, cols=2, horizontal_spacing=0.075)
    fig.add_trace(fig1.data[0], row=1, col=1)
    fig.add_trace(fig3.data[0], row=1, col=2)

    # fiddle with the annotations
    newfont = [go.layout.Annotation()] * len(fig.layout.annotations)
    fig_annots = [newfont, fig1.layout.annotations, fig3.layout.annotations]
    for j in range(2, len(fig_annots)):
        for k  in range(len(fig_annots[j])):
            fig_annots[j][k]['xref'] = f'x{j}'
            fig_annots[j][k]['yref'] = f'y{j}'
    new_annotations = __recursive_extend(fig_annots[::-1], len(fig_annots))

    # now finalize the output
    fig.update_layout(annotations=new_annotations,  title_text="Confusion matrixes")
    fig.update_xaxes(title_text="predicted label", row=1, col=1)
    fig.update_xaxes(title_text="predicted label", row=1, col=2)
    fig.update_yaxes(title_text="true label", row=1, col=1)
    fig.update_yaxes(title_text="true label", row=1, col=2)
    fig.show()


def __recursive_extend (mylist, nr):
    #mylist is a list of lists
    # initial nr =len(mylist)
    result = []
    
    if nr> 1:
        result.extend(mylist[nr-1])
        result.extend(__recursive_extend(mylist, nr-1))
    else: 
        result.extend(mylist[nr-1])
    
    return result

def multiple_time_series(df:pd.DataFrame, x, y, color, title) -> None:
    fig = px.line(df, x=x, y=y, color=color, title=title)
    fig.show()

def time_series(ts:pd.DataFrame, title) -> None:
    fig = px.line(ts, x=ts.index, y=ts, title=title)
    fig.show()

# def class_distribution(y_train, class_names, title='', threshold=1000):
#     # count per value
#     values, counts = np.unique(y_train, return_counts=True)

#     # map the values to labels
#     labels = [class_names[value] for value in values]

#     # create a dataframe (so i can sort) and then plot the value
#     df = pd.DataFrame({'labels':labels, 'counts':counts}).sort_values('counts', ascending=True).set_index('labels')

#     # define a subplot
#     fig, ax = plt.subplots(figsize=(20, 15))
    
#     # plot the bars
#     bars = ax.barh(df.index, df['counts'])
    
#     # draw the vertical threshold line
#     ax.axvline(x=threshold, color='red', linewidth=0.8, linestyle="--", label='threshold')

#     # remove the borders
#     for spine in plt.gca().spines.values():
#         spine.set_visible(False)

#     # loop over the bars
#     for bar in bars:
#         # get the bar value
#         label = bar.get_width()
        
#         # determine the label y position
#         label_y_pos = bar.get_y() + bar.get_height() / 2
        
#         # add the label
#         ax.text(label, label_y_pos, s=f'{label:.0f}', va='center', ha='right', fontsize=15, color='white')
        
#         # color the bars 
#         if label > threshold:
#             bar.set_color('green')
#         else:
#             bar.set_color('orange')

#     plt.title(title)
#     plt.xlabel('Nr images')
#     plt.ylabel('Class label')
#     plt.legend()


# def history(history, threshold):
#     plt.figure(figsize=(22, 10))

#     # summarize history for accuracy
#     plt.subplot(3, 1, 1)
#     plt.plot(history.history['accuracy'], label='Training')
#     plt.plot(history.history['val_accuracy'], label='Validation')
#     plt.axhline(y=threshold, color='red', linewidth=0.8, linestyle="--", label='threshold')
#     plt.legend(loc='lower right')
#     plt.title('Training & Validation history')
#     plt.ylabel('Accuracy')
#     plt.ylim([0,1.0])

#     # summarize history for loss
#     plt.subplot(3, 1, 2)
#     plt.plot(history.history['loss'], label='Training')
#     plt.plot(history.history['val_loss'], label='Validation')
#     plt.legend(loc='upper right')
#     plt.ylabel('Loss')
#     plt.ylim([0,1.0])

#     # summarize history for learning rate
#     plt.subplot(3, 1, 3)
#     plt.plot(history.history['lr'])
#     plt.ylabel('Learning rate')
#     plt.xlabel('Epoch')


# def evaluations(df, threshold = 0.9):
#     # define a subplot
#     fig, ax = plt.subplots(figsize=(20, 5))
    
#     # plot the bars
#     bars = ax.barh(df.index, df['accuracy'])
    
#     # draw the vertical threshold line
#     ax.axvline(x=threshold, color='red', linewidth=0.8, linestyle="--", label='threshold')

#     # remove the borders
#     for spine in plt.gca().spines.values():
#         spine.set_visible(False)

#     # loop over the bars
#     for bar in bars:
#         # get the bar value
#         label = bar.get_width()
        
#         # determine the label y position
#         label_y_pos = bar.get_y() + bar.get_height() / 2
        
#         # add the label
#         ax.text(label, label_y_pos, s=f'{label:.3f}', va='center', ha='right', fontsize=15, color='white')
        
#         # color the bars 
#         if label > threshold:
#             bar.set_color('green')
#         else:
#             bar.set_color('orange')

#     plt.title('Accuracy per dataset')
#     plt.xlabel('Accuracy')
#     plt.ylabel('Dataset')
#     plt.legend()