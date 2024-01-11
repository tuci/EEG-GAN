import numpy as np
import matplotlib.pyplot as plt

def results_plot(readFile_df,readFile_df1,file_names):
    color=list(iter(plt.cm.tab20(np.linspace(0,1,16))))
    f, axes=plt.subplots(2,5,sharex=True, figsize=(20,10))
    for i,files in enumerate(readFile_df):
        c1=color[i]
        c2=color[i+1]
        c3=color[i+2]

        if file_names[i].strip('.xlsx')=='WGAN_WC':
            axes.flatten()[i].plot(files[0][1:],files[1][1:]/min(files[1][1:]),label='D', color=c1)
        else:
            axes.flatten()[i].plot(files[0][1:],files[1][1:]/max(files[1][1:]),label='D', color=c1)
        axes.flatten()[i].plot(files[0][1:],readFile_df1[i][1][1:]/max(readFile_df1[i][1][1:]),label='GP', color=c3)
        axes.flatten()[i].plot(files[0][1:],files[2][1:]/max(files[2][1:]),label='G', color=c2)
        axes.flatten()[i].set_title(file_names[i].strip('.xlsx'),fontsize=10)
        axes.flatten()[i].legend(loc='best')

    f.delaxes(axes.flatten()[9])
    f.text(0.52,0.04,'Epoch', ha='center',fontsize=13)
    f.text(0.09,0.5,'Loss', ha='center', rotation=90, fontsize=13)
    #plt.savefig('loss_scores_graph.pdf')
    plt.show()