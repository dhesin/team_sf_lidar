import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt


def plot_filtered_pr_curve(pr_history, val_history, outdir, maxgap=0.03):
    pr_df = pd.DataFrame(data=pr_history)
    vh_df = pd.DataFrame(data=val_history)
    pr_df['recall_grp'] = (pr_df['recall']/maxgap).astype(int)*maxgap
    vh_df['val_recall_grp'] = (vh_df['val_recall'] / maxgap).astype(int) * maxgap
    aggregation_pr = {'recall': {
                             'mean_recall': 'mean'
                             },
                   'precision': {'max_precision': lambda x: max(x),
                                 'min_precision': lambda x: min(x),
                                 'mean_precision': 'mean'
                                 }
                   }
    aggregation_vh = {
                        'val_recall': {
                                       'mean_val_recall': 'mean'
                                       },
                        'val_precision': {'max_val_precision': lambda x: max(x),
                                          'min_val_precision': lambda x: min(x),
                                          'mean_val_precision': 'mean'
                                          }
                    }
    grouped_recall_with_precision = pr_df.groupby('recall_grp').agg(aggregation_pr)
    grouped_vh_recall_with_precision = vh_df.groupby('val_recall_grp').agg(aggregation_vh)
    plt.clf()
    plt.plot(grouped_recall_with_precision[('recall', 'mean_recall')],
             grouped_recall_with_precision[('precision', 'max_precision')], lw=2, color='darkcyan', label='Max Precision')
    plt.plot(grouped_recall_with_precision[('recall', 'mean_recall')],
             grouped_recall_with_precision[('precision', 'mean_precision')], lw=2, color='c', label='Mean Precision')
    plt.plot(grouped_recall_with_precision[('recall', 'mean_recall')],
             grouped_recall_with_precision[('precision', 'min_precision')], lw=2, color='paleturquoise', label='Min Precision')

    plt.plot(grouped_vh_recall_with_precision[('val_recall', 'mean_val_recall')],
             grouped_vh_recall_with_precision[('val_precision', 'max_val_precision')], '--', dashes=[30,5,30,5], lw=2, color='peru',
             label='Max Val Precision')
    plt.plot(grouped_vh_recall_with_precision[('val_recall', 'mean_val_recall')],
             grouped_vh_recall_with_precision[('val_precision', 'mean_val_precision')], '--', dashes=[20,5,30,5], lw=2, color='burlywood',
             label='Mean Val Precision')
    plt.plot(grouped_vh_recall_with_precision[('val_recall', 'mean_val_recall')],
             grouped_vh_recall_with_precision[('val_precision', 'min_val_precision')], '--', dashes=[40,5,10,5], lw=2, color='wheat',
             label='Min Val Precision')

    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Filtered Precision-Recall Curve, filterGap={0}'.format(maxgap))
    plt.legend(loc="upper left")
    fname = '{0}/precision_recall_curve_filtered.png'.format(outdir)
    plt.savefig(fname)
    print 'Filtered Precision-Recall Curve saved at ' + fname


def plot_pr_curve(loss_history, outdir):
    data = np.array(zip(loss_history.precisions, loss_history.recalls, loss_history.losses,
                        loss_history.val_precisions, loss_history.val_recalls, loss_history.val_losses),
                        dtype =[('precision', float), ('recall', float), ('loss', float),
                        ('val_precision', float), ('val_recall', float), ('val_loss', float)])
    #data = loss_history #for test
    data_filename = 'precision_recall_value.csv'
    np.savetxt('{0}/{1}'.format(outdir,data_filename), data, delimiter=",")
    data = pd.DataFrame(data)
    pr_history = data[['precision','recall']]
    pr_history = pr_history.dropna()
    #pr_history = pr_history.sort_values('recall', axis=0, ascending=True)

    val_history = data[['val_precision','val_recall']]
    val_history = val_history.dropna()
    val_history = val_history.sort_values('val_recall', axis=0, ascending=True)


    print 'Plotting Precision-Recall Curve...'

    plt.clf()
    plt.plot(pr_history['recall'], pr_history['precision'], '--', dashes=[15, 2, 5, 2], lw=2, color='navy',
             label='Precision-Recall curve', alpha=0.5)
    plt.scatter(val_history['val_recall'], val_history['val_precision'], alpha=1.0, s = 50,color='darkcyan', label='Precision-Recall Validation')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper left")

    fname = '{0}/precision_recall_curve.png'.format(outdir)
    plt.savefig(fname)
    print 'Precision-Recall Curve saved at '+ fname
    #plt.show()

    plot_filtered_pr_curve(pr_history, val_history, outdir)


def test():

    outdir = '/home/teamsf/workplace/output'
    data_filename = 'precision_recall_value_100_complete.csv'
    fname = '{0}/{1}'.format(outdir, data_filename)  # precision_recall_value_100epoch
    df = pd.read_csv(fname, header=None, names=['precision', 'recall','losses','val_precision','val_recall','val_loss'])

    plot_pr_curve(df,outdir)

    #print df

if __name__ == "__main__":
    test()