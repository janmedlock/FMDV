#!/usr/bin/python3

from matplotlib import gridspec, pyplot, ticker
from matplotlib.backends import backend_pdf
import numpy
import pandas
import seaborn

import h5
import herd
import herd.samples
import stats


# Nature
rc = {}
# Sans-serif, preferably Helvetica or Arial.
rc['font.family'] = 'sans-serif'
rc['font.sans-serif'] = 'DejaVu Sans'


def _get_extinction(infected):
    t = infected.index.get_level_values('time (y)')
    time = t.max() - t.min()
    observed = (infected.iloc[-1] == 0)
    return {'extinction_time': time,
            'extinction_observed': observed}


def _load_extinction_times():
    with h5.HDFStore('run_samples.h5', mode='r') as store:
        df = []
        for model in ('acute', 'chronic'):
            for SAT in (1, 2, 3):
                by = ['model', 'SAT', 'sample']
                columns=['exposed', 'infectious', 'chronic']
                where = f'model={model} & SAT={SAT}'
                print(where)
                extinction = {}
                for (ix, group) in store.groupby(by, where=where,
                                                 columns=columns):
                    infected = group.sum(axis='columns')
                    extinction[ix] = _get_extinction(infected)
                extinction = pandas.DataFrame.from_dict(extinction,
                                                        orient='index')
                extinction.index.names = by
                samples = herd.samples.load(model=model, SAT=SAT)
                samples.index = extinction.index
                df.append(pandas.concat([extinction, samples],
                                        axis='columns', copy=False))
        return pandas.concat(df, axis='index', copy=False)


def load_extinction_times():
    try:
        df = h5.load('plot_samples.h5')
    except OSError:
        df = _load_extinction_times()
        h5.dump(df, 'plot_samples.h5')
    return df


def _get_labels(name, rank):
    base = name.replace('_', ' ')
    if not rank:
        label = base
        label_resid = 'Residual ' + base.lower()
    else:
        label = 'Rank ' + base.lower()
        label_resid = 'Residual rank ' + base.lower()
    return (label, label_resid)


def plot_times(df):
    groups = df.groupby(['SAT', 'model'])
    palette = seaborn.color_palette('Paired', len(groups))
    colors = {}
    for k in groups.groups.keys():
        (SAT, model) = k
        i = 2 * (SAT - 1)
        if model == 'chronic': i += 1
        colors[k] = palette[i]
    with seaborn.color_palette(palette):
        fig, ax = pyplot.subplots()
        for ((SAT, model), group) in groups:
            survival = stats.get_survival(group,
                                          'extinction_time',
                                          'extinction_observed')
            ax.step(survival.index, survival, color=colors[(SAT, model)],
                    where='post', label=f'SAT {SAT}, {model} model')
        ax.set_xlabel('time (y)')
        ax.set_ylabel('Survival')
        # ax.set_yscale('log')
        # Next smaller power of 10.
        # a = numpy.ceil(numpy.log10(1 / len(df)) - 1)
        # ax.set_ylim(10 ** a, 1)
        ax.legend()
        fig.savefig('samples_times.pdf')


def plot_parameters(df, rank=True, marker='.', s=1, alpha=0.6):
    outcome = 'extinction_time'
    models = df.index.get_level_values('model').unique()
    SATs = df.index.get_level_values('SAT').unique()
    params = df.columns.drop([outcome, 'extinction_observed'])
    (ylabel, ylabel_resid) = _get_labels(outcome, rank)
    colors = seaborn.color_palette('tab20', 20)
    # Put dark colors first, then light.
    colors = colors[0::2] + colors[1::2]
    with backend_pdf.PdfPages('samples_parameters.pdf') as pdf:
        for model in models:
            for SAT in SATs:
                X = df.loc[(model, SAT, slice(None)), params]
                X = X.dropna(axis='columns', how='all')
                y = df.loc[(model, SAT, slice(None)), outcome]
                if rank:
                    X = (X.rank() - 1) / (len(X) - 1)
                    y = (y.rank() - 1) / (len(y) - 1)
                fig, axes = pyplot.subplots(X.shape[1], 2)
                for (i, (param, x)) in enumerate(X.items()):
                    color = colors[i]
                    (xlabel, xlabel_resid) = _get_labels(param, rank)
                    axes[i, 0].scatter(x, y,
                                       color=color, marker=marker, s=s,
                                       alpha=alpha)
                    Z = X.drop(columns=param)
                    x_res = stats.get_residuals(Z, x)
                    y_res = stats.get_residuals(Z, y)
                    axes[i, 1].scatter(x_res, y_res,
                                       color=color, marker=marker, s=s,
                                       alpha=alpha)
                    for j in (0, 1):
                        axes[i, j].tick_params(pad=0, labelsize='small')
                    axes[i, 1].yaxis.set_ticks_position('right')
                    axes[i, 1].yaxis.set_label_position('right')
                    for (j, l) in enumerate([xlabel, xlabel_resid]):
                        axes[i, j].set_xlabel(l, labelpad=0, size='small')
                # Single ylabel per column.
                for (x, l, ha) in [[0, ylabel, 'left'],
                                   [1, ylabel_resid, 'right']]:
                    fig.text(x, 0.5, l, size='small', rotation=90,
                             horizontalalignment=ha, verticalalignment='center')
                fig.suptitle(f'{model.capitalize()} model, SAT {SAT}',
                             y=1, size='medium')
                w = 0.02
                fig.tight_layout(pad=0, h_pad=0, w_pad=0,
                                 rect=[w, 0, 1 - w, 0.98])
                pdf.savefig(fig)


class Colors:
    '''Add colors to dictionary dynamically as requested.'''

    def __init__(self):
        palette = seaborn.color_palette('tab20', 20)
        # Put dark colors first, then light.
        self._palette = palette[0::2] + palette[1::2]
        self._colors = {}

    def __getitem__(self, k):
        if k not in self._colors:
            self._colors[k] = self._palette.pop(0)
        return self._colors[k]



param_transforms = {
    'chronic_recovery_mean': 'chronic_duration_mean',
    'chronic_recovery_mean': 'chronic_duration_shape',
    'progression_mean': 'latent_duration_mean',
    'progression_shape': 'latent_duration_shape',
    'recovery_mean': 'infectious_duration_mean',
    'recovery_shape': 'infectious_duration_shape',
    'transmission_rate': 'acute_transmission_rate'
}


def plot_sensitivity(df, rank=True, errorbars=False):
    outcome = 'extinction_time'
    # models = df.index.get_level_values('model').unique()
    # Do chronic mdoel first to get the same colors as in Figure 4.
    models = ('chronic', 'acute')
    SATs = df.index.get_level_values('SAT').unique()
    samples = df.index.get_level_values('sample').unique()
    params = df.columns.drop([outcome, 'extinction_observed'])
    n_samples = len(samples)
    colors = Colors()
    width = 390 / 72.27
    height = 0.8 * width
    rc_ = rc.copy()
    rc_['figure.figsize'] = (width, height)
    rc_['xtick.labelsize'] = 7
    rc_['ytick.labelsize'] = 7
    rc_['axes.labelsize'] = 8
    rc_['axes.titlesize'] = 9
    for model in models:
        rho = pandas.DataFrame(index=params, columns=SATs, dtype=float)
        if errorbars:
            columns = pandas.MultiIndex.from_product((SATs,
                                                      ('lower', 'upper')))
            rho_CI = pandas.DataFrame(index=params, columns=columns,
                                      dtype=float)
        for SAT in SATs:
            p = df.loc[(model, SAT, slice(None)), params]
            p = p.dropna(axis='columns', how='all')
            o = df.loc[(model, SAT, slice(None)), outcome]
            if rank:
                rho[SAT] = stats.prcc(p, o)
                xlabel = 'PRCC'
                if errorbars:
                    rho_CI[SAT] = stats.prcc_CI(rho[SAT], n_samples)
            else:
                rho[SAT] = stats.pcc(p, o)
                xlabel = 'PCC'
                if errorbars:
                    rho_CI[SAT] = stats.pcc_CI(rho[SAT], n_samples)
        rho.dropna(axis='index', how='all', inplace=True)
        if errorbars:
            rho_CI.dropna(axis='index', how='all', inplace=True)
        # Sort rows on mean absolute values.
        order = rho.abs().mean(axis='columns').sort_values().index
        rho = rho.loc[order]
        if errorbars:
            rho_CI = rho_CI.loc[order]
        xabsmax = rho.abs().max().max()
        y = range(len(rho))
        ylabels = [param_transforms.get(p, p)
                                   .capitalize()
                                   .replace('_', ' ')
                   for p in rho.index]
        ncols = len(SATs)
        with pyplot.rc_context(rc_):
            fig = pyplot.figure(constrained_layout=True)
            gs = gridspec.GridSpec(1, ncols, figure=fig)
            axes = numpy.empty(ncols, dtype=object)
            axes[0] = None  # Make sharey work for axes[0].
            for col in range(ncols):
                # Share the y scale.
                sharey = axes[0]
                axes[col] = fig.add_subplot(gs[0, col],
                                            sharey=sharey)
            for ((SAT, rho_SAT), ax) in zip(rho.items(), axes):
                colors_ = [colors[z] for z in order[::-1]][::-1]
                if errorbars:
                    rho_err = pandas.DataFrame({
                        'lower': rho[SAT] - rho_CI[(SAT, 'lower')],
                        'upper': rho_CI[(SAT, 'upper')] - rho[SAT]}).T
                    kwds = dict(xerr=rho_err.values,
                                error_kw=dict(ecolor='black',
                                              elinewidth=1.5,
                                              capthick=1.5,
                                              capsize=5,
                                              alpha=0.6))
                else:
                    kwds = dict()
                ax.barh(y, rho_SAT, height=1, left=0,
                        align='center', color=colors_, edgecolor=colors_,
                        **kwds)
                ax.set_xlabel(xlabel)
                ax.set_xlim(- xabsmax, xabsmax)
                ax.set_ylim(- 0.5, len(rho) - 0.5)
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
                ax.yaxis.set_tick_params(which='both', left=False, right=False,
                                         pad=85)
                ax.set_title(f'SAT{SAT}')
                if ax.is_first_col():
                    ax.set_yticks(y)
                    ax.set_yticklabels(ylabels, horizontalalignment='left')
                else:
                    ax.yaxis.set_tick_params(which='both',
                                             labelleft=False, labelright=False)
                    ax.yaxis.offsetText.set_visible(False)
                for sp in ('top', 'left', 'right'):
                    ax.spines[sp].set_visible(False)
            fig.savefig(f'samples_sensitivity_{model}.pdf')


if __name__ == '__main__':
    df = load_extinction_times()
    # plot_times(df)
    # plot_parameters(df)
    plot_sensitivity(df)
    pyplot.show()
