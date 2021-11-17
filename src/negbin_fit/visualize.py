import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
import seaborn as sns
from negbin_fit.fit_nb import get_p, make_negative_binom_density
from negbin_fit.helpers import read_weights, alleles, get_nb_weight_path, get_counts_dist_from_df, \
    make_cover_negative_binom_density, make_geom_dens, combine_densities, make_inferred_negative_binom_density
from scipy import stats as st

sns.set(font_scale=1.55, style="ticks", font="lato", palette=('#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
                                                              '#D55E00', '#CC79A7'))
# sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
plt.rcParams['font.weight'] = "medium"
plt.rcParams['axes.labelweight'] = 'medium'
plt.rcParams['figure.titleweight'] = 'medium'
plt.rcParams['axes.titleweight'] = 'medium'
plt.rcParams['figure.figsize'] = 18, 5
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams["legend.framealpha"] = 0.8


def local_read_weights(np_weights_dict, np_weights_path):
    r = {}
    w = {}
    gof = {}
    for fixed_allele in alleles:
        coefs_array = read_weights(np_weights_dict=np_weights_dict, np_weights_path=np_weights_path,
                                   allele=fixed_allele)
        r[fixed_allele] = coefs_array[:, 0]
        w[fixed_allele] = coefs_array[:, 1]
        gof[fixed_allele] = coefs_array[:, 3]
        # first_bad_gof = min(x for x in range(len(gof[fixed_allele])) if gof[fixed_allele][x] > 0.05)
        # gof[fixed_allele][first_bad_gof:] = 1
        # r[fixed_allele][BAD][first_bad_gof:] = 0
        # w[fixed_allele][BAD][first_bad_gof:] = 1
    return r, w, gof


def r_ref_bias(df_ref, df_alt, BAD, out, to_show=False):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.tight_layout(pad=2)

    # ax.set_xlim(allele_tr, 50)
    y_max = max(max(df_ref['r']), max(df_alt['r']))
    # ax.set_ylim(0, y_max * 1.05)
    ax.grid(True)

    ax.plot([0, y_max], [0, y_max], c='grey', label='y=x', linestyle='dashed')

    x = [row['r'] for index, row in df_alt.iterrows() if index in df_ref.index and row['r'] != 0]
    y = [row['r'] for index, row in df_ref.iterrows() if index in df_alt.index and row['r'] != 0]

    ax.scatter(x=x, y=y, color='C1')

    ax.set_xlabel('Fitted r value for fixed Alt')
    ax.set_ylabel('Fitted r value for fixed Ref')

    plt.title('BAD={}'.format(BAD))

    slope, intercept, r_value, p_value, std_err = st.linregress(x, y)
    ax.plot(x, np.array(x) * slope + intercept, color='#DC3220', label='y={:.2f} x {} {:.2f}'
            .format(slope, '+' if np.sign(intercept) > 0 else '-', abs(intercept)))

    plt.legend()
    plt.savefig(os.path.expanduser(out))
    if to_show:
        plt.show()
    plt.close(fig)


def r_vs_count_scatter(df_ref, df_alt,
                       out, BAD,
                       max_read_count=50,
                       allele_tr=5, to_show=False,
                       weights_dict=None
                       ):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.tight_layout(pad=2)

    ax.set_xlim(allele_tr, max_read_count)
    y_max = max(max(df_ref['r']), max(df_alt['r']))
    ax.set_ylim(0, y_max * 1.05)
    ax.grid(True)

    ax.plot([allele_tr, y_max], [allele_tr, y_max], c='grey', label='y=x', linestyle='dashed')
    if weights_dict is not None:
        line_plot_colors = {
            'alt': 'orange',
            'ref': 'red'
        }
        for allele in alleles:
            a = weights_dict[allele]['a']
            b = weights_dict[allele]['b']
            ax.plot([allele_tr, max_read_count], [a * allele_tr + b, a * max_read_count + b],
                    c=line_plot_colors[allele], label='Line fit for {}'.format(allele.upper()))
    # if BAD == 4 / 3:
    #     ax.plot([10 * 4 / 3, y_max * 4 / 3], [10, y_max], label='y=3/4 x', c='black', linestyle='dashed')

    x_alt, y_alt = zip(*([(x, y) for x, y in zip(df_alt.index, df_alt["r"].tolist()) if y != 0]))
    x_ref, y_ref = zip(*([(x, y) for x, y in zip(df_alt.index, df_ref["r"].tolist()) if y != 0]))

    ax.scatter(x=x_alt, y=y_alt, color='C1', label='Alt')
    ax.scatter(x=x_ref, y=y_ref, color='C2', label='Ref')

    ax.set_xlabel('Read count for the fixed allele')
    ax.set_ylabel('Fitted r value')

    ax.legend()

    plt.title('BAD={}'.format(BAD))

    plt.savefig(out)
    if to_show:
        plt.show()
    plt.close(fig)


def gof_scatter(df_ref, df_alt, BAD, out,
                max_read_count=50,
                allele_tr=5, to_show=False,
                weights_dict=None,
                ):
    # gof vs read cov
    df_ref = df_ref[(df_ref['gof'] > 0) & (df_ref.index <= max_read_count)]
    df_alt = df_alt[(df_alt['gof'] > 0) & (df_alt.index <= max_read_count)]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.tight_layout(pad=2)

    ax.set_xlim(allele_tr, max_read_count)
    ax.set_ylim(0, max(max(df_ref['gof']), max(df_alt['gof'])) * 1.05)
    ax.grid(True)

    ax.axhline(y=0.05, lw=2, linestyle='--', color='#505050')

    ax.scatter(x=df_alt.index,
               y=df_alt["gof"].tolist(),
               color='C1',
               label='Alt')
    ax.scatter(x=df_ref.index,
               y=df_ref["gof"].tolist(),
               color='C2',
               label='Ref')

    if weights_dict is not None:
        for allele, df, color in zip(alleles, [df_ref, df_alt], ['C4', 'C3']):
            x = df.index
            y = [weights_dict[allele]['point_gofs'].get(str(k), 0) for k in x]
            ax.scatter(x=x,
                       y=y,
                       color=color,
                       label=allele.capitalize() + ' new')

    ax.set_xlabel('Read count for the fixed allele')
    ax.set_ylabel('Goodness of fit, RMSEA')

    plt.title('BAD={}'.format(BAD))

    ax.legend(title='Fixed allele')

    plt.savefig(out)
    if to_show:
        plt.show()
    plt.close(fig)


def read_dfs(out):
    try:
        result = [pd.read_table(get_nb_weight_path(out, allele)) for allele in alleles]
    except Exception:
        raise AssertionError("No weight dfs found")
    return result


def make_image_path(out, image_name, image_type):
    return os.path.join(out, image_name + '.' + image_type)


def slices(df_ref, df_alt, stats_df, BAD, out,
           weights_dict=None,
           allele_tr=5,
           max_read_count=50,
           cover_list=None,
           to_show=False,
           ):
    if cover_list is None:
        cover_list = [10, 15, 20, 25, 30]
    lw = 1.25
    color_maxlog = 4
    p = get_p(BAD)

    t = stats_df.copy()
    t = t[(t['ref'] >= allele_tr) & (t['alt'] >= allele_tr)]
    t = t[(t['ref'] <= max_read_count) & (t['alt'] <= max_read_count)]
    for count in range(allele_tr, max_read_count + 1):
        if not t[(t['ref'] == count) & (t['alt'] == allele_tr)]['counts'].tolist():
            t = t.append(pd.DataFrame({'ref': [count], 'alt': [allele_tr], 'counts': [0]}))
        if not t[(t['ref'] == allele_tr) & (t['alt'] == count)]['counts'].tolist():
            t = t.append(pd.DataFrame({'ref': [allele_tr], 'alt': [count], 'counts': [0]}))
    t.reset_index(inplace=True, drop=True)

    print(t['counts'].sum(axis=0))

    t.rename(
        columns={
            'ref': 'Reference allele read count',
            'alt': 'Alternative allele read count',
        },
        inplace=True
    )
    t = t.pivot('Alternative allele read count', 'Reference allele read count', 'counts')
    t.sort_index(ascending=False, inplace=True)
    t.fillna(0, inplace=True)

    t = np.log10(t + 1)

    for cover in cover_list:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.tight_layout(pad=1.5)
        sns.heatmap(t, cmap="BuPu", ax=ax1, vmin=0, vmax=color_maxlog)

        cbar = ax1.collections[0].colorbar
        cbar.set_ticks(np.arange(0, color_maxlog + 1, 1))
        cbar.set_ticklabels(["10⁰", "10¹", "10²", "10³", "10⁴", "10⁵", "10⁶", "10⁷"])

        if max_read_count <= 50:
            div = 5
        elif max_read_count <= 100:
            div = 10
        elif max_read_count % 30 == 0:
            div = 30
        else:
            div = 50

        ax1.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0, max_read_count - allele_tr + 1, div) + 0.5))
        ax1.yaxis.set_major_formatter(ticker.FixedFormatter(range(div - allele_tr, max_read_count + 1)[::-div]))
        ax1.tick_params(axis="y", rotation=0)

        ax1.xaxis.set_major_locator(ticker.FixedLocator(np.arange(div - allele_tr, max_read_count + 1, div) + 0.5))
        ax1.xaxis.set_major_formatter(ticker.FixedFormatter(range(div, max_read_count + 1)[::div]))
        ax1.tick_params(axis="x", rotation=0)

        ax1.hlines(y=max_read_count + 1 - allele_tr, xmin=0, xmax=max_read_count + 1 - allele_tr, colors=['black', ], linewidth=lw * 2)
        ax1.vlines(x=0, ymin=0, ymax=max_read_count + 1 - allele_tr, colors=['black', ], linewidth=lw * 2)
        ax1.hlines(y=0, xmin=0, xmax=max_read_count + 1 - allele_tr, colors=['black', ], linewidth=lw * 2)
        ax1.vlines(x=max_read_count + 1 - allele_tr, ymin=0, ymax=max_read_count + 1 - allele_tr, colors=['black', ], linewidth=lw * 2)

        for cov, fix in zip([cover, cover], alleles.keys()):
            if fix == 'alt':
                ax1.axhline(xmin=0, y=max_read_count - cov + 0.5, linestyle='dashed', linewidth=lw, color='black')
            elif fix == 'ref':
                ax1.axvline(x=cov - allele_tr + 0.5, ymin=0, linestyle='dashed', linewidth=lw, color='black')

        # Params
        for fix_c, fixed_allele, ax in zip([cover, cover], alleles.keys(), (ax2, ax3)):
            main_allele = "ref" if fixed_allele == "alt" else "alt"
            stats = stats_df
            stats_filtered = stats[stats[fixed_allele] == fix_c].astype(int)
            max_cover_in_stats = max(stats_filtered[main_allele], default=10)
            counts_array = np.zeros(max_cover_in_stats + 1, dtype=np.int64)
            for index, row in stats_filtered.iterrows():
                k, SNP_counts = row[main_allele], row['counts']
                counts_array[k] = SNP_counts

            chop_counts_array = np.zeros(max_read_count + 1)
            chop_counts_array[:min(max_read_count, max_cover_in_stats) + 1] = counts_array[:min(max_read_count, max_cover_in_stats) + 1]

            total_snps = counts_array[0:max_cover_in_stats + 1].sum()
            x = list(range(max_read_count + 1))
            sns.barplot(x=x,
                        y=chop_counts_array / total_snps, ax=ax, color='C1')

            df = df_ref if fixed_allele == 'ref' else df_alt
            r, w, gof = (df['r'][fix_c],
                         df['w'][fix_c],
                         df['gof'][fix_c])
            print(r, w, gof, BAD, fix_c)
            if r == 0:
                col = 'C6'
                r = fix_c
            else:
                col = '#4d004b'

            current_density = np.zeros(max_read_count + 1)
            current_density[:min(max_read_count, max_cover_in_stats) + 1] = \
                make_negative_binom_density(r, p,
                                            w,
                                            max_cover_in_stats,
                                            left_most=allele_tr
                                            )[:min(max_read_count, max_cover_in_stats) + 1]
            ax.plot(sorted(x + [allele_tr]), [0] + list(current_density), color=col)

            if weights_dict is not None:
                current_lin_density = np.zeros(max_read_count + 1)
                neg_bin_dens1 = make_inferred_negative_binom_density(fix_c, weights_dict[main_allele]['r0'],
                                                                     weights_dict[main_allele]['p0'], p,
                                                                     max_cover_in_stats, allele_tr)
                neg_bin_dens2 = make_inferred_negative_binom_density(fix_c, 1,
                                                                     weights_dict[main_allele]['th0'], p,
                                                                     max_cover_in_stats, allele_tr)
                neg_bin_dens = (1 - weights_dict[main_allele]['w0']) * neg_bin_dens1 + weights_dict[main_allele]['w0'] * neg_bin_dens2
                current_lin_density[:min(max_read_count, max_cover_in_stats) + 1] = \
                    neg_bin_dens[:min(max_read_count, max_cover_in_stats) + 1]
                ax.plot(sorted(x + [allele_tr]), [0] + list(current_lin_density), color='red')
                label = 'negative binom fit for {}' \
                        '\ntotal observations: {}\nr={:.2f}, p={:.2f}, w={:.2f}\ngof={:.4}\ngof_red={:.4}'.format(main_allele,
                                                                                                   total_snps,
                                                                                                   r, p, w, gof,
                                                                                                   weights_dict[main_allele]['point_gofs'][str(fix_c)])
            else:
                label = 'negative binom fit for {}' \
                        '\ntotal observations: {}\nr={:.2f}, p={:.2f}, w={:.2f}\ngof={:.4}'.format(main_allele,
                                                                                                   total_snps,
                                                                                                   r, p, w, gof)
            ax.text(s=label, x=0.65 * fix_c, y=max(current_density) * 0.6)

            ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, max_read_count + 1, div)))
            ax.xaxis.set_major_formatter(ticker.FixedFormatter(range(0, max_read_count + 1)[::div]))
            ax.tick_params(axis="x", rotation=0)
            ax.set_xlabel('{} allele read count'.format({'ref': 'Reference', 'alt': 'Alternative'}[main_allele]))

        plt.suptitle('Ref bias BAD={:.2f}'.format(BAD))
        plt.savefig(out(cover))
        if to_show:
            plt.show()
        plt.close(fig)

    # # ref bias in r scatter
    #
    # df_ref = df_ref[(df_ref['gof'] <= 0.05)]
    # df_alt = df_alt[(df_alt['gof'] <= 0.05)]
    #
    # fig, ax = plt.subplots(figsize=(6, 5))
    # fig.tight_layout(pad=2)
    #
    # ax.set_xlim(allele_tr, 50)
    # y_max = max(max(df_ref['r']), max(df_alt['r']))
    # ax.set_ylim(0, y_max * 1.05)
    # ax.grid(True)
    #
    # ax.plot([allele_tr, y_max], [allele_tr, y_max], c='grey', label='y=x', linestyle='dashed')
    # if BAD == 4 / 3:
    #     ax.plot([10 * 4 / 3, y_max * 4 / 3], [10, y_max], label='y=3/4 x', c='black', linestyle='dashed')
    #
    # ax.scatter(x=df_alt.index, y=df_alt["r"].tolist(), color='C1', label='Alt')
    # ax.scatter(x=df_ref.index, y=df_ref["r"].tolist(), color='C2', label='Ref')
    #
    # ax.set_xlabel('Read count for the fixed allele')
    # ax.set_ylabel('Fitted r value')
    #
    # ax.legend(title='Fixed allele')
    #
    # plt.title('BAD={}'.format(BAD))
    #
    # plt.savefig(os.path.expanduser('~/AC_10/Figure_AS_10_r_scatter_{:.2f}.svg'.format(BAD)))
    # plt.close(fig)
    #
    #
    #

    # # ref bias in r scatter 2
    #
    # fig, ax = plt.subplots(figsize=(6, 5))
    # fig.tight_layout(pad=2)
    #
    # # ax.set_xlim(allele_tr, 50)
    # y_max = max(max(df_ref['r']), max(df_alt['r']))
    # # ax.set_ylim(0, y_max * 1.05)
    # ax.grid(True)
    #
    # ax.plot([0, y_max], [0, y_max], c='grey', label='y=x', linestyle='dashed')
    #
    # x = [row['r'] for index, row in df_alt.iterrows() if index in df_ref.index]
    # y = [row['r'] for index, row in df_ref.iterrows() if index in df_alt.index]
    #
    # ax.scatter(x=x, y=y, color='C1')
    #
    # ax.set_xlabel('Fitted r value for fixed Alt')
    # ax.set_ylabel('Fitted r value for fixed Ref')
    #
    # plt.title('BAD={}'.format(BAD_dict[BAD]))
    #
    # slope, intercept, r_value, p_value, std_err = st.linregress(x, y)
    # print(slope, intercept)
    # ax.plot(x, np.array(x) * slope + intercept, color='#DC3220', label='y={:.2f} x {} {:.2f}'
    #         .format(slope, '+' if np.sign(intercept) > 0 else '-', abs(intercept)))
    #
    # plt.legend()
    #
    # plt.savefig(os.path.expanduser('~/AC_10/Figure_AS_10_r_scatter_ref_alt_{:.2f}.svg'.format(BAD)))
    # plt.close(fig)


def main(stats, out, BAD,
         weights_dict=None,
         allele_tr=5,
         image_type='svg',
         to_show=False,
         cover_list=None,
         max_read_count=50,
         line_fit=False,
         ):
    if cover_list is None:
        cover_list = [5, 10]
    df_ref, df_alt = read_dfs(out)
    if not line_fit:
        gof_scatter(df_ref, df_alt,
                    out=make_image_path(out, 'gof', image_type),
                    BAD=BAD,
                    max_read_count=max_read_count,
                    allele_tr=allele_tr,
                    to_show=to_show)
        r_ref_bias(df_ref, df_alt,
                   out=make_image_path(out, 'r_bias', image_type),
                   BAD=BAD,
                   to_show=to_show)
        r_vs_count_scatter(df_ref, df_alt,
                           out=make_image_path(out, 'r_vs_counts', image_type),
                           BAD=BAD,
                           max_read_count=max_read_count,
                           allele_tr=allele_tr,
                           to_show=to_show)
        slices(df_ref, df_alt, stats_df=stats, BAD=BAD,
               allele_tr=allele_tr,
               to_show=to_show,
               max_read_count=max_read_count,
               cover_list=cover_list,
               out=lambda x: make_image_path(out, 'negbin_slices_{}'.format(x), image_type))
    else:
        # r_vs_count_scatter(df_ref, df_alt,
        #                    out=make_image_path(out, 'r_vs_counts.line_fit', image_type),
        #                    BAD=BAD,
        #                    allele_tr=allele_tr,
        #                    max_read_count=max_read_count,
        #                    weights_dict=weights_dict,
        #                    to_show=to_show)
        slices(df_ref, df_alt, stats_df=stats, BAD=BAD,
               allele_tr=allele_tr,
               to_show=to_show,
               max_read_count=max_read_count,
               cover_list=cover_list,
               weights_dict=weights_dict,
               out=lambda x: make_image_path(out, 'negbin_slices_with_inferred_fit_N_{}'.format(x), image_type))
        gof_scatter(df_ref, df_alt,
                    out=make_image_path(out, 'gof', image_type),
                    BAD=BAD,
                    max_read_count=max_read_count,
                    allele_tr=allele_tr,
                    to_show=to_show,
                    weights_dict=weights_dict,
                    )


def draw_cover_fit(stats_df, weights_dict, cover_allele_tr, max_read_count, BAD=1):
    fig, ax = plt.subplots()
    draw_cover_dist(stats_df, weights_dict,
                    ax=ax,
                    cover_allele_tr=cover_allele_tr,
                    max_read_count=max_read_count,
                    BAD=BAD)


def draw_barplot(x, y, ax, cover_array, r, p, w, th, frac, max_cover, max_read_count, cover_allele_tr, it='final', BAD=1, draw_rest=False):
    sns.barplot(x=x,
                y=y,
                ax=ax, color='C1')
    current_density = np.zeros(len(cover_array))
    current_density[:max_cover] = \
        combine_densities(make_cover_negative_binom_density(
            r,
            p,
            len(cover_array) - 1,
            left_most=cover_allele_tr,
            draw_rest=draw_rest,
        )[:max_cover],
            make_geom_dens(
            th,
            cover_allele_tr,
            len(cover_array) - 1,
        )[:max_cover],
        w, frac, 1 / (BAD + 1), allele_tr=5)
    ax.plot(sorted(x + ([cover_allele_tr] if not draw_rest else [])),
            ([0] if not draw_rest else []) + list(current_density[:max_cover]),
            color='C6')

    current_density = np.zeros(len(cover_array))
    current_density[:max_cover] = \
        combine_densities(make_cover_negative_binom_density(
            r,
            p,
            len(cover_array) - 1,
            left_most=cover_allele_tr,
            draw_rest=draw_rest,
        )[:max_cover],
            make_geom_dens(
            th,
            cover_allele_tr,
            len(cover_array) - 1,
        )[:max_cover],
        w, frac, 1 / (BAD + 1), allele_tr=5, only_negbin=True)
    ax.plot(sorted(x + ([cover_allele_tr] if not draw_rest else [])),
            ([0] if not draw_rest else []) + list(current_density[:max_cover]),
            color='C2')

    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, max_read_count + 1, 5)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(range(0, max_read_count + 1)[::5]))
    ax.tick_params(axis="x", rotation=0)
    # ax.set_xlabel('{} allele read count'.format({'ref': 'Reference', 'alt': 'Alternative'}[main_allele]))
    plt.savefig('D:\\Sashok\\Desktop\\fit_cover\\cover_fit_{}_{:.2f}_{:.2f}.png'.format(it, r, p))


def draw_cover_dist(stats_df, weights_dict, ax, cover_allele_tr, max_read_count, BAD=1):
    cover_array, max_cover, sum_counts, x, y = get_params_for_plot(stats_df, cover_allele_tr, max_read_count)
    draw_barplot(x, y, ax, cover_array, weights_dict['r0'], weights_dict['p0'], weights_dict['w0'], weights_dict['th0'], weights_dict['frac'], max_cover, max_read_count, cover_allele_tr, it='final', BAD=BAD)


def get_params_for_plot(stats_df, cover_allele_tr, max_read_count, draw_rest=False):
    cover_array = get_counts_dist_from_df(stats_df)
    max_cover = min(max_read_count, len(cover_array) - 1) + 1
    sum_counts = sum(cover_array[cover_allele_tr:max_cover])
    x = list(range(max_cover))
    y = [z / sum_counts if k >= cover_allele_tr or draw_rest else 0 for k, z in enumerate(cover_array[:max_cover])]
    return cover_array, max_cover, sum_counts, x, y


def get_callback_plot(cover_allele_tr, max_read_count, stats_df, BAD=1):
    cover_array, max_cover, sum_counts, x, y = get_params_for_plot(stats_df, cover_allele_tr, max_read_count)

    def callback(xk):
        callback.n += 1
        r, p, w, th, frac = xk
        print(r, p)
        fig, ax = plt.subplots()
        draw_barplot(x, y, ax, cover_array, r, p, w, th, frac, max_cover, max_read_count, cover_allele_tr, it=callback.n, BAD=BAD)
        plt.close(fig)

    callback.n = 0

    return callback
