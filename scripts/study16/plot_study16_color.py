#!/usr/bin/env python3
"""Recreate MAIC-Study16 figures using a restrained, consistent academic color palette.

Final-checkpoint figures use the 90 per-seed records. Learning trajectories
and checkpoint stability use original per-run JSON if --runs is available.
Never interpolate or reconstruct missing intermediate evaluation records.

    python3 plot_study16_color.py \
        --csv reports_study16/figures/per_seed_values.csv \
        --runs runs_study16 --out reports_study16/figures_academic
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ENVS = ('MTE16', 'PushBox16', 'PassSparse16')
ALGOS = ('COMA_Phi', 'MADDPG', 'MAIC', 'MAIC_ExactKL', 'MASAC', 'MASAC_ExactKL')
LABELS = ('COMA*', 'MADDPG', 'MAIC', 'MAIC + KL', 'MASAC', 'MASAC + KL')
ALGO_CAPTIONS = {
    'COMA_Phi': 'COMA-configured MAIC',
    'MADDPG': 'MADDPG',
    'MAIC': 'MAIC',
    'MAIC_ExactKL': 'MAIC + Exact KL',
    'MASAC': 'MASAC',
    'MASAC_ExactKL': 'MASAC + Exact KL',
}
SEEDS = (42, 43, 44, 45, 46)
COLORS = ('#78828E', '#7365A4', '#2A6FA5', '#1C998E', '#D18C2C', '#B75E6A')
SEED_COLORS = ('#687687',) * 5
LINE_STYLES = ('-', '--', '-.', ':', (0, (5, 1.5, 1, 1.5)), (0, (3, 2)))
FONT = 'DejaVu Sans'

plt.rcParams.update({
    'font.family': FONT, 'font.size': 9.3, 'axes.titlesize': 10.3,
    'axes.labelsize': 10, 'axes.edgecolor': '#4B5864',
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': '#2F3D4A', 'ytick.color': '#2F3D4A',
    'grid.color': '#DCE4EB', 'grid.linewidth': .55,
    'legend.fontsize': 8.6, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'savefig.facecolor': 'white',
})


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ('env','algo','seed','train_reward_last10pct','unique_states_final',
                'greedy_reward','greedy_success','stochastic_reward','stochastic_success')
    miss = set(expected)-set(df.columns)
    if miss: raise RuntimeError(f'CSV missing columns: {sorted(miss)}')
    if df.duplicated(['env','algo','seed']).any():
        raise RuntimeError('Duplicated environment-algorithm-seed keys')
    wanted={(e,a,s) for e in ENVS for a in ALGOS for s in SEEDS}
    given=set(map(tuple,df[['env','algo','seed']].itertuples(index=False,name=None)))
    if wanted != given:
        raise RuntimeError(f'Experiment matrix mismatch: {len(wanted-given)} missing, {len(given-wanted)} unexpected combinations')
    if df[list(expected[3:])].isna().any().any():
        raise RuntimeError('Incomplete final evaluation input: do not substitute zero for missing values')
    for col in ('greedy_success','stochastic_success'):
        if not df[col].between(0,1).all(): raise RuntimeError(f'Invalid success rate in {col}')
    return df


def ax_finish(ax, xlabel=None, ylabel=None, title=None):
    if title: ax.set_title(title, loc='left', pad=12, fontweight='semibold')
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.75)
    ax.tick_params(length=3)


def output(fig: plt.Figure, out: Path, basename: str, catalog: list, title_en: str, caption_en: str, caption_vi: str, **meta):
    out.mkdir(parents=True,exist_ok=True)
    p=out/basename
    fig.savefig(p.with_suffix('.png'), dpi=240, bbox_inches='tight', pad_inches=.13)
    fig.savefig(p.with_suffix('.pdf'), bbox_inches='tight', pad_inches=.13)
    plt.close(fig)
    catalog.append(dict(file=basename, title_en=title_en, caption_en=caption_en, caption_vi=caption_vi, **meta))
    print(f'  [OK] {basename}')


def finite_values(df, env, algo, metric):
    d=df[(df.env==env)&(df.algo==algo)].sort_values('seed')
    vals=d[metric].to_numpy(float)
    if len(vals)!=5 or not np.all(np.isfinite(vals)):
        raise RuntimeError(f'Missing/invalid seed data {env}/{algo}/{metric}')
    return vals


def metric_panel(df, env, metric, out, catalog):
    fig,axs=plt.subplots(1,2,figsize=(12.7,4.5),sharey=True)
    is_success = metric=='success'
    for ax,mode in zip(axs,('greedy','stochastic')):
        for i,algo in enumerate(ALGOS):
            y=finite_values(df,env,algo,mode+'_'+metric)*(100 if is_success else 1)
            ax.bar(i,np.mean(y),width=.65,color=COLORS[i],alpha=.42,
                   edgecolor=COLORS[i],linewidth=.85,zorder=2)
            ax.scatter(i+np.linspace(-.2,.2,5),y,s=26,marker='o',
                       facecolor=COLORS[i],edgecolor='#FBFCFF',linewidth=.7,zorder=3)
        ax.set_xticks(range(6),LABELS,rotation=25,ha='right')
        ax_finish(ax,title=mode.capitalize())
        if is_success: ax.set_ylim(-4,105)
        else: ax.set_ylim(bottom=min(0,ax.get_ylim()[0]))
    axs[0].set_ylabel('Task success rate (%)' if is_success else 'Evaluation return')
    fig.subplots_adjust(left=.08,right=.99,bottom=.21,top=.94,wspace=.13)
    basename=f'{env}_{metric}_greedy_stochastic'
    kind_en='success rate' if is_success else 'return'
    kind_vi='tỷ lệ thành công' if is_success else 'phần thưởng đánh giá'
    caption_en=(f'Final-checkpoint {kind_en} under greedy and stochastic action selection on {env}. '
                'Bars represent the mean across five training seeds (42–46); dots are individual seed values. '
                'Each policy is evaluated over 100 episodes per action-selection mode.')
    caption_vi=(f'{kind_vi.capitalize()} tại checkpoint cuối của các thuật toán trên {env} theo chế độ greedy và stochastic. '
                'Cột biểu diễn trung bình trên 5 seed (42–46); chấm biểu diễn kết quả từng seed. '
                'Mỗi chính sách được đánh giá qua 100 episode cho mỗi chế độ.')
    output(fig,out,basename,catalog,f'{env}: final-checkpoint {kind_en}',caption_en,caption_vi,category='final')


def training_final(df,env,out,catalog):
    fig,ax=plt.subplots(figsize=(9.5,4.5))
    for i,a in enumerate(ALGOS):
        v=finite_values(df,env,a,'train_reward_last10pct')
        ax.bar(i,v.mean(),width=.68,color=COLORS[i],alpha=.38,
               edgecolor=COLORS[i],zorder=2)
        ax.scatter(i+np.linspace(-.20,.20,5),v,s=30,color=COLORS[i],
                   edgecolors='white',linewidths=.6,zorder=3)
    ax.set_xticks(range(6),LABELS,rotation=23,ha='right')
    ax_finish(ax,ylabel='Mean episode return')
    ax.set_ylim(bottom=min(0,ax.get_ylim()[0]))
    fig.tight_layout()
    output(fig,out,f'{env}_train_reward_last10pct',catalog,
           f'{env}: late-stage training return',
           f'Mean episode return during the final 10% of training on {env}. '
           'Each dot represents one of five training seeds; bars show the across-seed mean.',
           f'Phần thưởng trung bình trong 10% episode huấn luyện cuối trên {env}. '
           'Mỗi chấm tương ứng một seed trong 5 seed; cột là trung bình giữa các seed.',category='final')


def state_final(df,env,out,catalog):
    fig,ax=plt.subplots(figsize=(9.5,4.5))
    for i,a in enumerate(ALGOS):
        v=finite_values(df,env,a,'unique_states_final')
        x=i+np.linspace(-.2,.2,5)
        ax.scatter(x,v,s=34,color=COLORS[i],edgecolors='white',linewidths=.6,zorder=3)
        ax.hlines(np.median(v),i-.28,i+.28,color=COLORS[i],linewidth=2.4,zorder=4)
    ax.set_xticks(range(6),LABELS,rotation=23,ha='right')
    ax.set_yscale('log')
    ax_finish(ax,ylabel='Recorded unique-state count (log scale)')
    fig.tight_layout()
    output(fig,out,f'{env}_unique_states_final',catalog,
           f'{env}: recorded unique-state counts',
           f'Recorded unique-state counts at the end of training on {env} (logarithmic axis). '
           'Dots denote individual seeds and horizontal lines denote seed medians. '
           'This metric follows the implementation-specific logging definition and is not a measure of successful coordination.',
           f'Số trạng thái duy nhất được ghi nhận cuối huấn luyện trên {env} (trục logarit). '
           'Các chấm biểu diễn từng seed; đoạn ngang biểu diễn trung vị theo seed. '
           'Chỉ số phụ thuộc định nghĩa ghi log của chương trình, không trực tiếp đo khả năng phối hợp thành công.',category='final')


def seed_scatter(df,env,out,catalog):
    fig,ax=plt.subplots(figsize=(10.1,4.7))
    for i,a in enumerate(ALGOS):
        v=finite_values(df,env,a,'stochastic_success')*100
        for j,seed in enumerate(SEEDS):
            ax.scatter(i+(j-2)*.103,v[j],s=31,marker='o',
                       color=COLORS[i],edgecolors='white',linewidths=.7,zorder=4)
    ax.set_xticks(range(6),LABELS,rotation=23,ha='right')
    ax.set_ylim(-5,105)
    ax_finish(ax,ylabel='Task success rate (%)')
    fig.tight_layout()
    output(fig,out,f'{env}_stochastic_success_by_seed',catalog,
           f'{env}: seed-wise stochastic success',
           f'Final-checkpoint stochastic task success on {env}, disaggregated by training seed (42–46). '
           'Each dot represents one policy evaluated over 100 episodes. Seeds appear left to right in ascending order within each method.',
           f'Tỷ lệ hoàn thành nhiệm vụ theo chính sách stochastic tại checkpoint cuối trên {env}, tách theo seed huấn luyện (42–46). '
           'Mỗi điểm biểu diễn một chính sách được đánh giá trên 100 episode. Các seed xếp từ trái sang phải theo thứ tự tăng dần trong từng thuật toán.',category='final')


def pushbox_paired(df,out,catalog):
    env='PushBox16'; chosen=('MAIC','MAIC_ExactKL','MADDPG')
    fig,ax=plt.subplots(figsize=(7.3,4.8))
    for i,a in enumerate(chosen):
        vals=finite_values(df,env,a,'stochastic_success')*100
        ax.bar(i,np.mean(vals),width=.59,color=COLORS[ALGOS.index(a)],alpha=.35,
               edgecolor=COLORS[ALGOS.index(a)],linewidth=1,zorder=2)
        for j,seed in enumerate(SEEDS):
            ax.scatter(i+(j-2)*.095, vals[j],s=38,marker='o',
                       color=COLORS[ALGOS.index(a)],edgecolors='white',linewidths=.7,zorder=4)
    ax.set_xticks(range(len(chosen)),('MAIC','MAIC + Exact KL','MADDPG'))
    ax.set_ylim(-6,105)
    ax_finish(ax,ylabel='Stochastic success rate (%)')
    fig.tight_layout()
    output(fig,out,'PushBox16_success_per_seed',catalog,
           'PushBox16: seed-wise final task success',
           'Final-checkpoint stochastic task success on PushBox16 for three representative methods. '
           'Bars denote the across-seed mean and dots denote individual training seeds (ordered 42–46 from left to right). '
           'MAIC achieved 0/5 successful seeds, whereas MAIC + Exact KL and MADDPG each achieved 3/5.',
           'Tỷ lệ hoàn thành nhiệm vụ theo chính sách stochastic tại checkpoint cuối trên PushBox16 của ba thuật toán. '
           'Cột là trung bình giữa các seed; mỗi điểm tương ứng một seed (42–46 từ trái sang phải). '
           'MAIC có 0/5 seed thành công; MAIC + Exact KL và MADDPG đều có 3/5 seed thành công.',category='synthesis')


def paired_mte(df,out,catalog):
    fig,axes=plt.subplots(2,2,figsize=(10.3,7.8),sharey=True)
    differences=[]
    for row,(baseline,variant) in enumerate((('MAIC','MAIC_ExactKL'),('MASAC','MASAC_ExactKL'))):
        for col,mode in enumerate(('greedy','stochastic')):
            ax=axes[row,col]
            b=finite_values(df,'MTE16',baseline,mode+'_success')*100
            v=finite_values(df,'MTE16',variant,mode+'_success')*100
            differences.append(float(np.mean(v-b)))
            for j,seed in enumerate(SEEDS):
                ax.plot([0,1],[b[j],v[j]],marker='o',
                        markersize=3.7,linewidth=1.05,color='#ABB6BF',alpha=.87,zorder=2)
            ax.plot([0,1],[b.mean(),v.mean()],marker='o',
                    markersize=5.2,linewidth=2.25,color=COLORS[ALGOS.index(variant)],zorder=5)
            ax.set_xticks([0,1],[baseline,baseline+' + Exact KL'])
            ax.set_xlim(-.17,1.17);ax.set_ylim(-5,105)
            ax_finish(ax,title=f'{baseline} — {mode.capitalize()}')
            if col==0: ax.set_ylabel('Task success rate (%)')
    fig.subplots_adjust(top=.96,bottom=.10,left=.08,right=.98,hspace=.43,wspace=.2)
    output(fig,out,'MTE16_paired_seed_response',catalog,
           'MTE16: paired seed-wise effects of exact KL',
           'Seed-wise final-checkpoint success on MTE16 with and without exact KL regularization. '
           'Light gray lines connect matching training seeds under the original and modified objective; black lines connect across-seed means. '
           'The panels correspond to MAIC and MASAC under greedy and stochastic evaluation. '
           f'Mean paired differences, in panel order, are {differences[0]:+.1f}, {differences[1]:+.1f}, '
           f'{differences[2]:+.1f}, and {differences[3]:+.1f} percentage points. '
           'These descriptive differences are not claims of statistical significance.',
           'So sánh theo cặp cùng seed tỷ lệ hoàn thành nhiệm vụ tại checkpoint cuối trên MTE16 trước và sau khi sử dụng Exact KL. '
           'Đường xám nhạt nối hai cấu hình cùng seed huấn luyện; đường đen thể hiện trung bình của năm seed. Bốn ô lần lượt thể hiện MAIC và MASAC ở hai chế độ greedy/stochastic. '
           f'Chênh lệch trung bình theo thứ tự các ô là {differences[0]:+.1f}, {differences[1]:+.1f}, '
           f'{differences[2]:+.1f} và {differences[3]:+.1f} điểm phần trăm. '
           'Đây là khác biệt mô tả, chưa chứng minh ý nghĩa thống kê.',category='synthesis')


def coverage_pair(df,out,catalog):
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.7),sharey=True)
    for ax,env in zip(axes,('PushBox16','PassSparse16')):
        for i,a in enumerate(ALGOS):
            v=finite_values(df,env,a,'unique_states_final')
            ax.scatter(i+np.linspace(-.17,.17,5),v,s=27,color=COLORS[i],
                       edgecolors='white',linewidths=.4,zorder=3)
            ax.hlines(np.median(v),i-.28,i+.28,color=COLORS[i],linewidth=2.3)
        ax.set_xticks(range(6),LABELS,rotation=27,ha='right')
        ax.set_yscale('log')
        ax_finish(ax,title=env)
    axes[0].set_ylabel('Recorded unique-state count (log scale)')
    fig.tight_layout()
    output(fig,out,'PushBox_PassSparse_recorded_unique_states',catalog,
           'Recorded state counts on cooperative tasks',
           'Recorded unique-state counts at the final training step in PushBox16 and PassSparse16. '
           'Dots indicate training seeds and horizontal segments are seed medians. '
           'The logarithmic scale and the implementation-specific logging definition should be considered when comparing methods.',
           'Số trạng thái duy nhất được ghi nhận vào cuối huấn luyện trên PushBox16 và PassSparse16. '
           'Chấm biểu diễn từng seed, đoạn ngang là trung vị giữa các seed. '
           'Việc so sánh phải xét thang logarit và định nghĩa đếm trạng thái riêng của chương trình.',category='synthesis')


def get_evaluation(path):
    with path.open(encoding='utf-8') as f: item=json.load(f)
    return float(item['success_rate'])*100


def checkpoint_figures(runs,out,catalog):
    created=0
    if not runs or not runs.is_dir():
        print('  [SKIP] intermediate checkpoint plots: original run directories unavailable')
        return created
    for env in ENVS:
        for algo in ('MAIC','MAIC_ExactKL','MASAC','MASAC_ExactKL'):
            points={s:[] for s in SEEDS}
            complete_eps=[]
            for ep in (2000,4000,6000,8000,10000):
                vals=[]
                for seed in SEEDS:
                    fn=runs/env/algo/f'seed{seed}'/('eval_model_final_stochastic.json' if ep==10000 else f'eval_model_ep{ep}_stochastic.json')
                    if fn.is_file(): vals.append(get_evaluation(fn))
                    else: vals=[];break
                if len(vals)==5:
                    complete_eps.append(ep)
                    for j,s in enumerate(SEEDS):points[s].append(vals[j])
            if len(complete_eps)<3:
                print(f'  [SKIP] {env} {algo}: fewer than three fully evaluated checkpoints across five seeds')
                continue
            fig,ax=plt.subplots(figsize=(7.4,4.3))
            for j,seed in enumerate(SEEDS):
                ax.plot(complete_eps,points[seed],marker='o',
                        markersize=3,linewidth=.95,color='#B6C3CE',alpha=.85,zorder=2)
            mean=np.mean(np.array([points[s] for s in SEEDS]),axis=0)
            ax.plot(complete_eps,mean,marker='o',markersize=4.7,linewidth=2.25,
                    color=COLORS[ALGOS.index(algo)],zorder=5)
            ax.set_xticks(complete_eps,[f'{e//1000}k' for e in complete_eps])
            ax.set_ylim(-5,105)
            ax_finish(ax,xlabel='Training episodes',ylabel='Stochastic success rate (%)')
            fig.tight_layout()
            output(fig,out,f'{env}_{algo}_checkpoint_stability',catalog,
                   f'{env}: checkpoint-wise stochastic success, {ALGO_CAPTIONS[algo]}',
                   f'Seed-wise stochastic task success for {ALGO_CAPTIONS[algo]} on {env} at evaluated checkpoints. '
                   'Light gray lines represent individual training seeds across checkpoints; the dark line denotes their arithmetic mean. '
                   'Each point is based on 100 evaluation episodes per seed. Only checkpoints evaluated for all five seeds are shown.',
                   f'Tỷ lệ thành công stochastic của {ALGO_CAPTIONS[algo]} trên {env} theo checkpoint huấn luyện. '
                   'Đường xám nhạt tương ứng từng seed xuyên suốt các checkpoint; đường đậm là trung bình theo seed. '
                   'Mỗi điểm sử dụng 100 episode đánh giá cho mỗi seed; chỉ hiển thị checkpoint có đủ dữ liệu của cả 5 seed.',category='checkpoint')
            created+=1
    return created


def moving_mean(x,window):
    x=np.asarray(x,float)
    if x.size<window:return x
    a=np.cumsum(np.insert(x,0,0))
    z=(a[window:]-a[:-window])/window
    return np.concatenate([np.full(window-1,np.nan),z])


def full_learning_curves(runs,out,catalog,window):
    created=0
    if not runs or not runs.is_dir():
        print('  [SKIP] training curves: original metrics JSON unavailable')
        return created
    for env in ENVS:
        fig,axes=plt.subplots(1,2,figsize=(12.9,4.45))
        complete=True
        for i,algo in enumerate(ALGOS):
            rs=[];us=[]
            for seed in SEEDS:
                path=runs/env/algo/f'seed{seed}'/'metrics_latest.json'
                if not path.is_file():complete=False;break
                d=json.loads(path.read_text(encoding='utf-8'))
                r=np.asarray(d['episode_rewards'],dtype=float)
                u=np.asarray([np.nan if v is None else v for v in d.get('unique_states_seen',[])],dtype=float)
                if len(r)<10000 or len(u)<10000:complete=False;break
                rs.append(moving_mean(r[:10000],window))
                us.append(u[:10000])
            if not complete:break
            x=np.arange(1,10001)
            for ax,M in zip(axes,(np.stack(rs),np.stack(us))):
                valid=np.flatnonzero(np.any(np.isfinite(M),axis=0))
                if not len(valid):
                    complete=False
                    break
                X=x[valid]
                mean=np.nanmean(M[:,valid],axis=0)
                lo=np.nanpercentile(M[:,valid],25,axis=0)
                hi=np.nanpercentile(M[:,valid],75,axis=0)
                if np.nanmin(M[:,valid])>=0:
                    lo=np.maximum(0,lo)
                ax.plot(X,mean,color=COLORS[i],linestyle=LINE_STYLES[i],linewidth=1.35,label=LABELS[i])
                ax.fill_between(X,lo,hi,color=COLORS[i],alpha=.055,linewidth=0)
        if not complete:
            plt.close(fig)
            print(f'  [SKIP] {env} training trajectory: full 5-seed metrics not available')
            continue
        ax_finish(axes[0],xlabel='Training episodes',ylabel='Episode return (moving mean)')
        ax_finish(axes[1],xlabel='Training episodes',ylabel='Recorded unique-state count')
        axes[1].set_yscale('symlog',linthresh=100)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.015),
                   ncol=6,frameon=False,fontsize=8.7)
        fig.subplots_adjust(top=.83,bottom=.17,left=.07,right=.99,wspace=.21)
        output(fig,out,f'curves_{env}',catalog,
               f'{env}: training return and recorded state counts',
               f'Training trajectories on {env} for the six algorithms over 10,000 episodes. '
               f'The left panel shows the {window}-episode moving mean of return; the right panel shows recorded unique-state counts on a symmetric-logarithmic axis. '
               'Lines denote means across five training seeds, and shaded regions span the 25th–75th seed percentiles. '
               'The state-count metric uses the implementation-specific logging convention.',
               f'Diễn biến huấn luyện trên {env} của 6 thuật toán qua 10.000 episode. '
               f'Hình trái biểu diễn phần thưởng trung bình trượt {window} episode; hình phải là số trạng thái duy nhất ghi nhận với trục đối xứng logarit. '
               'Đường là trung bình của 5 seed và vùng tô biểu diễn phân vị 25%–75% theo seed. '
               'Chỉ số trạng thái tuân theo quy tắc ghi log của chương trình.',category='training')
        created+=1
    return created


def latex_escape(text):
    for find,replace in (('\\','\\textbackslash{}'),('&','\\&'),('%','\\%'),('_','\\_'),('#','\\#'),('$','\\$')):
        text=text.replace(find,replace)
    return text


def caption_files(out,cat):
    lines=['# MAIC Study16: Academic figure captions (English / Tiếng Việt)',
           '', 'All captions correspond to figures generated from actual experiment records.',
           'The final-checkpoint comparisons use the 90-row seed CSV.',
           'Checkpoint and training curves are generated only if source JSON files exist.',
           'No finding is claimed to be statistically significant based solely on five seeds.', '']
    for k,c in enumerate(cat,1):
        lines.extend([f'## {k}. `{c["file"]}`',f'**English.** {c["caption_en"]}',
                      f'**Tiếng Việt.** {c["caption_vi"]}', ''])
    (out/'CAPTIONS_EN_VI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for lang in ('en','vi'):
        data=['% Figure blocks for Overleaf; insert only selected figures.',
              '% Requires \\usepackage{graphicx}. Copy the figures directory into Overleaf.', '']
        for c in cat:
            cap=latex_escape(c['caption_'+lang])
            data.extend(['\\begin{figure}[htbp]', '  \\centering',
                         f'  \\includegraphics[width=\\linewidth]{{figures_study16/{c["file"]}.pdf}}',
                         f'  \\caption{{{cap}}}',f'  \\label{{fig:{c["file"]}}}', '\\end{figure}',''])
        (out/f'FIGURES_OVERLEAF_{lang.upper()}.tex').write_text('\n'.join(data),encoding='utf-8')
    (out/'FIGURE_CATALOG.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2),encoding='utf-8')
    existing={c['file'] for c in cat}
    master=list(cat)
    # Full bilingual captions for figures that require original per-run JSON on VAST.
    for env in ENVS:
        fn=f'curves_{env}'
        if fn not in existing:
            master.append(dict(file=fn,category='training',status='needs_original_training_metrics',
                caption_en=f'Training trajectories for the six algorithms on {env} across 10,000 episodes. '
                           'The left panel shows the 100-episode moving-average return, while the right panel shows recorded unique-state counts. '
                           'Lines represent across-seed means and shaded regions denote interquartile ranges (five seeds, 42–46). '
                           'The state-count axis is symmetric-logarithmic.',
                caption_vi=f'Diễn biến huấn luyện của sáu thuật toán trên {env} trong 10.000 episode. '
                           'Hình trái biểu diễn phần thưởng trung bình trượt 100 episode; hình phải biểu diễn số trạng thái duy nhất được ghi nhận. '
                           'Các đường là trung bình giữa 5 seed (42–46), vùng tô là khoảng tứ phân vị. '
                           'Trục chỉ số trạng thái sử dụng tỷ lệ đối xứng logarit.'))
    for env in ENVS:
        for algo in ('MAIC','MAIC_ExactKL','MASAC','MASAC_ExactKL'):
            fn=f'{env}_{algo}_checkpoint_stability'
            if fn not in existing:
                master.append(dict(file=fn,category='checkpoint',status='needs_intermediate_evaluation_json',
                    caption_en=f'Stochastic task success of {ALGO_CAPTIONS[algo]} on {env} at selected training checkpoints. '
                               'Light gray lines trace individual seeds, and the dark line shows their mean. '
                               'Only checkpoints with complete five-seed evaluation records should be included.',
                    caption_vi=f'Tỷ lệ thành công stochastic của {ALGO_CAPTIONS[algo]} trên {env} tại các checkpoint huấn luyện. '
                               'Mỗi đường xám nhạt biểu diễn một seed trong 5 seed, đường đậm là trung bình theo seed. '
                               'Chỉ hiển thị checkpoint có đầy đủ bản ghi đánh giá của cả 5 seed.'))
    ml=['# Complete figure-caption catalogue (English / Vietnamese)','',
        'All 33 planned figures are documented. Figures with status PENDING require the original per-run JSON on VAST.','',
        '**Notation:** COMA* denotes the COMA-configured MAIC ablation, not a standalone on-policy COMA implementation.','']
    for k,c in enumerate(master,1):
        status='READY' if c['file'] in existing else 'PENDING — original raw data required'
        ml.extend([f'## {k}. `{c["file"]}` — {status}',f'**English:** {c["caption_en"]}',
                   f'**Tiếng Việt:** {c["caption_vi"]}',''])
    (out/'ALL_FIGURE_CAPTIONS_EN_VI.md').write_text('\n'.join(ml)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='reports_study16/figures/per_seed_values.csv')
    p.add_argument('--runs',default='runs_study16',help='Original 90-run directory, if available')
    p.add_argument('--out',default='reports_study16/figures_academic')
    p.add_argument('--window',type=int,default=100)
    a=p.parse_args()
    df=load_csv(Path(a.csv));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    catalog=[]
    for env in ENVS:
        for metric in ('success','reward'):
            metric_panel(df,env,metric,out,catalog)
        training_final(df,env,out,catalog)
        state_final(df,env,out,catalog)
        seed_scatter(df,env,out,catalog)
    pushbox_paired(df,out,catalog)
    paired_mte(df,out,catalog)
    coverage_pair(df,out,catalog)
    checkpoint_figures(Path(a.runs),out,catalog)
    full_learning_curves(Path(a.runs),out,catalog,a.window)
    caption_files(out,catalog)
    print(f'FINISHED: {len(catalog)} PDF + {len(catalog)} PNG, plus English/Vietnamese captions in {out}')
    print('Raw checkpoints and evaluation JSON files have not been changed.')

if __name__=='__main__':
    main()
