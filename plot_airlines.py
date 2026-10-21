#!/usr/bin/env python3
"""Три отдельных интерактивных HTML-графика Airlines: pandas + Plotly + Matplotlib + NetworkX.
python plot_airlines.py --input cleaned --output graphs
Исходные CSV не изменяются. HTML работает без интернета.
"""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import html
import re
from difflib import SequenceMatcher
import gc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle
import networkx as nx

BG = '#f5f6f8'
INK = '#172840'
BLUE = '#386bdb'
TEAL = '#148681'
ORANGE = '#d88932'
GROUPS = ['2–4 перелёта', '5–9 перелётов', '10–19 перелётов', '20+ перелётов']
BINS = ['1', '2', '3', '4', '5', '6', '7', '8–13', '14', '15–29', '30', '31–59', '60–89', '90+']


def csv(frame, path):
    frame.to_csv(path, sep=';', encoding='utf-8-sig', index=False)


def layout(fig, height=570):
    fig.update_layout(template='plotly_white', height=height, font=dict(family='Arial', size=13, color=INK),
                      margin=dict(l=75, r=45, t=55, b=75), paper_bgcolor='white',
                      colorway=[BLUE, TEAL, ORANGE], hoverlabel=dict(font_size=13))
    return fig


def setup_static():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': INK, 'axes.labelcolor': INK,
                         'axes.titleweight': 'bold', 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white'})


def prepare(folder, out, stats):
    f = pd.read_csv(folder / 'flights.csv', sep=';', dtype='string', keep_default_na=False)
    x = pd.read_csv(folder / 'user_flights.csv', sep=';', dtype='string', keep_default_na=False,
                    usecols=['user_id', 'flight_id'])
    for key in ['flight_id', 'flight_date', 'flight_number', 'departure_airport', 'arrival_airport']:
        if key not in f:
            raise ValueError(f'Нет столбца flights.{key}')
    if f.flight_id.eq('').any() or f.flight_id.duplicated().any():
        raise ValueError('flight_id должен быть заполненным и уникальным')
    if x.user_id.eq('').any() or not x.flight_id.isin(f.flight_id).all():
        raise ValueError('Пустой user_id или ссылка на отсутствующий рейс')
    stats['input_user_flights_rows'] = len(x)
    stats['input_users_with_trips'] = int(x.user_id.nunique())

    # ВРЕМЕННЫЕ псевдонимы только для полностью совпадающих записей рейса,
    # исключая flight_id. Различающиеся время, города, флаги и т.п. не склеиваем.
    fields = [c for c in f if c != 'flight_id']
    mask = f.duplicated(fields, keep=False)
    candidates = f.loc[mask].copy()
    aliases = pd.DataFrame(columns=['old_flight_id', 'canonical_flight_id'])
    if len(candidates):
        canonical = candidates.groupby(fields, dropna=False, sort=False).flight_id.transform('min')
        aliases = pd.DataFrame({'old_flight_id': candidates.flight_id,
                                'canonical_flight_id': canonical})
        aliases = aliases[aliases.old_flight_id.ne(aliases.canonical_flight_id)]
        mapping = aliases.set_index('old_flight_id').canonical_flight_id
        x['flight_id'] = x.flight_id.map(mapping).fillna(x.flight_id)
        f = f.loc[~f.flight_id.isin(aliases.old_flight_id)].copy()
    csv(aliases, out / 'flight_aliases_for_analysis.csv')
    stats['exact_flight_aliases'] = len(aliases)
    stats['duplicate_user_flight_links_collapsed'] = int(x.duplicated().sum())
    x = x.drop_duplicates()
    f = f[['flight_id', 'flight_date', 'flight_number', 'departure_airport', 'arrival_airport']]
    d = x.merge(f, on='flight_id', how='left', validate='many_to_one')
    d['date'] = pd.to_datetime(d.flight_date, format='%Y-%m-%d', errors='coerce')
    stats['invalid_date_rows_excluded_from_time_charts'] = int(d.date.isna().sum())
    stats['analysis_trip_rows'] = len(d)
    known = d.date.dropna()
    stats['date_min'] = str(known.min().date()) if len(known) else None
    stats['date_max'] = str(known.max().date()) if len(known) else None
    del x, f, candidates
    gc.collect()
    return d


def rhythm(d, out, stats):
    valid = d[d.date.notna()][['user_id', 'flight_id', 'date']].copy()
    trips = valid.groupby('user_id').size()
    dates = valid[['user_id', 'date']].drop_duplicates().sort_values(['user_id', 'date'])
    dates['gap_days'] = dates.groupby('user_id').date.diff().dt.days
    gaps = dates.dropna(subset=['gap_days']).copy()
    gaps['gap_days'] = gaps.gap_days.astype(int)
    gaps['trip_count'] = gaps.user_id.map(trips)
    gaps['group'] = pd.cut(gaps.trip_count, [1, 4, 9, 19, np.inf], labels=GROUPS)
    limits = [0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 29, 30, 59, 89, np.inf]
    gaps['bin'] = pd.cut(gaps.gap_days, limits, labels=BINS)
    counts = pd.crosstab(gaps['group'], gaps['bin']).reindex(index=GROUPS, columns=BINS, fill_value=0)
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    flat = counts.rename_axis('group').reset_index().melt(id_vars='group', var_name='interval_days', value_name='intervals')
    flat['percent_within_group'] = shares.to_numpy().reshape(-1, order='F')
    csv(flat, out / 'rhythm_by_group.csv')
    dist = gaps.gap_days.value_counts().sort_index().rename_axis('gap_days').reset_index(name='intervals')
    csv(dist, out / 'rhythm_exact_days.csv')
    stats['rhythm'] = {
        'users_with_two_or_more_flight_dates': int(gaps.user_id.nunique()),
        'positive_intervals': len(gaps),
        'median_gap_days': float(gaps.gap_days.median()) if len(gaps) else None,
        'additional_flights_on_same_user_date': len(valid) - len(dates),
        'largest_interval_counts': {str(int(k)): int(v) for k, v in gaps.gap_days.value_counts().head(10).items()},
        'intervals_per_group': {str(k): int(v) for k, v in counts.sum(axis=1).items()}}
    labels = [f'{g}<br>n={int(n):,}' for g, n in counts.sum(axis=1).items()]
    fig = make_subplots(rows=2, cols=1, row_heights=[.58, .42], vertical_spacing=.22,
                        subplot_titles=('Интервалы внутри каждой группы, %', 'Точное распределение интервалов, все пассажиры'))
    fig.add_trace(go.Heatmap(z=shares.to_numpy(), x=BINS, y=labels,
                   customdata=counts.to_numpy(), colorscale='Blues', colorbar=dict(title='%', len=.5, y=.78),
                   text=np.round(shares.to_numpy(), 1), texttemplate='%{text:.1f}',
                   hovertemplate='%{y}<br>Интервал: %{x} дн.<br>%{z:.2f}% · %{customdata:,} интервалов<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dist.gap_days, y=dist.intervals, mode='lines', line=dict(color=BLUE, width=2),
                   hovertemplate='%{x} дней: %{y:,} интервалов<extra></extra>'), row=2, col=1)
    fig.update_xaxes(title_text='Интервал между разными датами полётов, дней', row=2, col=1)
    fig.update_yaxes(title_text='Число интервалов', row=2, col=1)
    fig.update_yaxes(autorange='reversed', row=1, col=1)
    layout(fig, 750)
    fig.update_layout(showlegend=False)
    # Статическая копия.
    st, axes = plt.subplots(2, 1, figsize=(13.2, 8), gridspec_kw={'height_ratios': [1.1, 1]}, constrained_layout=True)
    im = axes[0].imshow(shares, cmap='Blues', aspect='auto', vmin=0)
    axes[0].set_xticks(range(len(BINS)), BINS)
    axes[0].set_yticks(range(4), [f'{g}\nn={int(n):,}' for g,n in counts.sum(axis=1).items()])
    for i in range(4):
        for j in range(len(BINS)):
            v = shares.iloc[i,j]
            axes[0].text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8,
                         color='white' if v > shares.to_numpy().max()*.55 else INK)
    st.colorbar(im, ax=axes[0], label='% интервалов внутри группы', fraction=.025)
    axes[0].set_title('Ритм поездок · интервалы между разными датами полётов', loc='left', pad=16)
    axes[0].set_xlabel('Интервал, дней; диапазоны имеют разную ширину')
    axes[1].plot(dist.gap_days, dist.intervals, color=BLUE, lw=1.8)
    axes[1].set(xlabel='Интервал, дней', ylabel='Число интервалов')
    axes[1].grid(axis='y', alpha=.15)
    axes[1].set_title('Точное распределение · без объединения дней в диапазоны', loc='left', fontsize=11)
    st.savefig(out / '01_rhythm.png', dpi=160)
    plt.close(st)
    return fig


def companions(d, out, stats, folder, min_days=2, max_edges=30):
    valid = d[d.date.notna()][['flight_id', 'user_id', 'date']].drop_duplicates(['flight_id', 'user_id'])
    # Числовые коды экономят память при подсчёте пар.
    users = np.array(sorted(valid.user_id.unique()))
    lookup = pd.Series(np.arange(len(users), dtype=np.int64), index=users)
    valid['uid'] = valid.user_id.map(lookup).astype('int64')
    dates = np.array(sorted(valid.date.unique()))
    date_index = {pd.Timestamp(day): i for i, day in enumerate(dates)}
    n = len(users)
    size = valid.groupby('flight_id', sort=False).size()
    stats['companions_pair_comparisons'] = int((size * (size-1)//2).sum())
    multi = valid[valid.flight_id.isin(size[size.gt(1)].index)].sort_values(['flight_id', 'uid'])
    uid = multi.uid.to_numpy()
    fid = multi.flight_id.to_numpy()
    time = multi.date.to_numpy()
    cuts = np.r_[0, np.flatnonzero(fid[1:] != fid[:-1])+1, len(fid)] if len(fid) else np.array([0])
    counts = Counter()
    daymasks = {}
    print(f'Совместные рейсы: {len(cuts)-1:,} групп, {stats["companions_pair_comparisons"]:,} сравнений пар', flush=True)
    for start, end in zip(cuts[:-1], cuts[1:]):
        bit = 1 << date_index[pd.Timestamp(time[start])]
        people = uid[start:end].tolist()
        for a,b in combinations(people, 2):
            key = a*n + b
            counts[key] += 1
            daymasks[key] = daymasks.get(key, 0) | bit
    all_pairs = len(counts)
    rows = []
    for key, num in counts.items():
        days = daymasks[key].bit_count()
        if days >= min_days:
            a,b = divmod(key,n)
            mask = daymasks[key]
            indices = [i for i in range(len(dates)) if mask >> i & 1]
            rows.append((str(users[a]), str(users[b]), num, days,
                         str(pd.Timestamp(dates[indices[0]]).date()), str(pd.Timestamp(dates[indices[-1]]).date())))
    pairs = pd.DataFrame(rows, columns=['user_a','user_b','shared_flights','shared_dates','first_shared_date','last_shared_date'])
    pairs = pairs.sort_values(['shared_dates','shared_flights','user_a','user_b'], ascending=[False,False,True,True]).reset_index(drop=True)
    # Совпадения поездок могут быть вызваны двумя ID одного человека.
    u = pd.read_csv(folder / 'users.csv', sep=';', dtype='string', keep_default_na=False).set_index('user_id')
    if not u.index.is_unique:
        raise ValueError('user_id в users должен быть уникальным')
    display_names = (u.first_name + ' ' + u.last_name).str.strip()
    display_names = display_names.where(u.first_name.ne('') & u.last_name.ne(''), u.full_name)
    def signature(value):
        return re.sub(r'[^\w\s]', '', str(value).upper()).strip()
    signatures = display_names.map(signature)
    pairs['name_a'] = pairs.user_a.map(display_names).fillna('')
    pairs['name_b'] = pairs.user_b.map(display_names).fillna('')
    a_names = pairs.user_a.map(signatures).fillna('')
    b_names = pairs.user_b.map(signatures).fillna('')
    pairs['name_similarity'] = [round(max(SequenceMatcher(None,a,b).ratio(), SequenceMatcher(None,' '.join(sorted(a.split())),' '.join(sorted(b.split()))).ratio()), 3) if a and b else 0.0
                                for a,b in zip(a_names,b_names)]
    pairs['similar_names_review'] = pairs.name_similarity.ge(.82)
    trips_per_user = d.groupby('user_id').flight_id.nunique()
    pairs['shared_fraction_of_smaller_history'] = pairs.shared_flights / np.minimum(
        pairs.user_a.map(trips_per_user), pairs.user_b.map(trips_per_user))
    pairs['identity_note'] = np.where(pairs.similar_names_review,
        'Похожие имена: проверить, не два ли это ID одного человека',
        'Сходство имён не отмечено; личности не проверены')
    csv(pairs, out / 'repeated_companions.csv')
    histogram = Counter(counts.values())
    csv(pd.DataFrame(sorted(histogram.items()), columns=['shared_flights','pairs']), out / 'companions_pair_distribution.csv')
    stats['companions'] = {'pairs_on_any_flight': all_pairs, 'pairs_on_repeated_dates': len(pairs),
                           'min_shared_dates': min_days, 'network_edge_limit': max_edges,
                           'pair_distribution_by_shared_flights': dict(sorted(histogram.items())),
                           'similar_name_pairs_for_review': int(pairs.similar_names_review.sum()),
                           'top_pairs': pairs.head(8).to_dict('records')}
    del counts, daymasks, valid, multi
    gc.collect()
    display = pairs.head(max_edges).copy()
    graph = nx.Graph()
    node_names = {}
    for row in display.itertuples(index=False):
        graph.add_edge(row.user_a, row.user_b, weight=row.shared_dates,
                       shared_flights=row.shared_flights, review=bool(row.similar_names_review))
        node_names[row.user_a] = row.name_a
        node_names[row.user_b] = row.name_b
    # Размещаем компоненты отдельно: пружинная раскладка сжимала отдельные пары.
    positions = {}
    components = sorted(nx.connected_components(graph),
                        key=lambda nodes: (-max(v['weight'] for _,_,v in graph.subgraph(nodes).edges(data=True)), min(nodes)))
    columns = min(5, max(1, int(np.ceil(np.sqrt(len(components))))))
    for index_component, nodeset in enumerate(components):
        nodeset = sorted(nodeset)
        cx = (index_component % columns) * 3.0
        cy = -(index_component // columns) * 2.0
        if len(nodeset) == 2:
            local = {nodeset[0]: (-.75, 0), nodeset[1]: (.75, 0)}
        else:
            local = nx.circular_layout(graph.subgraph(nodeset))
        for node, pos in local.items():
            positions[node] = np.array([cx + pos[0], cy + pos[1]])
    # Короткие подписи; полный user_id — при наведении и в CSV.
    ids = {node: f'P{i+1:03d}' for i,node in enumerate(sorted(graph))}
    csv(pd.DataFrame([{'label':label,'user_id':node} for node,label in ids.items()]), out / 'network_labels.csv')
    fig = go.Figure()
    for a,b,data in graph.edges(data=True):
        x1,y1 = positions[a]; x2,y2 = positions[b]
        text = (f'{a} ↔ {b}<br>{html.escape(node_names[a])} / {html.escape(node_names[b])}'
                f'<br>Общих рейсов: {data["shared_flights"]}<br>Разных дат: {data["weight"]}'
                + ('<br>Похожие имена: проверить ID' if data['review'] else '<br>Личности не проверены'))
        edge_color = ORANGE if data['review'] else BLUE
        fig.add_trace(go.Scatter(x=[x1,x2],y=[y1,y2],mode='lines',
                      line=dict(width=1+np.log2(data['weight']),color=edge_color),
                      hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=[(x1+x2)/2],y=[(y1+y2)/2],mode='markers+text',
                      marker=dict(size=8,color='rgba(255,255,255,.1)'),text=[str(data['weight'])],
                      textposition='top center',customdata=[text],
                      hovertemplate='%{customdata}<extra></extra>',showlegend=False))
    nodes = list(graph)
    if nodes:
        fig.add_trace(go.Scatter(x=[positions[a][0] for a in nodes],y=[positions[a][1] for a in nodes],
                      mode='markers+text',text=[ids[a] for a in nodes],textposition='top center',
                      customdata=[[a,graph.degree(a),node_names[a]] for a in nodes],
                      marker=dict(size=[9+3*np.sqrt(graph.degree(a)) for a in nodes], color=TEAL,
                                  line=dict(width=1,color='white')),
                      hovertemplate='%{customdata[0]}<br>%{customdata[2]}<br>Связей на показанном графе: %{customdata[1]}<extra></extra>',showlegend=False))
    else:
        fig.add_annotation(text=f'Пар с совместными перелётами в {min_days} разные даты не найдено',
                           x=.5,y=.5,xref='paper',yref='paper',showarrow=False)
    layout(fig, 680)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor='x', scaleratio=1)
    fig.update_layout(title=f'Повторные совместные перелёты · {len(display)} показанных пар из {len(pairs):,}')
    st,ax=plt.subplots(figsize=(13.2,8.2),constrained_layout=True)
    if nodes:
        nx.draw_networkx_edges(graph,positions,ax=ax,edge_color=[ORANGE if v['review'] else BLUE for _,_,v in graph.edges(data=True)],alpha=.8,
                               width=[.6+np.log2(v['weight']) for _,_,v in graph.edges(data=True)])
        nx.draw_networkx_nodes(graph,positions,ax=ax,node_color=TEAL,edgecolors='white',linewidths=.7,
                               node_size=[45+22*np.sqrt(graph.degree(a)) for a in graph])
        label_positions = {a: pos + np.array([0, -.22]) for a,pos in positions.items()}
        nx.draw_networkx_labels(graph,label_positions,labels=ids,ax=ax,font_size=7,font_color=INK)
        nx.draw_networkx_edge_labels(graph,positions,ax=ax,
            edge_labels={(a,b): str(v['weight']) for a,b,v in graph.edges(data=True)},
            font_size=9,rotate=False,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
        ax.margins(.12)
    else:
        ax.text(.5,.5,'Повторные совместные поездки не найдены',ha='center',va='center')
    ax.set_title(f'Кто снова летает вместе · {len(display)} пар из {len(pairs):,}',loc='left',pad=18)
    ax.text(0,1.01,f'Число на линии — общие даты. Оранжевый: похожие имена, нужно проверить ID. Синий: сходство не отмечено.',
            transform=ax.transAxes,fontsize=10)
    ax.axis('off')
    st.savefig(out / '02_companions.png',dpi=160)
    plt.close(st)
    stats['companions']['network_nodes_shown']=len(graph)
    stats['companions']['network_edges_shown']=len(display)
    return fig


def sankey(d,out,stats,top_n=10):
    valid=d[d.departure_airport.ne('') & d.arrival_airport.ne('')].copy()
    ranking=pd.concat([valid.departure_airport,valid.arrival_airport]).value_counts()
    top=ranking.head(top_n).index.tolist()
    valid['source']=valid.departure_airport.where(valid.departure_airport.isin(top),'Другие')
    valid['target']=valid.arrival_airport.where(valid.arrival_airport.isin(top),'Другие')
    flows=valid.groupby(['source','target'],sort=True).size().reset_index(name='passenger_flight_records')
    csv(flows,out / 'sankey_flows.csv')
    route=valid.groupby(['departure_airport','arrival_airport']).size().reset_index(name='passenger_flight_records')
    csv(route.sort_values('passenger_flight_records',ascending=False),out / 'all_route_counts.csv')
    # Две отдельные колонки узлов, даже когда код аэропорта одинаков.
    categories=top+(['Другие'] if (valid.source.eq('Другие') | valid.target.eq('Другие')).any() else [])
    left=valid.source.value_counts().reindex(categories,fill_value=0)
    right=valid.target.value_counts().reindex(categories,fill_value=0)
    total=len(valid)
    colors=['#386bdb','#148681','#d88932','#8b68b6','#cf6679','#5f89a8','#6f9851','#a37a55','#527478','#a9a24d','#a4acb7']
    color={name:colors[i%len(colors)] for i,name in enumerate(categories)}
    def rgba(hexcolor,alpha=.27):
        return f'rgba({int(hexcolor[1:3],16)},{int(hexcolor[3:5],16)},{int(hexcolor[5:7],16)},{alpha})'
    labels=[f'{c} · {int(v):,}' for c,v in left.items()]+[f'{c} · {int(v):,}' for c,v in right.items()]
    index={c:i for i,c in enumerate(categories)}
    n=len(categories)
    def centers(counts):
        heights=counts.to_numpy()/max(total,1)*.82
        pad=.18/max(n-1,1)
        starts=np.r_[0,np.cumsum(heights[:-1])+np.arange(1,n)*pad]
        return starts,heights,starts+heights/2
    ls,lh,ly=centers(left);rs,rh,ry=centers(right)
    fig=go.Figure(go.Sankey(arrangement='fixed',
        node=dict(label=labels,x=[.025]*n+[.975]*n,y=list(ly)+list(ry),pad=12,thickness=18,
                  color=[color[c] for c in categories]*2,line=dict(color='white',width=.5),
                  hovertemplate='%{label}<extra></extra>'),
        link=dict(source=[index[c] for c in flows.source],target=[n+index[c] for c in flows.target],
                  value=flows.passenger_flight_records.tolist(),color=[rgba(color[c]) for c in flows.source],
                  customdata=[[a,b] for a,b in zip(flows.source,flows.target)],
                  hovertemplate='%{customdata[0]} → %{customdata[1]}<br>%{value:,} записей пассажирских перелётов<extra></extra>')))
    layout(fig,780)
    fig.update_layout(margin=dict(l=35,r=35,t=65,b=30),
                      title=f'Направления · топ-{top_n} аэропортов и «Другие» · {total:,} записей')
    stats['sankey']={'included_trip_rows':total,'excluded_missing_airports':len(d)-total,
                     'top_airports':top,'flow_sum':int(flows.passenger_flight_records.sum()),
                     'top_routes':route.sort_values('passenger_flight_records',ascending=False).head(10).to_dict('records')}
    assert int(flows.passenger_flight_records.sum())==total
    # Статическая Санки: ширина лент пропорциональна потоку с единым масштабом.
    st,ax=plt.subplots(figsize=(13.2,9.2),constrained_layout=True)
    lo=ls.copy();ro=rs.copy()
    scale=.82/max(total,1)
    for row in flows.itertuples(index=False):
        i=index[row.source];j=index[row.target];h=row.passenger_flight_records*scale
        y0=lo[i];y1=ro[j];lo[i]+=h;ro[j]+=h
        vertices=[(.15,y0),(.4,y0),(.6,y1),(.85,y1),(.85,y1+h),(.6,y1+h),(.4,y0+h),(.15,y0+h),(.15,y0)]
        codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(vertices,codes),facecolor=color[row.source],edgecolor='none',alpha=.28))
    for i,c in enumerate(categories):
        ax.add_patch(Rectangle((.13,ls[i]),.02,lh[i],facecolor=color[c],edgecolor='white',linewidth=.5))
        ax.add_patch(Rectangle((.85,rs[i]),.02,rh[i],facecolor=color[c],edgecolor='white',linewidth=.5))
        ax.text(.115,ly[i],f'{c}\n{int(left.iloc[i]):,}',ha='right',va='center',fontsize=9)
        ax.text(.885,ry[i],f'{c}\n{int(right.iloc[i]):,}',ha='left',va='center',fontsize=9)
    ax.set_xlim(-.03,1.03);ax.set_ylim(1.03,-.04);ax.axis('off')
    ax.set_title('Потоки между аэропортами · записи пассажирских перелётов',loc='left',pad=30)
    ax.text(.13,-.025,'ВЫЛЕТ',fontsize=10,fontweight='bold')
    ax.text(.85,-.025,'ПРИЛЁТ',fontsize=10,fontweight='bold')
    st.savefig(out / '03_sankey.png',dpi=160)
    plt.close(st)
    return fig


def make_html(figures, out, stats):
    """Один самостоятельный HTML на график: только заголовок, легенда и график."""
    pages = [
        ('01_rhythm.html', 'Ритм поездок',
         '<span>Цвет — доля интервалов внутри группы, %</span>'
         '<span>n — число интервалов</span>', figures[0]),
        ('02_companions.html', 'Кто снова летает вместе',
         '<span><i class="dot"></i>Профиль пассажира</span>'
         '<span><i class="line orange"></i>Похожие ФИО</span>'
         '<span><i class="line blue"></i>Сходство ФИО не отмечено</span>'
         '<span>Число на линии — общие даты</span>', figures[1]),
        ('03_sankey.html', 'Потоки между аэропортами',
         '<span>Слева — вылет · справа — прилёт</span>'
         '<span>Ширина ленты — число записей перелётов</span>'
         '<span>«Другие» — остальные аэропорты</span>', figures[2]),
    ]
    script = get_plotlyjs()
    for filename, title, legend, original in pages:
        fig = go.Figure(original)
        fig.update_layout(title=None, margin=dict(t=55))
        chart = fig.to_html(full_html=False, include_plotlyjs=False,
            div_id=Path(filename).stem + '-plot',
            config={'responsive': True, 'displaylogo': False,
                    'toImageButtonOptions': {'format': 'png', 'scale': 2}})
        page = ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>' + html.escape(title) + '</title><style>'
                '*{box-sizing:border-box}body{margin:0;background:#fff;color:#172840;'
                'font:14px/1.45 Arial,sans-serif}main{max-width:1440px;margin:0 auto;padding:24px 20px}'
                'h1{font-size:26px;line-height:1.2;margin:0 0 15px;font-weight:600}'
                '.legend{display:flex;flex-wrap:wrap;gap:10px 24px;color:#526175;margin-bottom:14px}'
                '.legend span{display:inline-flex;align-items:center;gap:8px}'
                '.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#148681}'
                '.line{display:inline-block;width:24px;height:3px}.orange{background:#d88932}.blue{background:#386bdb}'
                '@media(max-width:700px){main{padding:18px 8px}h1{font-size:22px}.legend{gap:8px 16px}}'
                '</style><script>' + script + '</script></head><body><main><h1>'
                + html.escape(title) + '</h1><div class="legend" aria-label="Легенда графика">'
                + legend + '</div>' + chart + '</main></body></html>')
        (out / filename).write_text(page, encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--min-shared-days',type=int,default=2)
    p.add_argument('--max-network-edges',type=int,default=30)
    p.add_argument('--top-airports',type=int,default=10)
    a=p.parse_args()
    if a.min_shared_days<2 or a.max_network_edges<1 or not 2<=a.top_airports<=15:
        p.error('min-shared-days >=2, max-network-edges >=1, top-airports от 2 до 15')
    if a.output.exists():
        p.error('Выходная папка уже существует: укажите новую, например graphs_v2')
    a.output.mkdir(parents=True)
    setup_static()
    stats={}
    print('Читаю таблицы и проверяю связи…',flush=True)
    d=prepare(a.input,a.output,stats)
    print('Строю ритм поездок…',flush=True)
    fig1=rhythm(d,a.output,stats)
    print('Считаю повторные совместные перелёты…',flush=True)
    fig2=companions(d,a.output,stats,a.input,a.min_shared_days,a.max_network_edges)
    print('Строю потоки направлений…',flush=True)
    fig3=sankey(d,a.output,stats,a.top_airports)
    make_html([fig1,fig2,fig3],a.output,stats)
    (a.output/'analysis_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2,default=int),encoding='utf-8')
    print(f'Готово. Отдельные графики в {a.output.resolve()}:\n01_rhythm.html\n02_companions.html\n03_sankey.html', flush=True)


if __name__=='__main__':
    main()
