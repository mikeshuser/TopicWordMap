# -*- coding: utf-8 -*-
"""
Visualize words and topics in a standalone data dashboard.
Using the IMDB review data processed in previous steps, render a word map and 
corresponding topic breakdown, filterable by review sentiment.

Dependencies:
    pandas == 0.23
    bokeh == 2.2.3
"""

__author__ = "Mike Shuser"

import math

import pandas as pd
from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.layouts import row, column, gridplot, Spacer
from bokeh.models.widgets import Tabs, Panel, Div
from bokeh.models import (
    ColumnDataSource, 
    CDSView, 
    GroupFilter,
    HoverTool, 
    Panel, 
    Legend,
    NumeralTickFormatter, 
    Label,
    CustomJS, 
    RadioButtonGroup,
)

DATA_DIR = "../modelling/"
WORD_COORDS = f"{DATA_DIR}/umap_projection_n6_t100_reviewed.csv"
CLUSTER_DATA = f"{DATA_DIR}/cluster_map_n6_t100.csv"
LEGEND_LAYOUT = f"{DATA_DIR}/legend_layout.csv"
POS_STATS = f"{DATA_DIR}/positive_text_vocab_stats.csv"
NEG_STATS = f"{DATA_DIR}/negative_text_vocab_stats.csv"
PLOT_HEIGHT = 596
PLOT_WIDTH = 1400

def build_map_cds(
    map_data: pd.DataFrame, 
    clusters: pd.DataFrame, 
    scale_factor: int = 100
) -> ColumnDataSource:

    """
    convert DataFrame inputs into a ColumnDataSource
    """

    df = map_data.copy()
    df['colour'] = df['cluster'].map(clusters.colour)
    df['topic'] = df['cluster'].map(clusters.topic)
    df['subtopic'] = df['cluster'].map(clusters.subtopic)
    df['other_label'] = df.sentiment.replace({
        'positive' : 'neg.',
        'negative' : 'pos.',
    })

    cds = ColumnDataSource(data={
        'word': df['word'],
        'x': df['x'],
        'y': df['y'],
        'colours': df['colour'],
        'mentions': df['mentions'],
        'change': df['chg_vs_other'],
        'other': df['other_label'],
        'cluster_num': df['cluster'],
        'topic': df['topic'],
        'subtopic': df['subtopic'],
        'sentiment': df['sentiment'],
        'sizing': df['log2_mentions']/scale_factor
    })

    return cds

def prep_bar_df(clusters: pd.DataFrame, sentiment: str):

    """
    convert cluster DataFrame into a ColumnDataSource
    """

    #reformat the clusters dataframe
    df = clusters.copy()
    if sentiment == 'positive':
        filtered_df = df.loc[:, ~df.columns.str.contains("negative")].copy()
    else:
        filtered_df = df.loc[:, ~df.columns.str.contains("positive")].copy()
    filtered_df['sentiment'] = sentiment

    new_col_names = [
        'topic',
        'subtopic',
        'colour',
        'mentions',
        'percent',
        'sentiment',
    ]
    filtered_df.columns = new_col_names

    #get the total percents for a sentiment group
    totals = filtered_df.groupby(['sentiment', 'topic'])['percent'].sum()
    filtered_totals = totals.loc[sentiment].sort_values(ascending=True)

    #convert df to a data dict to pass to JS Callback
    topics = filtered_totals.index.tolist()

    data_dict = {}
    for topic in topics:
        subset = filtered_df.loc[filtered_df.topic==topic, :].copy()
        subset.sort_values(by='percent', ascending=False, inplace=True)
        subtopics = subset['subtopic'].tolist()
        colours = subset['colour'].tolist()

        data_dict[topic] = {
            'subtopics' : subset['subtopic'].tolist(),
            'colours' : subset['colour'].tolist(),
            'percents': subset['percent'].tolist(),
        }
    return filtered_df, filtered_totals, data_dict

def make_map(cds: ColumnDataSource, legend_layout: pd.Series):

    """
    main func to build the map tab. 
    """

    source = cds
    assert 'x' in source.data, "map cds missing 'x' coordinate"
    assert 'y' in source.data, "map cds missing 'y' coordinate" 

    #-------title and description-----------------------------------------------
    header = Div(text="""
        <header>
        <h1 style="font-size:14pt; 
            font-family:arial; 
            color:#444444; 
            margin:1px">
        IMDB Review Word Map
        </h1>
        </header>
    """)

    description = Div(text="""
        <section style="font-family:arial; color:#444444">
        <p style="font-size:12pt; margin:1px">
        This scatter plot is a 2d projection of the semantic relationships 
        underpinning the IMDB review dataset
        </p>
        <p style="font-size:11pt; font-style:italic; margin:1px">
        -Words with similar contexts are grouped closely together<br>
        -Each cluster covers a different topic<br>
        -Some clusters can be further divided into subtopics, visible when 
        hovering over a marker<br>
        -Markers are sized by frequency (log-scale)<br>
        </p>
        </section>
    """) 

    #-------main plot-----------------------------------------------------------
    x_rng = (source.data['x'].min() - .5, source.data['x'].max() + .5)
    y_rng = (source.data['y'].min() - .5, source.data['y'].max() + .5)
    fig = figure(
        plot_height=PLOT_HEIGHT,
        plot_width=PLOT_WIDTH,
        x_range=x_rng,
        y_range=y_rng,
        toolbar_location="below",
        tools="wheel_zoom, box_zoom, reset, save"
    )
    
    fig.toolbar.logo = None
    fig.axis.visible = False
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None
    
    tooltips = """
        <font face="Arial" size="1">
        <i>Word</i>: @word<br>
        <i>Topic:</i> @topic<br>
        <i>Subtopic:</i> @subtopic<br>
        <i>Mentions (vs @other):</i> @mentions (@change{:+%}) <br>
        </font>
        <hr style="height:1px;border-width:0;color:gray;background-color:gray">
    """

    fig.add_tools(HoverTool(tooltips=tooltips))

    note = "Note: Only includes words with more than " \
        + "100 mentions in at least one corpus"
    fig.add_layout(
        Label(
            text=note,
            x=0, y=0,
            x_units='screen', y_units='screen',
            text_font='arial', 
            text_font_size = '8pt',
            text_font_style = 'normal'
        ), 'below'
    )

    #-------legend--------------------------------------------------------------

    sent_filter = GroupFilter(column_name='sentiment', group='positive')
    legend_items = []
    for topic in legend_layout:
        topic_filter = GroupFilter(column_name='topic', group=topic)
        view = CDSView(source=source, filters=[topic_filter, sent_filter])
        c = fig.circle(
            x='x',
            y='y',
            source=source,
            view=view,
            line_color='white',
            fill_color='colours',
            radius='sizing',
            alpha=0.8,
            muted_color='#F7F4F6', 
            muted_alpha=0.6
        )
        legend_items.append((topic, [c]))
    
    legend = Legend(items=legend_items, location=(0,0))
    legend.click_policy = "mute"
    legend.title = "topics"
    legend.title_text_font = "arial"
    legend.title_text_font_size = "11pt"
    legend.title_text_font_style = "bold"
    legend.label_text_font = "arial"
    legend.label_text_font_size = "9pt"
    
    legend.spacing = 2
    fig.add_layout(legend, 'right')

    #-------sentiment filter----------------------------------------------------

    filter_header = Div(text="""
        <header>
        <h1 style="font-size:11pt; 
            font-family:arial; 
            color:#444444; 
            margin:5px 1px 1px 1px">
        Sentiment Filter
        </h1>
        </header>
    """)

    radio_buttons = RadioButtonGroup(
        labels=['Positive', 'Negative'], 
        active=0,
        width_policy="min"
    )
    callback = CustomJS(
        args=dict(
            source=source,
            filter=sent_filter,
        ),
        code="""

            if (this.active == 0) {
                filter.group = "positive";
            } else {
                filter.group = "negative";
            }
            source.change.emit();
        """
    )
    radio_buttons.js_on_click(callback)   

    layout = column(
        header,
        description,
        Spacer(height=10),
        row(filter_header, radio_buttons),
        fig,
    )
    return layout
    

def make_hbar(df: pd.DataFrame):

    """
    main func to build the map tab. 
    """

    #-------title and description-----------------------------------------------
    header = Div(text="""
        <header>
        <h1 style="font-size:14pt; 
            font-family:arial; 
            color:#444444; 
            margin:1px">
        IMDB Review Topic Summary
        </h1>
        </header>
    """)

    description = Div(text="""
        <section style="font-family:arial; color:#444444">
        <p style="font-size:12pt; margin:1px">
        This chart summarizes the topical content of the 
        language model word map
        </p>
        <p style="font-size:11pt; font-style:italic; margin:1px">
        -The proportions represent each topic's share of total mentions<br>
        -Subtopic shares are visible when hovering over a bar segment<br>
        </p>
        </section>
    """) 

    #-------main plot-----------------------------------------------------------

    pos_df, pos_totals, pos_data_dict = prep_bar_df(df, 'positive')
    neg_df, neg_totals, neg_data_dict = prep_bar_df(df, 'negative')

    figs = []
    figs.append(figure(
        plot_height=PLOT_HEIGHT,
        plot_width=700,
        y_range=list(pos_data_dict.keys()),
        x_range=(0, 0.255),
        toolbar_location=None,
        tools='',
        title='Total Share of Positive Mentions',
    ))
    figs.append(figure(
        plot_height=PLOT_HEIGHT,
        plot_width=540,
        y_range=list(pos_data_dict.keys()),
        x_range=(0, 0.255),
        toolbar_location=None,
        tools='',
        title='Total Share of Negative Mentions',
    ))

    for fig in figs:
        fig.toolbar.logo = None
        fig.axis.visible = True
        fig.title.text_font = 'arial'
        fig.title.text_font_size = '11pt'

        grid_colour = '#adafb3'
        fig.outline_line_color = None
        fig.xaxis.axis_line_color = grid_colour
        fig.xaxis.major_tick_line_color = grid_colour
        fig.xaxis.major_label_text_font = "arial"
        fig.xaxis.major_label_text_font_size = "9pt"
        fig.xaxis.major_label_overrides = {0:""}
        fig.xaxis.minor_tick_line_alpha = 0
        fig.xaxis.formatter = NumeralTickFormatter(format='0%')
        fig.xgrid.grid_line_color = grid_colour
        fig.xgrid.grid_line_alpha = 0.3
        
        fig.yaxis.axis_line_color = grid_colour
        fig.yaxis.major_label_text_font = "arial"
        fig.yaxis.major_label_text_font_size = "9pt"
        fig.yaxis.major_tick_line_color = grid_colour
        fig.yaxis.minor_tick_line_color = grid_colour
        fig.ygrid.grid_line_color = None

    r = 0
    for topic in pos_data_dict.keys():
    
        pos_slice = {'topic': [topic]}
        neg_slice = {'topic': [topic]}

        for i in range(len(pos_data_dict[topic]['subtopics'])):

            subtopic = pos_data_dict[topic]['subtopics'][i]
            percent = pos_data_dict[topic]['percents'][i]
            pos_slice[subtopic] = [percent]

            subtopic = neg_data_dict[topic]['subtopics'][i]
            percent = neg_data_dict[topic]['percents'][i]
            neg_slice[subtopic] = [percent]

        figs[0].hbar_stack(
            pos_data_dict[topic]['subtopics'],
            y='topic',
            source=pos_slice,
            width=1,
            line_color='white',
            fill_color=pos_data_dict[topic]['colours'],
        )
        figs[1].hbar_stack(
            neg_data_dict[topic]['subtopics'],
            y='topic',
            source=neg_slice,
            width=1,
            line_color='white',
            fill_color=neg_data_dict[topic]['colours'],
        )

        x_offset = 3
        y_offset = 4
        pos_labels = Label(
            y=r,
            x=pos_totals.loc[topic],
            x_units='data',
            y_units='data',
            text="{:.1%}".format(pos_totals.loc[topic]),
            x_offset=x_offset,
            y_offset=y_offset,
            text_font='arial',
            text_font_size="11pt",
            text_font_style="bold",
            level='glyph',
            render_mode='css',
        )
        neg_labels = Label(
            y=r,
            x=neg_totals.loc[topic],
            x_units='data',
            y_units='data',
            text="{:.1%}".format(neg_totals.loc[topic]),
            x_offset=x_offset,
            y_offset=y_offset,
            text_font='arial',
            text_font_size="11pt",
            text_font_style="bold",
            level='glyph',
            render_mode='css',
        )

        diff = 100 * (
            round(neg_totals.loc[topic], 3) - round(pos_totals.loc[topic], 3)
        )
        diff_offset = 50 if r >= 20 else 40
        if round(diff, 1) == 0:
            text_color = 'grey'
            text = "0.0" 
        elif diff > 0:
            text_color = 'green'
            text = "{:+.1f}".format(diff)
        else:
            text_color = 'red'
            text = "{:+.1f}".format(diff)

        diff_labels = Label(
            y=r,
            x=neg_totals.loc[topic],
            x_units='data',
            y_units='data',
            text=text,
            x_offset=x_offset + diff_offset,
            y_offset=y_offset,
            text_font='arial',
            text_font_size="11pt",
            text_font_style="italic",
            level='glyph',
            render_mode='css',
            text_color=text_color,
        )

        figs[0].add_layout(pos_labels)
        figs[1].add_layout(neg_labels)
        figs[1].add_layout(diff_labels)
        
        r+=1

    diff_header_1 = Label(
        y=23.5,
        x=0.23,
        x_units='data',
        y_units='data',
        text="Chg. vs Pos.",
        text_align="center",
        text_font='arial',
        text_font_size="11pt",
        text_font_style="italic",
        level='glyph',
        render_mode='css',
        text_color='#444444',
    )
    diff_header_2 = Label(
        y=23,
        x=0.23,
        x_units='data',
        y_units='data',
        text="(%)",
        text_align="center",
        text_font='arial',
        text_font_size="11pt",
        text_font_style="italic",
        level='glyph',
        render_mode='css',
        text_color='#444444',
    )
    figs[1].add_layout(diff_header_1)
    figs[1].add_layout(diff_header_2)


    pos_tooltips = """
        <font face="Arial" size="2">
        <strong>Subtopic:</strong> $name<br>
        <strong>Share:</strong> @$name{% 0.0}
        </font>
    """
    figs[0].add_tools(HoverTool(tooltips=pos_tooltips))

    figs[1].add_tools(HoverTool(tooltips=pos_tooltips))
    figs[1].yaxis.visible = False

    main_plt = gridplot(
        [figs],
        toolbar_options=dict(
            logo=None,
            autohide=True,
        ),
    )

    layout = column(
        header,
        description,
        Spacer(height=10),
        main_plt,
    )

    return layout 
    
if __name__ == '__main__':

    #load and prepare dataframes
    coords = pd.read_csv(WORD_COORDS, index_col=0)
    coords = coords[coords['remove']!=1]
    legend_layout = pd.read_csv(LEGEND_LAYOUT, index_col=0)

    pos = pd.read_csv(POS_STATS, index_col=0)
    pos['sentiment'] = 'positive'
    neg = pd.read_csv(NEG_STATS, index_col=0)
    neg['sentiment'] = 'negative'
    
    assert (pos.index.astype(str) == neg.index.astype(str)).all()
    pos['chg_vs_other'] = pos.mentions / neg.mentions - 1
    neg['chg_vs_other'] = neg.mentions / pos.mentions - 1

    map_data = coords.merge(
        pd.concat([pos, neg]), 
        how='left',
        left_on='word',
        right_index=True,
        sort=False,
    )
    map_data.index = pd.RangeIndex(len(map_data))

    clusters = pd.read_csv(CLUSTER_DATA, index_col=0)
    clusters['colour'] = clusters.topic.map(legend_layout.loc[:, 'colour'])

    total_mentions = map_data.groupby(["sentiment", "cluster"]).mentions.sum()
    clusters['positive_mentions'] = clusters.index.map(
        total_mentions.loc["positive"]
    )
    clusters['positive_%'] = (
        clusters.positive_mentions / clusters.positive_mentions.sum()
    )

    clusters['negative_mentions'] = clusters.index.map(
        total_mentions.loc["negative"]
    )
    clusters['negative_%'] = (
        clusters.negative_mentions / clusters.negative_mentions.sum()
    )

    map_cds = build_map_cds(map_data, clusters)
    map_fig = make_map(map_cds, legend_layout.index)
    bar_fig = make_hbar(clusters)

    map_panel = Panel(child=map_fig, title='Word Map')
    bar_panel = Panel(child=bar_fig, title='Topic Summary')
    tabs = Tabs(
        tabs=[map_panel, bar_panel],
        tabs_location="below",
    )

    output_file('../TopicWordMap.html', title='Topic Word Map')
    show(tabs)
