"""
File: chr_circos.py
Description: Instance a chromosome circos plot object.
CreateDate: 2025/7/9
Author: xuwenlin
E-mail: wenlinxu.njfu@outlook.com
"""
from typing import Union, Tuple, List, Dict
from functools import partial
from math import log2, cos, sin
from numpy import pi, linspace, array
import pandas
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class ChromosomeCircos:
    """
    Draw genome circos figure.
    :param chr_len_file: Chromosome length file. (ChrName\\tChrLen\\tEtc)
    :param spacing: Set the spacing between chromosomes to be one quarter (spacing=4) of the shortest chromosome.
    :param font: Global font of figure.
    :param figsize: Figure dimension (width, height) in inches.
    :param dpi: Dots per inch.
    """
    def __init__(
        self,
        chr_len_file: str,
        spacing: int = 4,
        font: str = None,
        font_size: Union[int, float] = 8.0,
        figsize: Tuple[float, float] = (10, 8),
        dpi: int = 100
    ):
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.size'] = font_size
        if font is not None:
            plt.rcParams['font.family'] = font
        self.spacing = spacing
        chr_len_dict, chr_theta_dict, chr_width_dict = self.__get_chr_theta_width(chr_len_file)
        self.chr_len_dict = chr_len_dict
        self.chr_theta_dict = chr_theta_dict
        self.chr_width_dict = chr_width_dict
        self.figure = None
        self.figsize = figsize
        self.dpi = dpi
        self.font = font

    def __get_chr_theta_width(self, chr_len_file: str) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
        """
        Get theta angle and width of each chromosome bar.
        :return: tuple(raw_chr_len_dict: dict, chr_theta_dict: dict, chr_width_dict: dict)
                 raw_chr_len_dict -> chromosome length dict. {chr_name: length, ...}
                 chr_theta_dict -> chromosome theta dict. {chr_name: theta, ...}
                 chr_width_dict -> chromosome width dict. {chr_name: width, ...}
        """
        # Read in chromosome length file.
        with open(chr_len_file) as f:
            raw_chr_len_dict = {
                line.split('\t')[0]: int(line.strip().split('\t')[1])
                for line in f
            }

        # Calculate length percentage of each chromosome.
        chr_width_dict = {}
        each_space_len = min(raw_chr_len_dict.values()) / self.spacing
        total_space_len = each_space_len * len(raw_chr_len_dict.keys())
        for chr_name, chr_len in raw_chr_len_dict.items():
            chr_width_dict[chr_name] = 2 * pi * chr_len / (sum(raw_chr_len_dict.values()) + total_space_len)

        # Calculate theta angle of each chromosome.
        chr_theta_dict = {}
        chr_theta = 0
        space_angle = 2 * pi * each_space_len / (sum(raw_chr_len_dict.values()) + total_space_len)
        for chr_name, chr_len in chr_width_dict.items():
            if chr_theta == 0:
                chr_theta_dict[chr_name] = chr_len / 2
            else:
                chr_theta_dict[chr_name] = chr_theta + chr_len / 2
            chr_theta += chr_len + space_angle

        return raw_chr_len_dict, chr_theta_dict, chr_width_dict

    def __loci_to_polar(
        self,
        chr_name: str,
        start: int,
        end: int
    ) -> float:
        """Convert genome loci to polar coordinates radian."""
        angle = self.chr_theta_dict[chr_name] - self.chr_width_dict[chr_name] / 2
        x = (start + end) / 2 / self.chr_len_dict[chr_name] * self.chr_width_dict[chr_name] + angle
        return x

    def __get_bottom_dict(self, bottom: Union[int, float, list]) -> dict:
        if isinstance(bottom, list):
            bottom_dict = {
                chr_name: _bottom
                for chr_name, _bottom in zip(self.chr_len_dict.keys(), bottom)
            }
        else:
            bottom_dict = {
                chr_name: bottom
                for chr_name in self.chr_len_dict.keys()
            }
        return bottom_dict

    def __bar_frame(
        self,
        axes: matplotlib.axes.Axes,
        height: Union[int, float],
        bottom: Union[int, float, list],
        face_color: str = 'white',
        edge_color: str = 'black',
        line_width: Union[int, float] = 0.5
    ) -> matplotlib.axes.Axes:
        axes.bar(
            self.chr_theta_dict.values(),
            height,
            self.chr_width_dict.values(),
            bottom=bottom,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=line_width,
        )
        return axes

    def save(self, out_file: str = 'GenomeCircos.pdf'):
        self.figure.savefig(out_file, bbox_inches='tight')

    def chr_bar(
        self,
        bottom: Union[int, float, list],
        height: Union[int, float],
        face_color: str = 'lightgrey',
        edge_color: str = 'black',
        line_width: Union[int, float] = 0.5,
        font_size: Union[int, float] = 6
    ) -> matplotlib.axes.Axes:
        """
        Draw chromosome bar.
        :param height: Bar height of each chromosome.
        :param bottom: Y-axis of bar bottom for each chromosome.
        :param face_color: Face color of each chromosome bar.
        :param edge_color: Edge color of each chromosome bar.
        :param line_width: Line width of each chromosome bar.
        :param font_size: Font size of each chromosome bar.
        :return: tuple(axes: matplotlib.axes.Axes, BarContainer: matplotlib.container.BarContainer)
                 axes -> Axes object in matplotlib.
                 BarContainer -> Chromosome bar container.
        """
        self.figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = self.figure.add_subplot(111, polar=True, frame_on=False)
        ax.get_xaxis().set_visible(False)  # hide x-axis
        ax.get_yaxis().set_visible(False)  # hide y-axis

        # draw chromosome
        ax.bar(
            self.chr_theta_dict.values(),
            height,
            self.chr_width_dict.values(),
            bottom=bottom,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=line_width,
            label='chromosome'
        )

        # set chromosome label
        i = 0
        for chr_name, chr_theta in self.chr_theta_dict.items():
            rotation = 180 / pi * chr_theta - 90 if 0 <= chr_theta < pi else 180 * chr_theta / pi - 270
            _bottom = bottom[i] + height / 2 if isinstance(bottom, list) else bottom + height / 2
            plt.text(
                chr_theta,
                _bottom,
                chr_name,
                rotation=rotation,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=font_size
            )
            i += 1

        return ax

    def bar(
        self,
        axes: matplotlib.axes.Axes,
        stat_file: str,
        bottom: Union[int, float, list],
        height: Union[int, float],
        frame: bool = False
    ) -> matplotlib.axes.Axes:
        """
        Draw the statistical bar chart.
        :param axes: Axes object of matplotlib.axes.Axes.
        :param stat_file: Feature statistics file. (ChrName\\tFeatureType\\tCount\\tColor\\n)
        :param bottom: Y-axis coordinate bottom of statistical bar chart for each chromosome.
        :param height: Height of statistical bar.
        :param frame: Show frame.
        :return: axes -> Axes object of matplotlib.axes.Axes.
        """
        bottom_dict = self.__get_bottom_dict(bottom)
        if frame:
            self.__bar_frame(
                axes=axes,
                height=height + height / 10,
                bottom=[i - 0.05 for i in bottom_dict.values()],
                face_color='white',
                edge_color='black',
                line_width=0.5
            )

        stat_dict = {}  # {Chr01: {'type': [str, ...], 'count': [int, ...], 'color': [str, ...]}, ...}
        all_count = []
        with open(stat_file) as f:
            for line in f:
                split = line.strip().split('\t')
                chr_name, _type, count, color = split[0], split[1], log2(int(split[2]) + 1), split[3]
                all_count.append(count)
                if chr_name in stat_dict:
                    stat_dict[chr_name]['type'].append(_type)
                    stat_dict[chr_name]['count'].append(count)
                    stat_dict[chr_name]['color'].append(color)
                else:
                    stat_dict[chr_name] = {'type': [_type], 'count': [count], 'color': [color]}

        j = 0
        type_set = set()
        for chr_name, d in stat_dict.items():
            total_bar_num = len(d['type']) + 2
            chr_theta, chr_width = self.chr_theta_dict[chr_name], self.chr_width_dict[chr_name]
            bar_width = chr_width / total_bar_num
            x0 = chr_theta - chr_width / 2 + 3 / 2 * bar_width
            x = [x0 + i * bar_width for i in range(len(d['type']))]
            height_list = [d['count'][i] / max(all_count) for i in range(len(d['type']))]
            label = d['type']
            color = d['color']
            for i in range(len(d['type'])):
                if label[i] not in type_set:
                    _label = label[i]
                    type_set.add(_label)
                else:
                    _label = None
                axes.bar(
                    x[i],
                    height_list[i] * height,
                    bar_width,
                    bottom_dict[chr_name],
                    color=color[i],
                    label=_label
                )
            j += 1
        return axes

    def plot(
        self,
        gene_density_file: str,
        axes: matplotlib.axes.Axes,
        bottom: Union[int, float, list],
        height: Union[int, float],
        line_width: Union[int, float],
        color: str = None,
        label: str = 'gene density',
        frame: bool = True
    ) -> matplotlib.axes.Axes:
        """
        Draw gene density with curve.
        :param gene_density_file: Gene density file. (ChrName\\tStart\\tEnd\\tCount\\n)
        :param axes: Axes object of matplotlib.axes.Axes.
        :param bottom: Y-axis coordinate bottom of gene density chart for each chromosome.
        :param height: Curve height.
        :param line_width: Gene density curve width for each chromosome.
        :param color: Gene density plot color.
        :param label: Gene density plot label.
        :param frame: Whether draw borders.
        :return: axes -> Axes object of matplotlib.axes.Axes.
        """
        bottom_dict = self.__get_bottom_dict(bottom)
        df = pandas.read_csv(gene_density_file, sep='\t', header=None, names=['Chr', 'Start', 'End', 'Count'])
        df['theta'] = df['Chr'].map(self.chr_theta_dict)
        df['width'] = df['Chr'].map(self.chr_width_dict)
        df['x_s'] = df['theta'] - df['width'] / 2
        df['x_e'] = df['theta'] + df['width'] / 2
        df['bottom'] = df['Chr'].map(bottom_dict)
        df['y'] = df.groupby('Chr')['Count'].transform(lambda value: value / value.max())
        df['Y'] = df['bottom'] + df['y'] * height

        label_set = set()
        for _, grouped_df in df.groupby('Chr'):
            X = linspace(start=grouped_df.x_s.unique()[0], stop=grouped_df.x_e.unique()[0], num=len(grouped_df))
            Y = grouped_df['Y']
            if label_set:
                _label = None
            else:
                label_set.add(label)
                _label = label
            axes.plot(X, Y, linewidth=line_width, color=color, label=_label)

        if frame:
            self.__bar_frame(
                axes=axes,
                height=height + 0.15,
                bottom=[i - 0.05 for i in bottom_dict.values()],
                face_color='white',
                edge_color='black',
                line_width=line_width
            )
        return axes

    def density_heatmap(
        self,
        gene_density_file: str,
        axes: matplotlib.axes.Axes,
        bottom: Union[int, float, list],
        height: Union[int, float],
        linewidths: float = 0.1,
        cmap: str = 'cool',
        label: str = 'gene density',
        n_min: int = 0,
        n_max: int = 100
    ) -> matplotlib.axes.Axes:
        """
        Draw gene density with line.
        :param gene_density_file: Gene density file. (ChrName\\tStart\\tEnd\\tCount\\n)
        :param axes: Axes object of matplotlib.axes.Axes.
        :param bottom: Y-axis coordinate bottom of gene density chart for each chromosome.
        :param height: Heatmap height.
        :param linewidths: Gene density heatmap curve width for each chromosome.
        :param cmap: Gene density plot color.
        :param label: Gene density plot label.
        :param n_min: The data value mapped to the bottom of the colormap (i.e. 0).
        :param n_max: The data value mapped to the top of the colormap (i.e. 1).
        :return: axes -> Axes object of matplotlib.axes.Axes.
        """
        bottom_dict = self.__get_bottom_dict(bottom)
        df = pandas.read_csv(gene_density_file, sep='\t', header=None, names=['Chr', 'Start', 'End', 'Count'])
        df['X-coordinate'] = df.apply(func=lambda row: self.__loci_to_polar(row.Chr, row.Start, row.End), axis=1)
        for chr_name in df.Chr.unique():
            df.loc[df.Chr == chr_name, 'Y-coordinate'] = bottom_dict[chr_name]

        segments = []
        for x, y in zip(df['X-coordinate'], df['Y-coordinate']):
            line = [(x, y), (x, y + height)]
            segments.append(line)
        # plot heatmap
        cmap = plt.cm.get_cmap(cmap)
        norm = plt.Normalize(vmin=n_min, vmax=n_max)
        colors = cmap(norm(df.Count))
        lc = LineCollection(
            segments,
            colors=colors,
            linewidths=linewidths,
            capstyle='butt',
            alpha=0.8
        )
        axes.add_collection(lc)
        # add color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        figure_width, figure_height = self.figsize
        aspect_ratio = figure_width / figure_height
        cbar = plt.colorbar(
            mappable=sm,
            ax=axes,
            pad=0.0005,
            orientation='horizontal',
            shrink=0.5,
            aspect=aspect_ratio * 80,
            label=label
        )
        cbar.ax.set_xticks([n_min, n_max], ['low', 'high'])
        return axes

# sequence identity heatmap=============================================================================================
    def __transform(self, row: pandas.Series, bottom_dict: dict) -> pandas.Series:
        row['x1'] = self.__loci_to_polar(chr_name=row.Chr1, start=row.Start1, end=row.End1)
        row['x2'] = self.__loci_to_polar(chr_name=row.Chr2, start=row.Start2, end=row.End2)
        row['X'] = (row.x1 + row.x2) / 2
        row['dif_x'] = abs(row.x1 - row.x2)
        row['Y'] = bottom_dict[row.Chr1]
        return row

    def identity_heatmap(
        self,
        cis_acting_file: str,
        matches_cutoff: int,
        axes: matplotlib.axes.Axes,
        bottom: Union[int, float, list],
        height: Union[int, float],
        marker_size: float,
        cmap: str = 'YlOrRd'
    ) -> matplotlib.axes.Axes:
        bottom_dict = self.__get_bottom_dict(bottom)
        raw_df = pandas.read_csv(
            cis_acting_file,
            sep='\t',
            header=None,
            names=['Chr1', 'Start1', 'End1', 'Chr2', 'Start2', 'End2', 'Identity', 'Matches']
        )
        df = raw_df[(raw_df.Chr1 == raw_df.Chr2) & (raw_df.Matches >= matches_cutoff)]
        df = df.apply(
            axis=1,
            func=partial(
                self.__transform,
                bottom_dict=bottom_dict
            )
        )
        df['Y'] += (df['dif_x'] / df['dif_x'].max()) * height
        df['c'] = df['Identity'] * df['Matches']
        df['C'] = df['c'] / df['c'].max() * 100

        ret = axes.scatter(
            x='X',
            y='Y',
            s=marker_size,
            c='C',
            marker='o',
            cmap=cmap,
            data=df,
            label=None
        )

        # add color bar
        figure_width, figure_height = self.figsize
        aspect_ratio = figure_width / figure_height
        cbar = plt.colorbar(
            mappable=ret,
            orientation='horizontal',
            pad=0.005,
            shrink=0.5,
            aspect=aspect_ratio * 100,
            label='Identity'
        )
        cbar.ax.set_xticks([df['C'].min(), df['C'].max()], ['low', 'high'])

        return axes

# link method==================================================================================================
    @staticmethod
    def __coordinates_conversion(theta: float, radius: float) -> List[float]:
        """
        Convert polar coordinates to rectangular coordinates.
        """
        x = radius * cos(theta)
        y = radius * sin(theta)
        return [x, y]

    @staticmethod
    def __bezier_curve(P0, P1, P2) -> tuple:
        """
        Define a second-order Bézier curve.
        :param P0: A point with known coordinates.
        :param P1: The control points.
        :param P2: Another point with known coordinates.
        :return: The coordinates of the 50 points on the Bézier curve.
        """
        P = lambda t: (1 - t) ** 2 * P0 + 2 * t * (1 - t) * P1 + t ** 2 * P2  # Define the Bezier curve
        points = array(
            [P(t) for t in linspace(start=0, stop=1, num=50)]  # Verify Bézier curves at 50 points in the range [0, 1]
        )
        x, y = points[:, 0], points[:, 1]  # Get the x and y coordinates of the points respectively
        return x, y

    def link(
        self,
        axes: matplotlib.axes.Axes,
        chr1: str,
        s1: int,
        e1: int,
        y1: float,
        chr2: str,
        s2: int,
        e2: int,
        y2: float,
        line_width: float = 0.6,
        line_color: str = 'grey',
        label: str = None,
        alpha: float = 0.5
    ) -> matplotlib.axes.Axes:
        """
        Connect the two loci of the genome using a Bézier curve.
        :param axes: Polar coordinates object (type=matplotlib.axes.Axes)
        :param chr1: Chromosome name of loci 1. (type=str)
        :param s1: The start position of the chromosome where locus 1 is located. (type=int)
        :param e1: The end position of the chromosome where locus 1 is located. (type=int)
        :param y1: The y-polar coordinates of point P0. (type=float)
        :param chr2: Chromosome name of loci 2. (type=str)
        :param s2: The start position of the chromosome where locus 2 is located. (type=int)
        :param e2: The end position of the chromosome where locus 2 is located. (type=int)
        :param y2: The y-polar coordinates of point P2. (type=float)
        :param line_width: Line width. (type=float, default=0.6)
        :param line_color: Line color. (type=str, default='grey')
        :param label: Line2D label. (type=str, default=None)
        :param alpha: Line alpha. (type=float, default=0.5)
        :return: axes -> matplotlib.axes.Axes
        """
        # Convert genome loci to polar coordinates radian.
        x1 = self.__loci_to_polar(chr_name=chr1, start=s1, end=e1)
        x2 = self.__loci_to_polar(chr_name=chr2, start=s2, end=e2)
        # Convert polar coordinates to rectangular coordinates.
        P0, P1, P2 = array(
            [
                self.__coordinates_conversion(x1, y1),
                self.__coordinates_conversion((x1 + x2) / 2, 0),
                self.__coordinates_conversion(x2, y2)
            ]
        )
        # Get the coordinates of the points on the Bézier curve.
        x, y = self.__bezier_curve(P0, P1, P2)
        # Draw the Bézier curve.
        axes.plot(
            x,
            y,
            transform=axes.transData._b,
            linewidth=line_width,
            color=line_color,
            label=label,
            alpha=alpha
        )
        return axes

    def links(
        self,
        axes: matplotlib.axes.Axes,
        link_file: str,
        bottom: Union[int, float, list],
        line_width: float = 0.6,
        alpha: float = 0.5
    ) -> matplotlib.axes.Axes:
        """
        Batch connect the two loci of the genome using a Bézier curve.
        :param axes: Polar coordinates object.
        :param link_file: Associated site file. (ChrName\\tStart\\tEnd\\tChrName\\tStart\\tEnd\\tColor\\tLabel)
        :param bottom: Y-axis coordinate of Bézier curve for each chromosome.
        :param line_width: Bézier curve width.
        :param alpha: Bézier curve transparency.
        :return: axes -> matplotlib.axes.Axes.
        """
        color_set = set()
        label_set = set()
        bottom_dict = self.__get_bottom_dict(bottom)
        with open(link_file) as f:
            for line in f:
                split = line.strip().split('\t')
                chr1, start1, end1 = split[0], int(split[1]), int(split[2])
                chr2, start2, end2 = split[3], int(split[4]), int(split[5])
                color, label = split[-2], split[-1]
                if color in color_set and label in label_set:
                    label = None
                color_set.add(color)
                label_set.add(label)
                self.link(
                    axes=axes,
                    chr1=chr1,
                    s1=start1,
                    e1=end1,
                    y1=bottom_dict[chr1],
                    chr2=chr2,
                    s2=start2,
                    e2=end2,
                    y2=bottom_dict[chr2],
                    line_width=line_width,
                    line_color=color,
                    label=label,
                    alpha=alpha
                )
        return axes
