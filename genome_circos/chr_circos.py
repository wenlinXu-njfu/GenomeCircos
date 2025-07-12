"""
File: chr_circos.py
Description: Instance a chromosome example plot object.
CreateDate: 2025/7/9
Author: xuwenlin
E-mail: wenlinxu.njfu@outlook.com
"""
from typing import Union, Tuple, List, Dict
from math import log2, cos, sin
from numpy import pi, linspace, array
import matplotlib
import matplotlib.pyplot as plt


class ChromosomeCircos:
    """
    Draw genome example figure.
    :param chr_len_file: Chromosome length file. (ChrName\\tChrLen\\tEtc)
    :param font: Global font of figure.
    :param figsize: Figure dimension (width, height) in inches.
    :param dpi: Dots per inch.
    """
    def __init__(
        self,
        chr_len_file: str,
        font: str = None,
        figsize: Tuple[float, float] = (10, 8),
        dpi: int = 100
    ):
        chr_len_dict, chr_theta_dict, chr_width_dict = self.__get_chr_theta_width(chr_len_file)
        self.chr_len_dict = chr_len_dict
        self.chr_theta_dict = chr_theta_dict
        self.chr_width_dict = chr_width_dict
        self.font = font
        self.figsize = figsize
        self.dpi = dpi

    @staticmethod
    def __get_chr_theta_width(chr_len_file: str) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
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
        each_space_len = min(raw_chr_len_dict.values()) / 6
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

    def __get_bottom_dict(self, bottom: Union[int, float, list]):
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
    ):
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

    def chr_bar(
        self,
        height: Union[int, float] = 1.0,
        bottom: Union[int, float, list] = 10.0,
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
        plt.rcParams['pdf.fonttype'] = 42
        if self.font:
            plt.rcParams['font.family'] = self.font
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, polar=True, frame_on=False)
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
        bottom: Union[int, float, list] = 11.05,
        frame: bool = False
    ) -> matplotlib.axes.Axes:
        """
        Draw the statistical bar chart.
        :param axes: Axes object of matplotlib.axes.Axes.
        :param stat_file: Feature statistics file. (ChrName\\tFeatureType\\tCount\\tColor\\n)
        :param bottom: Y-axis coordinate bottom of statistical bar chart for each chromosome.
        :param frame: Show frame.
        :return: axes -> Axes object of matplotlib.axes.Axes.
        """
        bottom_dict = self.__get_bottom_dict(bottom)
        if frame:
            self.__bar_frame(
                axes=axes,
                height=1.1,
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
            height = [d['count'][i] / max(all_count) for i in range(len(d['type']))]
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
                    height[i],
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
        bottom: Union[int, float, list] = 8.5,
        line_width: Union[int, float] = 0.3,
        color: str = None,
        label: str = 'gene density',
        frame: bool = True
    ) -> matplotlib.axes.Axes:
        """
        Draw gene density with line.
        :param gene_density_file: Gene density file. (ChrName\\tStart\\tEnd\\tCount\\n)
        :param axes: Axes object of matplotlib.axes.Axes.
        :param bottom: Y-axis coordinate bottom of gene density chart for each chromosome.
        :param line_width: Gene density curve width for each chromosome.
        :param color: Gene density plot color.
        :param label: Gene density plot label.
        :param frame: Whether draw borders.
        :return: axes -> Axes object of matplotlib.axes.Axes.
        """
        y = []
        line_count = 0
        label_set = set()
        current_chr = None
        bottom_dict = self.__get_bottom_dict(bottom)
        total_line = sum(1 for _ in open(gene_density_file))
        with open(gene_density_file) as f:
            for line in f:
                line_count += 1
                split = line.strip().split('\t')
                if line_count != total_line:
                    if current_chr is None:
                        current_chr = split[0]
                        y.append(float(split[-1]))
                    else:
                        if current_chr != split[0]:
                            current_bottom = bottom_dict[current_chr]
                            x = linspace(
                                start=self.chr_theta_dict[current_chr] - self.chr_width_dict[current_chr] / 2,
                                stop=self.chr_theta_dict[current_chr] + self.chr_width_dict[current_chr] / 2,
                                num=len(y)
                            )
                            y = [current_bottom + 0.05 + i / max(y) for i in y]
                            if label not in label_set:
                                _label = label
                                label_set.add(_label)
                            else:
                                _label = None
                            axes.plot(x, y, linewidth=line_width, color=color, label=_label)
                            current_chr = split[0]
                            y = [float(line.strip().split('\t')[-1])]
                        else:
                            y.append(float(line.strip().split('\t')[-1]))
                else:
                    current_bottom = bottom_dict[current_chr]
                    x = linspace(
                        start=self.chr_theta_dict[current_chr] - self.chr_width_dict[current_chr] / 2,
                        stop=self.chr_theta_dict[current_chr] + self.chr_width_dict[current_chr] / 2,
                        num=len(y)
                    )
                    y = [current_bottom + 0.05 + i / max(y) for i in y]
                    axes.plot(x, y, linewidth=line_width, color=color)
        if frame:
            self.__bar_frame(
                axes=axes,
                height=1.15,
                bottom=[i - 0.05 for i in bottom_dict.values()],
                face_color='white',
                edge_color='black',
                line_width=0.5
            )
        return axes

# link method===========================================================================================================
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

    @staticmethod
    def __coordinates_conversion(theta: float, radius: float) -> List[float]:
        """
        Convert polar coordinates to rectangular coordinates.
        """
        x = radius * cos(theta)
        y = radius * sin(theta)
        return [x, y]

    @staticmethod
    def __bezier_curve(P0, P1, P2):
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
        bottom: Union[int, float, list] = 7,
        line_width: float = 0.6,
        alpha: float = 0.5
    ):
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
