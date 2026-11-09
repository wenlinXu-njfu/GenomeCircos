[![PyPI - Version](https://img.shields.io/pypi/v/genome-circos)](https://pypi.org/project/genome-circos/)
<img src="https://github.com/wenlinXu-njfu/GenomeCircos/blob/main/logo.png" align="right" width="30%">
# GenomeCircos: Genome circos diagram python package

## Install
```shell
pip install genome-circos --upgrade
```

## Usage example
**See [UserGuide.md](https://github.com/wenlinXu-njfu/GenomeCircos/blob/main/UserGuide.md)**

## Change logs
### v0.0.4
- New
  - New methods of ChromosomeCircos class
    - link_region: connect the starting and ending positions of two loci in the genome with two second-order Bézier curves and fill in the regions.
    - link_regions: the batch version of `link_region`
  - New parameter of `plot` method:
    - line_alpha: line transparency.
    - normalize: gene density normalization method (intra for intrachromosomal or inter for interchromosomal).
  - New parameter of `density_heatmap` method:
    - font_size: color bar ticks font size.
    - orientation: color bar orientation.
    - anchor: the anchor point of colorbar. (used for anchoring the position of the color bar)
- Modified
  - Modify thr effect of `frame` parameter of `plot` method:
    - not only can it add edge lines, but it can also fill in the background color.
- Deleted
  - Null
### v0.0.3
- New
  - New parameter of `ChromosomeCircos`:
    - font_size: global font size.
  - New parameter of `bar` method:
    - height: height of statistical bar.
  - New parameter of `plot` method:
    - height: curve height.
  - New method of `ChromosomeCircos`:
    - identity_heatmap: show sequence identity on each chromosome.
- Modified
  - Rename the `plot2` method to "density_heatmap`.
- Deleted
  - Null
### v0.0.2
- New
  - New attributions of `ChromosomeCircos`:
    - spacing: set the spacing between chromosomes to be one quarter (spacing=4) of the shortest chromosome.
  - New methods of `ChromosomeCircos`:
    - plot2: show feature density using heatmap.
    - save: save the image.
- Modified
  - Null
- Deleted
  - Null
