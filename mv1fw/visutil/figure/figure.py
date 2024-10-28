


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d import Axes3D
from math import isnan

from .._impl.types import (
    set_rc_serif,
)
from ..._impl.types import (
    tag_filename_front,
    sortdown,
)

from .plot import \
    plot_scatterplot, \
    plot_scatterplot3d, \
    plot_series, \
    plot_multiseries, \
    plot_errorbarseries, \
    plot_heatmap






class Figure:
    """
    Class containing methods for generating
    figures and housing default settings.
    Actions that wish to can re-instantiate this
    class and set their own settings for the
    parameters.

    Parameters:

        figsize:
            figsize=(5,4) => 5in by 4in
        dpi:
            figsize=(x,y),
            dpi=100 => x*100 by y*100 pixels in the generated image.
        scatter_point2_default:
            Size of the points in scatterplots,
            in point^2 (typographic points are 1/72 in.)
        cmap_default:
            colormap for heatmap plots.
            See e.g. `matplotlib documentation <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.

    """

    def __init__(
        self,
        figsize = (5, 4),
        dpi = 500,
        scatter_point2_default = 2,
        cmap_default = "plasma",
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.scatter_point2_default = scatter_point2_default
        self.cmap_default = cmap_default
        set_rc_serif()

        # deprecated:
        # visualization of sample sets
        # possible colors: {b,g, r, c (cyan), m (magenta), y (yellow), k (black), w (white)}
        # possible point styles: {. (point) , (pixel) o (circle) v ^ < > (triangle down, up, left, right) ...(see plot cmd docs)}
        # self.fmt1 = ".k"

        self.sampleset_colors = [
            "chocolate", "mediumturquoise", "gold", "cornflowerblue",
            "lime", "midnightblue", "lightcoral", "teal",
            "chocolate", "mediumturquoise", "gold", "cornflowerblue",
            "lime", "midnightblue", "lightcoral", "teal",
        ]


    def matplotlib_tools(
            self,
            n = 1,
            nj = 0,
    ):
        """
        Return the Matplotlib artifacts for
        custom plot-building.

        When you
        are done building, do::

            # fill the available space with the plot
            ax.set_aspect(aspect='auto', adjustable='datalim')
            fig.suptitle(title)
            fig.savefig(filename)
            plt.close()

        Arguments:

            n (integer):
                square integer setting number of plots
            nj (integer):
                additional figures arrayed horizontally,

        Returns:

            pair consisting of Matplotlib ``figure``
            and a tuple of Matplotlib axes.
            For example, ``n=2, nj=1`` gives these schematic locations::

                0 1 2
                3 4 5

        """
        # Set up figure and image grid
        fig = plt.figure(
            figsize=self.figsize,
            dpi=self.dpi,
        )
        grid = ImageGrid(
            fig=fig,
            # location of grid
            # 111 = top left cell of fig, ie the normal place
            rect=111,
            # allow control of aspects of axes in the "ax" loop below
            aspect=False,
            # rect=(0.1, 0.1, 0.8, 0.8),
            nrows_ncols=(n,n+nj),
            axes_pad=0.28,
        )
        axes = grid
        return fig, axes


    def close(self):
        """
        Can be used along with matplotlib_tools().

        """
        plt.close()



    def scatterplot_grid(
            self,
            filename,
            n,
            nj = 0,
            title = None,
            title_s = None,
            X_s = None,
            x_s = None,
            xlim_s = None,
            ylim_s = None,
            Xs_s = None,
            xs_s = None,
            Xslabels_s = None,
            vlines_s = None,
            vlineslim_s = None,
    ):
        """
        Generate a figure consisting
        of cells, each with a single scatterplot.
        Cf. :any:`Figure.scatterplot`.
        The arguments are lists
        that correspond index-wise
        to scatterplots in the figure
        going across and then down.
        Example::

            0, 1, 2
            3, 4, 5
            6, 7, 8

        The ``subtitles`` field is
        a list of strings corresponding
        to titles placed above each
        of the plots in the grid,
        using the same indexing convention
        as above.

        Generates a n x (n+nj) grid (height x length).
        The default nj is zero, so
        a square grid only requires n.
        For example::

            n=2 # (2x2)
            n=3 # (3x3)

        """
        # set up subtitles
        if n <= 0:
            raise ValueError
        if not isinstance(title_s, list) and not title_s is None:
            subtitles_ = [title_s]
        else:
            subtitles_ = title_s
        # Set up figure and image grid
        fig = plt.figure(
            figsize=self.figsize,
            dpi=self.dpi,
        )
        grid = ImageGrid(
            fig=fig,
            # location of grid
            # 111 = top left cell of fig, ie the normal place
            rect=111,
            # allow control of aspects of axes in the "ax" loop below
            aspect=False,
            # rect=(0.1, 0.1, 0.8, 0.8),
            nrows_ncols=(n,n+nj),
            axes_pad=0.28,
        )
        # Add data to image grid
        for i, ax in enumerate(grid):
            title_ = None if subtitles_ is None else subtitles_[i]
            # scatter returns a pathcollection object
            plot_scatterplot(
                ax=ax,
                title=title_,
                X = X_s[i],
                x = x_s[i],
                xlim = xlim_s[i],
                ylim = ylim_s[i],
                Xs = Xs_s[i],
                xs = xs_s[i],
                Xslabels=Xslabels_s[i],
                vlines = vlines_s[i],
                vlineslim = vlineslim_s[i],
            )
            # fill the available space with the plot
            ax.set_aspect(aspect='auto', adjustable='datalim')
        fig.suptitle(title)
        fig.savefig(filename)
        plt.close()



    def scatterplot(
            self,
            filename = "out.png",
            title = None,
            X = None,
            x = None,
            xlim = None,
            ylim = None,
            Xs = None,
            xs = None,
            Xslabels=None,
            vlines = None,
            vlineslim = None,
    ):
        """
        Generate a figure consisting
        of a single scatterplot.

        Arguments:

            filename (string):
                path to save to (Default: ``out.png``)
            X (numpy array):
                Plot coordinates ([:,0], [:,i]) for all i > 0 if x is not set,
                otherwise plot (x, [:,i]) for all i ≥ 0.
            x (float):
                an optional constant x-axis value to pin values to.
            title (string):
                optional title of plot
            xlim (pair of float):
                optional xlimits
            ylim (pair of float):
                optional ylimits
            Xs (list of numpy array):
                Pair (Xs[i], xs[i]) works the same as (X, x) for
                a basic scatterplot and it will add to (X, x), it
                does not replace (X, x).
                Use if you have more point sets
                you wish to see in the same field.
            xs (list of float):
                See Xs.
                todo document whether xs must be provided to use Xs
            Xslabels:
                labels to assign to sets in Xs that are plotted
            vlines:
                optional vlines, set ylims to plot vlines.
                Examples: (x1, x2, x3,) or (x1,)
            vlineslim:
                optional limits to vlines, required if vlines != None.
        """
        self.scatterplot_grid(
            filename,
            n = 1,
            title = title,
            title_s = None,
            X_s = [X],
            x_s = [x],
            xlim_s = [xlim],
            ylim_s = [ylim],
            Xs_s = [Xs],
            xs_s = [xs],
            Xslabels_s=[Xslabels],
            vlines_s = [vlines],
            vlineslim_s = [vlineslim],
        )




    def series_grid(
            self,
            filename,
            n,
            nj,
            title_s,
            X_s,
            inlabel_s,
            inidx_s,
            outlabels_s,
            outidxs_s,
            title = None,
            text = None,
            t_s=None,
            xlim_s=None,
            ylim_s=None,
            vlines_s=None,
            vlineslim_s=None,
            X_black_s=None,
            X_ref_s=None,
            reorder=False,
            half_linestyle=None,
            half_color=None,
            marker=None,
            half_marker=None,
    ):
        """
        Generate a figure consisting
        of a series or grid of series.
        Each series can have several
        output variables.
        (A "series" is an ordinary x-y plot.)

        See series() method for arguments.

        """
        # set up subtitles
        if n <= 0:
            raise ValueError
        if not isinstance(title_s, list) and not title_s is None:
            subtitles_ = [title_s]
        else:
            subtitles_ = title_s
        # Set up figure and image grid
        fig = plt.figure(
            figsize=self.figsize,
            dpi=self.dpi,
        )
        grid = ImageGrid(
            fig=fig,
            # location of grid
            # 111 = top left cell of fig, ie the normal place
            rect=111,
            # allow control of aspects of axes in the "ax" loop below
            aspect=False,
            # rect=(0.1, 0.1, 0.8, 0.8),
            nrows_ncols=(n,n+nj),
            axes_pad=0.28,
            # axes_pad=0.35,
            # ^--- this value adds more padding between
            # the subplots in the grid, as well as around
            # the outside on left and right, last I checked.
        )
        # Add data to image grid
        for i, ax in enumerate(grid):
            X_ = X_s[i]
            if reorder:
                X_ = sortdown(X_, inidx_s[i])
            title_ = None if subtitles_ is None else subtitles_[i]
            # scatter returns a pathcollection object
            plot_series(
                ax=ax,
                title=title_,
                X = X_,
                inlabel = inlabel_s[i],
                inidx = inidx_s[i],
                outlabels = outlabels_s[i],
                outidxs = outidxs_s[i],
                t = t_s[i] if t_s is not None else None,
                xlim = xlim_s[i] if xlim_s is not None else None,
                ylim = ylim_s[i] if ylim_s is not None else None,
                vlines = vlines_s[i] if vlines_s is not None else None,
                vlineslim = vlineslim_s[i] if vlineslim_s is not None else None,
                X_black = X_black_s[i] if X_black_s is not None else None,
                X_ref = X_ref_s[i] if X_ref_s is not None else None,
                half_linestyle = half_linestyle,
                half_color = half_color,
                marker=marker,
                half_marker=half_marker,
            )
            # fill the available space with the plot
            ax.set_aspect(aspect='auto', adjustable='datalim')
        fig.suptitle(title)
        if text is not None and text != "":
            # the meaning of the "bottom" is rather strange IME.
            # 0.1 cuts off the padding (clips the figure),
            # 0.2 is too much padding,
            # for a caption that is one line.
            # 0.15 is good.
            fig.subplots_adjust(bottom=0.151)
            fig.text(
                x=0.5,
                y=0.01,
                s=text,
                horizontalalignment="center",
            )
        fig.savefig(filename)
        plt.close()


    def series(
            self,
            filename,
            X,
            inlabel,
            outlabels,
            inidx = None,
            outidxs = None,
            title = None,
            text = None,
            t=None,
            xlim=None,
            ylim=None,
            vlines=None,
            vlineslim=None,
            X_black=None,
            X_ref=None,
            reorder=False,
            half_linestyle=None,
            half_color=None,
            marker=None,
            half_marker=None,
    ):
        """
        Generate a figure consisting
        of a single series.
        (A "series" is an ordinary x-y plot,
        what is called "plot" in Matplotlib.)
        Multiple curves can be present in the series,
        specified by "outlabels" and "outidxs".

        Arguments:

            filename (string):
                Path to store artifact.
            X (array):
                Array of shape [n, m] of points in the plane,
                where n is the number of points, and m ≥ 2.
                It may be that m = 2 and the array represents n pairs.
                Otherwise it represents m - 1 sets of n points in the plane.
                Ensure the x-axis values are in increasing order,
                or else set `reorder=True`.
            inlabel (string):
                label for input (x axis)
            inidx (integer):
                index for input, almost always 0.
            outlabels (list of string):
                list of labels for output.
            outidxs (list of integer):
                list of indices in ndarray to plot as outputs.
            title (optional string):
                A title string you may construct
            text (optional string):
                text to add to the figure (caption)
            t (optional scalar):
                optional time for a plot of a timeslice, it will modify the title.
            xlim (optional pair of scalars):
                optional xlimits list: (bottom, top).
            ylim (optional pair of scalars):
                optional ylimits list: (bottom, top).
            vlines (optional tuple of scalars):
                optional vlines, set ylims to plot vlines.
                Examples: (x1, x2, x3,) or (x1,)
            vlineslim (optional tuple of scalar):
                optional limits to vlines, required if vlines != None.
            X_black:
                Additional plot series can be passed in here, they will be
                overlayed, printed in black.
            X_ref:
                optional ndarray containing a reference curve to the
                main series ndarray.
                Ensure it is the same shape as X; we won't check you.
            reorder:
                Boolean, whether to sort the data down the x-axis
                before passing to the plotter.
                The plotting method connects (x,y) points in series,
                as they appear in the input array, so sorting beforehand
                may be necessary but is an unneeded step if the data is
                already sorted, so this step must be explicitly requested.
                Default: False
            half_linestyle:
                Optional string, whether to repeat the colors on the N = 2n inputs
                (if N is odd, the last input will dangle) and instead distinguish via
                the indicated line style.
                Can be used for reference data plots. String can be:
                {"solid", "dotted", "dashed", "dashdot", densely dotted", ...},
                see pyplot linestyles reference for more options.
                Default: None
            half_color:
                Optional string, color of latter half of plotted curves.
                Default: None
            marker:
                Optional character or integer indicating a marker for individual data points.
                Character an be:
                {'x', '+', '1', '2', '3', '4', '_', '|'}
                See pyplot documentation for integer markers or official docs.
                Default: None
            half_marker:
                Optional character or integer indicating a marker for individual data points,
                to be used on the back half. (See marker argument.)
                In order to be applied, half_linestyle must be set, otherwise
                the series are not halved.
        """
        self.series_grid(
            filename,
            n = 1,
            nj = 0,
            title_s = None,
            X_s = [X],
            inlabel_s = [inlabel],
            inidx_s = [inidx] if inidx is not None else [0],
            title = title,
            text = text,
            outlabels_s = [outlabels],
            outidxs_s = [outidxs] if outidxs is not None else [[1]],
            t_s=[t],
            xlim_s=[xlim],
            ylim_s=[ylim],
            vlines_s=[vlines],
            vlineslim_s=[vlineslim],
            X_black_s=[X_black],
            X_ref_s=[X_ref],
            reorder=reorder,
            half_linestyle=half_linestyle,
            half_color=half_color,
            marker=marker,
            half_marker=half_marker,
        )




    def multiseries_grid(
        self,
        filename,
        n,
        nj,
        XYs_s,
        inlabel_s,
        outlabels_s,
        title = None,
        text = None,
        title_s = None,
        xlim_s = None,
        ylim_s = None,
    ):
        """
        Plot varying input sequences X (in a common dimension),
        versus varying output sequences Y (in a common dimension),
        or in other words, a list (XY1, XY2, ...) of sets XY of (x,y) pairs.
        This is referred to as a "multiseries", whereas a "series" (XY1Y2Y3...)
        has a common set of inputs X common to all output sequences Y.

        See multiseries() method for arguments.

        """
        # set up subtitles
        if n <= 0:
            raise ValueError
        if not isinstance(title_s, list) and not title_s is None:
            subtitles_ = [title_s]
        else:
            subtitles_ = title_s
        # Set up figure and image grid
        fig = plt.figure(
            figsize=self.figsize,
            dpi=self.dpi,
        )
        grid = ImageGrid(
            fig=fig,
            # location of grid
            # 111 = top left cell of fig, ie the normal place
            rect=111,
            # allow control of aspects of axes in the "ax" loop below
            aspect=False,
            # rect=(0.1, 0.1, 0.8, 0.8),
            nrows_ncols=(n,n+nj),
            axes_pad=0.28,
            # axes_pad=0.35,
            # ^--- this value adds more padding between
            # the subplots in the grid, as well as around
            # the outside on left and right, last I checked.
        )
        # Add data to image grid
        for i, ax in enumerate(grid):
            XYs_ = XYs_s[i]
            title_ = None if subtitles_ is None else subtitles_[i]
            # scatter returns a pathcollection object
            plot_multiseries(
                ax=ax,
                title=title_,
                XYs = XYs_,
                inlabel = inlabel_s[i],
                outlabels = outlabels_s[i],
                xlim = xlim_s[i] if xlim_s is not None else None,
                ylim = ylim_s[i] if ylim_s is not None else None,
            )
            # fill the available space with the plot
            ax.set_aspect(aspect='auto', adjustable='datalim')
        fig.suptitle(title)
        if text is not None and text != "":
            # the meaning of the "bottom"-
            # 0.1 cuts off the padding (clips the figure),
            # 0.2 is too much padding,
            # for a caption that is one line 0.15 is ok.
            fig.subplots_adjust(bottom=0.151)
            fig.text(
                x=0.5,
                y=0.01,
                s=text,
                horizontalalignment="center",
            )
        fig.savefig(filename)
        plt.close()



    def multiseries(
            self,
            filename,
            XYs,
            inlabel,
            outlabels,
            title = None,
            text = None,
            xlim = None,
            ylim = None,
    ):
        """
        Plot varying input sequences X (in a common dimension),
        versus varying output sequences Y (in a common dimension),
        or in other words, a list (XY1, XY2, ...) of sets XY of (x,y) pairs.
        This is referred to as a "multiseries", whereas a "series" (XY1Y2Y3...)
        has a common set of inputs X common to all output sequences Y.

        Arguments:

            filename (string):
                target path
            XYs (list of Nx2 array):
                List of arrays XY = [x, y]. Each XY defines N (x,y) pairs.
                associates input value x to value/output/response y.
                The number N may vary between arrays, and
                the sequence X may vary between arrays.
                It is assumed that the sequence X increases down the array.
            inlabel (string):
                input label, assigned to x-axis and the 'type' of all values X.
            outlabels (list of string):
                output labels, in the identical order appearing in XYs: the
                ith label is assigned to the ith Y.
            title (optional string):
                top line title.
            text (optional string):
                text, to place in bottom gutter.
            xlim (optional pair of scalar):
                x-axis limits.
            ylim (optional pair of scalar):
                y-axis limits.

        """
        self.multiseries_grid(
            filename,
            n = 1,
            nj = 0,
            XYs_s = [XYs],
            inlabel_s = [inlabel],
            outlabels_s = [outlabels],
            title = title,
            text = text,
            title_s = None,
            xlim_s=[xlim],
            ylim_s=[ylim],
        )





    def errorbarseries_grid(
        self,
        filename,
        n,
        nj,
        X_s,
        center_s = None,
        title = None,
        title_s = None,
        error_idx_s = None,
        inlabel_s = None,
        inticklabels_s = None,
        outlabel_s = None,
        text = None,
        fmt_s = None,
        linewidth_s = None,
        capsize_s = None,
        xlim_s = None,
        xticks_s = None,
        ylim_s = None,
        yticks_s = None,
        fmt_999_s = None,
    ):
        """
        todo

        See errorbarseries() method for arguments.

        """
        # set up subtitles
        if n <= 0:
            raise ValueError
        if not isinstance(title_s, list) and not title_s is None:
            subtitles_ = [title_s]
        else:
            subtitles_ = title_s
        # Set up figure and image grid
        fig = plt.figure(
            figsize=self.figsize,
            dpi=self.dpi,
        )
        grid = ImageGrid(
            fig=fig,
            # location of grid
            # 111 = top left cell of fig, ie the normal place
            rect=111,
            # allow control of aspects of axes in the "ax" loop below
            aspect=False,
            # rect=(0.1, 0.1, 0.8, 0.8),
            nrows_ncols=(n,n+nj),
            axes_pad=0.28,
            # axes_pad=0.35,
            # ^--- this value adds more padding between
            # the subplots in the grid, as well as around
            # the outside on left and right, last I checked.
        )
        # Add data to image grid
        for i, ax in enumerate(grid):
            X_ = X_s[i]
            title_ = None if subtitles_ is None else subtitles_[i]
            # scatter returns a pathcollection object
            plot_errorbarseries(
                ax=ax,
                X = X_,
                center = center_s[i] if center_s is not None else None,
                error_idx = error_idx_s[i] if error_idx_s is not None else None,
                title=title_,
                inlabel = inlabel_s[i],
                inticklabels = inticklabels_s[i] if inticklabels_s is not None else None,
                outlabel = outlabel_s[i],
                fmt = fmt_s[i] if fmt_s is not None else 'none',
                linewidth = linewidth_s[i] if linewidth_s is not None else None,
                capsize = capsize_s[i] if capsize_s is not None else None,
                xlim = xlim_s[i] if xlim_s is not None else None,
                ylim = ylim_s[i] if ylim_s is not None else None,
                xticks = xticks_s[i] if xticks_s is not None else None,
                yticks = yticks_s[i] if yticks_s is not None else None,
                fmt_999 = fmt_999_s[i] if fmt_999_s is not None else False,
            )
            # fill the available space with the plot
            ax.set_aspect(aspect='auto', adjustable='datalim')
        fig.suptitle(title)
        if text is not None and text != "":
            # the meaning of the "bottom"-
            # 0.1 cuts off the padding (clips the figure),
            # 0.2 is too much padding,
            # for a caption that is one line 0.15 is ok.
            fig.subplots_adjust(bottom=0.151)
            fig.text(
                x=0.5,
                y=0.01,
                s=text,
                horizontalalignment="center",
            )
        fig.savefig(filename)
        plt.close()




    def errorbarseries(
            self,
            filename,
            X = None,
            center = None,
            error_idx = None,
            # FORMAT
            title = None,
            inlabel = None,
            inticklabels = None,
            outlabel = None,
            text = None,
            fmt = 'none',
            linewidth = 2,
            capsize = 3,
            xlim = None,
            xticks = None,
            ylim = None,
            yticks = None,
            fmt_999 = False,
    ):
        """
        todo

        Arguments:

            filename:
            X:
            center:
            error_idx:
            title:
            inlabel:
            inticklabels (optional list of string):
            outlabel:
            text:
            fmt:
            linewidth:
            capsize:
            xlim:
            xticks:
            ylim:
            yticks:
            fmt_999 (boolean):
                A custom format for a 90-95-99
                confidence interval plot, cf. :any:`Noner`
        """
        self.errorbarseries_grid(
            filename=filename,
            n=1,
            nj=0,
            X_s=[X],
            center_s = [center],
            error_idx_s = [error_idx],
            title=title,
            title_s=None,
            inlabel_s=[inlabel],
            inticklabels_s = [inticklabels],
            outlabel_s=[outlabel],
            text=text,
            fmt_s = [fmt],
            linewidth_s = [linewidth],
            capsize_s = [capsize],
            xlim_s = [xlim],
            xticks_s = [xticks],
            ylim_s = [ylim],
            yticks_s = [yticks],
            fmt_999_s = [fmt_999],
        )






    def heatmap_grid(
            self,
            filename,
            n,
            nj = 0,
            title = None,
            t = None,
            color_label = None,
            plot_xray = True,
            method = "nearest",
            title_s = None,
            X_s = None,
            in1_s = None,
            in2_s = None,
            out1_s = None,
            lbl_s = None,
            value_range_s = None,
            t_s = None,
            xlim_s = None,
            ylim_s = None,
            vlines_s = None,
            vlineslim_s = None,
            X_black_s = None,
            X_black_linewidth_s = None,
            size_interpolate_2D = None,
            colormap = None,
    ):
        """
        Generate a figure consisting
        of a heatmap or grid of heatmaps.

        See heatmap() for argument description.

        todo documentation

        """
        if n <= 0:
            raise ValueError
        if not isinstance(title_s, list) and not title_s is None:
            title_s_ = [title_s]
        else:
            title_s_ = title_s
        # Set up figure and image grid
        fig = plt.figure(
            figsize=self.figsize,
            dpi=self.dpi,
        )
        grid = ImageGrid(
            fig=fig,
            # location of grid
            # 111 = top left cell of fig, ie the normal place
            rect=111,
            # rect=(0.1, 0.1, 0.8, 0.8),
            nrows_ncols=(n,n + nj),
            axes_pad=0.28,
            share_all=True,
            # allow control of aspects of axes in the "ax" loop below
            aspect=False,
            cbar_location="right",
            # whether to have a colorbar for each axis (plot) or one for all
            cbar_mode="single",
            # "thickness" of the colorbar
            cbar_size="4%",
            # pad between plots and colorbar
            cbar_pad=0.16,
        )
        ax0 = None
        im0 = None
        # Add data to image grid
        for i, ax in enumerate(grid):
            ax0 = ax
            X = X_s[i]
            in1 = in1_s[i]
            in2 = in2_s[i]
            out1 = out1_s[i]
            lbl = lbl_s[i]
            title_ = None if title_s_ is None else title_s_[i]
            value_range = value_range_s[i] if value_range_s is not None else None
            t_ = None if t_s is None else t_s[i]
            xlim = None if xlim_s is None else xlim_s[i]
            ylim = None if ylim_s is None else ylim_s[i]
            vlines = None if vlines_s is None else vlines_s[i]
            vlineslim = None if vlineslim_s is None else vlineslim_s[i]
            X_black = None if X_black_s is None else X_black_s[i]
            X_black_linewidth = None if X_black_linewidth_s is None else X_black_linewidth_s[i]
            im0 = plot_heatmap(
                ax=ax0,
                title=title_,
                X=X,
                in1=in1,
                in2=in2,
                out1=out1,
                lbl=lbl,
                value_range=value_range,
                method=method,
                size_interpolate_2D=size_interpolate_2D,
                t=t_,
                xlim=xlim,
                ylim=ylim,
                vlines=vlines,
                vlineslim=vlineslim,
                X_black=X_black,
                X_black_linewidth=X_black_linewidth,
                cmap = colormap,
            )
            # fill the available space with the plot
            ax0.set_aspect(aspect='auto', adjustable='datalim')
        # Colorbar
        ax0.cax.colorbar(im0, label=color_label)
        if title is not None or t is not None:
            title_ = f"{title}, t = {t:.4f}" if title is not None and t is not None else f"t = {t:.4f}" if title is None and t is not None else title
            fig.suptitle(title_)
        fig.savefig(filename)
        plt.close()
        if plot_xray:
            # just emit one xray plot,
            # assuming a consistent sample set
            # at all plots on the grid.
            # Later possibly add more code
            # for when this assumption is false.
            i = 0
            X = X_s[i]
            in1 = in1_s[i]
            in2 = in2_s[i]
            lbl = lbl_s[i]
            xlim = None if xlim_s is None else xlim_s[i]
            ylim = None if ylim_s is None else ylim_s[i]
            vlines = None if vlines_s is None else vlines_s[i]
            vlineslim = None if vlineslim_s is None else vlineslim_s[i]
            filename_xray = tag_filename_front(filename, "xray")
            plt.scatter(X[:,in1:(in1+1)].tolist(), X[:,in2:(in2+1)].tolist(), s=self.scatter_point2_default)
            if lbl is not None:
                plt.xlabel(lbl[in1])
                plt.ylabel(lbl[in2])
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
            if vlines is not None:
                plt.vlines(vlines, vlineslim[0], vlineslim[1], linestyles="dashed", colors="black")
            if title is not None:
                plt.title(title + " xray")
            plt.tight_layout()
            plt.savefig(filename_xray)
            plt.close()



    def heatmap(
            self,
            filename,
            X,
            in1,
            in2,
            out1,
            lbl,
            title,
            value_range,
            color_label = None,
            plot_xray = False,
            method = "nearest",
            t=None,
            xlim=None,
            ylim=None,
            vlines = None,
            vlineslim = None,
            X_black = None,
            X_black_linewidth = None,
            size_interpolate_2D = None,
            colormap = None,
    ):
        """
        Plot as heatmap.
        Create xy field using index x=in1, y=in2,
        reading into X.
        Use output variable u=out1 for color/heat.
        Pass in a consistent value_range if desired.

        Arguments:

            filename:
                Full path of destination in file system.
            X (array):
            in1 (integer):
                index of X
            in2 (integer):
                index of X
            out1 (integer):
                index of X
            lbl (list of string):
            title (string):
            color_label (string):
            value_range: (optional [lower,upper])
                Pass in value_range if a consistent colorbar
                (assignment of color to value) is needed.
            plot_xray (bool):
                When plotting a heatmap, generate this plot
                to see what points the heatmap was generated from.
                Setting this to true is recommended for
                qualitative review, or for debugging.
            method (string):
                method of interpolation. Can be:
                antialiased, nearest, linear, bicubic, ...
            t:
                optional time for a plot of a timeslice, it will modify the title.
            xlim:
                optional xlimits list: (bottom, top).
            ylim:
                optional ylimits list: (bottom, top).
            vlines:
                optional vlines, set ylims to plot vlines.
                Examples: (x1, x2, x3,) or (x1,)
            vlineslim:
                optional limits to vlines, required if vlines != None.
            X_black:
                values to plot on the heatmap as black lines. The 0th index
                is used for x values and other indices are used for y values.
            X_black_linewidth:
                linewidth of X_black lines in points. Matplotlib default is
                1.5 point according to
                `this <https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html#plot>`_.
            size_interpolate_2D:
                None for default value, otherwise a size for interpolation
                to a pixel value grid for heatmap plot,
                suggested max 400, suggested min 50 (?).
            colormap:
                The colormap to use for visualizing the heatmap plot.
        """
        self.heatmap_grid(
            filename=filename,
            n = 1,
            nj = 0,
            title = title,
            color_label = color_label,
            plot_xray = plot_xray,
            method = method,
            title_s = None,
            X_s = [X],
            in1_s = [in1],
            in2_s = [in2],
            out1_s = [out1],
            lbl_s = [lbl],
            value_range_s = [value_range],
            t_s = [t],
            xlim_s = [xlim],
            ylim_s = [ylim],
            vlines_s = [vlines],
            vlineslim_s = [vlineslim],
            X_black_s = [X_black],
            X_black_linewidth_s = [X_black_linewidth],
            size_interpolate_2D = size_interpolate_2D,
            colormap = colormap,
        )





    def scatterplot3d(
        self,
        filename,
        Xs,
        in1,
        in2,
        in3,
        lbl,
        title = None,
        show = True,
    ):
        """

        Arguments:

            filename:
                Full path of destination in file system.
            Xs:
            in1:
            in2:
            in3:
            lbl:
            title:
            show:
                Whether or not to stop the process
                while a pyplot window opens
                to allow you to review the plot
                in a perspective view that can be
                changed with ordinary mouse control.
                This can be useful for noticing features
                or bugs/issues that are harder to detect
                with a single frozen perspective.
        """
        plt.figure(
            figsize=self.figsize,
            dpi=self.dpi,
        )
        ax = plt.axes(projection=Axes3D.name)
        plot_scatterplot3d(
            ax=ax,
            Xs=Xs,
            in1=in1,
            in2=in2,
            in3=in3,
            lbl=lbl,
            title=title,
            pointsize=1.1, # todo review
            sampleset_colors=self.sampleset_colors,
        )
        plt.savefig(filename)
        if show:
            plt.show()
        plt.close()





    def lrlosscurves_rev1(
            self,
            filename,
            title,
            caption,
            X,
            Y,
            endX,
            endY,
            tolerance,
            ic_constraints,
    ):
        """
        Legacy method in Figure
        that harkens back to when it was
        part of PyPinnch. It is "stuck" here
        to keep PyPinnch dependencies
        free of any plotting libraries, except via QueueG.

        Create a figure displaying:

            - loss curve (loss wrt iteration)
            - averages of this curve
            - breakdown in loss due to contributions from constraints, ic's
            - progress of learning rate (lr).
            - progress of tolerance cutoff, if any. method for LossCurve probe.

        Arguments:

            filename (string):
            title (string)
            caption (string):
            X:
            Y:
            endX:
            endY:
            tolerance:
            ic_constraints:

        """
        # todo review

        # todo
        #  - fix issues with log scaling
        #  - add the stddev of the average over epochs
        #  - add the lr to the legend. (had issues with twinx.)

        # todo perhaps generalize the routine
        #  and munge for the case that it's actually supposed
        #  to be a loss curve. (?)

        # Plotting parameters

        in1 = "iterations"
        out1 = "loss"
        loss_colors = [
            "salmon", "lightyellow", "paleturquoise",
            "plum", "silver", "greenyellow",
            "royalblue", "darkorange", "seagreen"
        ]
        tolerance_colors = ["red"]
        total_avg_loss_fmt = "k"

        # yscale = "log"
        yscale = "linear"

        # plot_avg_loss_components = True
        plot_avg_loss_components = False

        avg_loss_colors = ["green", "blue", "gold", "brown"]
        # avg_loss_linewidth = 0.2 # very thin
        # avg_loss_linewidth = 0.5 # still kind of thin
        avg_loss_linewidth = 0.7 # ok

        lr_colors = ["blue"]
        lr_loss_linewidth = 0.6

        #<><><><><><><><><>

        # set some simple derived inputs
        MN = X.shape[1] - 1
        M = len(ic_constraints)
        N = MN - M
        maxiter = X.shape[0]
        labels = []
        for label in ic_constraints:
            labels.append(label)
        for i in range(N):
            labels.append(f"c{i+1}")
        labels += ["tol", "epoch", "Lavg"] # , "lr"]

        ymin = 0
        # find ymax in general, assuming there may be huge outliers
        mean = np.mean(X[:,MN])
        stddev = np.std(X[:,MN])
        nsigma = 4 # todo this value still seems to depends on the data. :/
        # for very small sigma, pad by 10% to distinguish the top of the plot
        pad = max(nsigma*stddev, mean*0.1)
        ymax = mean + pad
        if isnan(ymax):
            print("[Warn] Could not finish LossCurves plot: ymax is nan")
            return

        plt.figure(
            figsize=self.figsize,
            dpi=self.dpi,
        )

        if endX == 0:
            print("[Warn] Could not finish LossCurves plot: endX == 0")
            return
        # plot X data
        x = np.arange(0,endX)
        plt.stackplot(
            x,
            np.transpose(X[:endX,:MN]),
            colors=loss_colors,
            baseline='zero',
        )
        ax = plt.gca()
        ax.set(
            xlim=(x[0], maxiter),
            # xticks=np.arange(),
            ylim=(ymin, ymax),
            # yticks=np.arange(),
            yscale=yscale,
        )
        # plot the tolerance cutoff line
        plt.hlines(
            tolerance,
            x[0],
            maxiter,
            # linewidth=2.0,
            colors=tolerance_colors,
        )
        # provided there are Y's, continue
        if endY is not None and endY > 0:
            # Another way to set logarithmic y-axis:
            # ax.set_yscale('log', base=2)
            iter_col = MN + 1 + 4
            # assumes: kit goes max_iter, tolerance, lr, gamma
            lr_col = MN + 1 + 2
            #######################################
            # We make a small adjustment to
            # create a nicer-looking plot. We plot the epoch-dependent
            # curves at the *beginning* (instead of the end) of an epoch,
            # even though the data was collected at the end.
            # This looks better, particular for running averages.
            # To complete the curve, we use the "incomplete epoch"
            # measurement taken by the Result action.
            #######################################
            # draw vertical lines to delineate epochs:
            plt.vlines(
                Y[:endY-1,iter_col],
                ymin=ymin,
                ymax=ymax,
                colors='whitesmoke',
                linestyles='dashed',
                )
            # plot the total average loss
            plt.plot(np.hstack((np.zeros([1]), Y[:endY-1,iter_col])), Y[:endY,MN], total_avg_loss_fmt)
            if plot_avg_loss_components:
                # todo deprecated?
                # plot the components of the average (optional)
                for i in range(MN):
                    plt.plot(
                        Y[:endY,iter_col],
                        Y[:endY,i],
                        linewidth=avg_loss_linewidth,
                        color=avg_loss_colors[i]
                    )
            plt.xlabel(in1)
            plt.ylabel(out1)
            plt.legend(labels)
            ax2 = ax.twinx()
            ax2.set_ylabel("lr")
            ax2.plot(
                np.hstack((np.zeros([1]), Y[0:endY-1,iter_col])),
                Y[0:endY,lr_col],
                linewidth=lr_loss_linewidth,
                color=lr_colors[0]
            )


        # if yscale == "log":
        #     # find minimal Y to plot: take tolerance / 2
        #     tolerance = log2(tolerance)
        #     ymin = tolerance - 1
        #     ymax = log2(ymax)
        #     X[:endX,:] = np.log2(X[:endX,:])
        #     Y[:endY,:N+2] = np.log2(Y[:endY,:N+2])


        plt.title(title)
        if caption is not None and caption != "":
            # Value 0.13 or so if there is no label on x-axis.
            # Value 0.15 or so if there is a label on x-axis.
            plt.subplots_adjust(bottom=0.13)
            plt.figtext(
                x=0.5, # center alignment
                y=0.01, # fraction: if 0.0, text touches the bottom edge
                s=caption,
                horizontalalignment="center",
            )
        # plt.tight_layout()
        plt.savefig(filename)
        plt.close()




